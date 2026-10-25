"""Train a small tooth-crop classifier (ConvNeXt-Tiny) to use as the critic's Reroute agent.

Each tooth is found with your YOLO tooth detector, cropped, and labelled with the expert DENTEX findings
that overlap it (multi-label: caries, deep caries, periapical lesion, impacted; none = healthy).
Only disease X-rays are used, because only there did the dentists mark every finding.
Same train / val / test images as the VQA split, so test X-rays are never trained on.

GPU job (~30-60 min).
  python scripts/16_train_tooth_classifier.py
Outputs:
  outputs/tooth_classifier/best.pt          weights + per-class thresholds (chosen on validation)
  results/tooth_classifier.json             test scores per diagnosis (AUROC, F1, precision, recall)
  data/processed/tooth_crops/               the crops (PNG) and an index per split
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import iou, load_config, read_jsonl, write_jsonl  # noqa: E402

CLASSES = ["caries", "deep caries", "periapical lesion", "impacted"]
SIZE = 224
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


# ------------------------------------------------------------------ crops (shared with script 17)
def crop_tooth(im, box, pad=0.15, size=SIZE):
    """Crop a tooth box with some margin, letterbox to a square (no stretching)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - pad * w), max(0, y1 - pad * h)
    x2, y2 = min(im.width, x2 + pad * w), min(im.height, y2 + pad * h)
    c = im.crop((int(x1), int(y1), int(max(x2, x1 + 2)), int(max(y2, y1 + 2))))
    s = size / max(c.width, c.height)
    c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.BICUBIC)
    out = Image.new("RGB", (size, size))
    out.paste(c, ((size - c.width) // 2, (size - c.height) // 2))
    return out


def to_tensor(img):
    a = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32))


def build_model(pretrained=True):
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(CLASSES))
    return m


def load_classifier(path, device):
    ck = torch.load(path, map_location="cpu")
    m = build_model(pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck["thresholds"]


@torch.no_grad()
def predict_crops(model, crops, device, batch=64):
    out = []
    for i in range(0, len(crops), batch):
        x = torch.stack([to_tensor(c) for c in crops[i:i + batch]]).to(device)
        out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


# ------------------------------------------------------------------ metrics (no sklearn needed)
def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    for v in np.unique(p):                       # average ranks for ties
        idx = p == v
        ranks[idx] = ranks[idx].mean()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def best_threshold(y, p):
    if y.sum() == 0:                      # no positives to tune on: keep the neutral 0.5
        return 0.5
    best = (0.5, -1)
    for t in np.arange(0.05, 0.96, 0.05):
        f = prf(y, (p >= t).astype(int))[2]
        if f > best[1]:
            best = (float(round(t, 2)), f)
    return best[0]


# ------------------------------------------------------------------ data
def make_crops(records, yolo, crop_dir, tooth_conf):
    """YOLO teeth on each disease X-ray -> crops labelled by overlapping expert findings."""
    rows = []
    for r in tqdm(records, desc="crops"):
        im = Image.open(r["image_path"]).convert("RGB")
        e = yolo.predict(r["image_path"], imgsz=1024, conf=tooth_conf, verbose=False)[0]
        teeth = [(b, int(e.names[int(c)])) for b, c in zip(e.boxes.xyxy.tolist(), e.boxes.cls.tolist())]
        used = set()
        for k, (tb, t) in enumerate(teeth):
            lab = [0] * len(CLASSES)
            for j, f in enumerate(r["findings"]):
                fb = f["box"]
                cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
                inside = tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]
                if (iou(tb, fb) >= 0.3 or inside) and f["diagnosis"] in CLASSES:
                    lab[CLASSES.index(f["diagnosis"])] = 1
                    used.add(j)
            p = crop_dir / f"{r['id']}_t{t}_{k}.png"
            crop_tooth(im, tb).save(p)
            rows.append(dict(path=str(p), image_id=r["id"], tooth=t, label=lab))
        for j, f in enumerate(r["findings"]):         # findings no detected tooth covered (e.g. missed impacted)
            if j in used or f["diagnosis"] not in CLASSES:
                continue
            lab = [0] * len(CLASSES)
            lab[CLASSES.index(f["diagnosis"])] = 1
            p = crop_dir / f"{r['id']}_f{j}.png"
            crop_tooth(im, f["box"]).save(p)
            rows.append(dict(path=str(p), image_id=r["id"], tooth=f.get("tooth"), label=lab))
    return rows


class Crops(torch.utils.data.Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = Image.open(self.rows[i]["path"]).convert("RGB")
        if self.train:                                  # light augmentation: brightness/contrast, small rotation
            from PIL import ImageEnhance
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
            img = img.rotate(random.uniform(-8, 8))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)   # a single tooth crop: flipping keeps its diagnosis
        return to_tensor(img), torch.tensor(self.rows[i]["label"], dtype=torch.float32)


def run_eval(model, rows, device):
    model.eval()
    dl = torch.utils.data.DataLoader(Crops(rows, False), batch_size=64, num_workers=0)
    P, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            P.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
            Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    ap.add_argument("--teeth_weights", default="runs/detect/outputs/dentex_enumeration_yolo/weights/best.pt")
    ap.add_argument("--tooth_conf", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="outputs/tooth_classifier")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true", help="debug only: skip downloading ImageNet weights")
    args = ap.parse_args()
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    proc = Path(cfg["paths"]["processed"])
    crop_root = proc / "tooth_crops"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = {}
    for split in ("train", "val", "test"):
        idx = crop_root / f"{split}.jsonl"
        if idx.exists():
            data[split] = read_jsonl(idx)
            continue
        from ultralytics import YOLO
        yolo = YOLO(args.teeth_weights)
        recs = [r for r in read_jsonl(proc / "splits" / f"{split}.jsonl") if r["source"] == "dentex_disease"]
        (crop_root / split).mkdir(parents=True, exist_ok=True)
        data[split] = make_crops(recs, yolo, crop_root / split, args.tooth_conf)
        write_jsonl(idx, data[split])
    for split, rows in data.items():
        lab = np.array([r["label"] for r in rows])
        print(f"{split}: {len(rows)} tooth crops | positives per class:",
              dict(zip(CLASSES, lab.sum(0).tolist())), "| healthy:", int((lab.sum(1) == 0).sum()))

    model = build_model(pretrained=not args.no_pretrained).to(device)
    lab = np.array([r["label"] for r in data["train"]])
    pos = lab.sum(0)
    pos_weight = torch.tensor(np.clip((len(lab) - pos) / np.maximum(pos, 1), 1, 50), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    dl = torch.utils.data.DataLoader(Crops(data["train"], True), batch_size=args.batch, shuffle=True,
                                     num_workers=0, drop_last=True)  # 0 = no helper processes (they froze on the cluster)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, history = -1, []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tqdm(dl, desc=f"epoch {ep}"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = lossf(model(x.to(device)).float(), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        P, Y = run_eval(model, data["val"], device)
        aucs = [auroc(Y[:, k], P[:, k]) for k in range(len(CLASSES))]
        macro = float(np.mean([a for a in aucs if a is not None]))
        history.append(dict(epoch=ep, train_loss=tot / len(dl), val_macro_auroc=macro))
        print(f"epoch {ep}: train loss {tot / len(dl):.4f}  val macro AUROC {macro:.4f}  "
              + "  ".join(f"{c} {a:.3f}" for c, a in zip(CLASSES, aucs) if a is not None))
        if macro > best:
            best = macro
            th = {c: best_threshold(Y[:, k], P[:, k]) for k, c in enumerate(CLASSES)}
            torch.save(dict(state_dict=model.state_dict(), thresholds=th, classes=CLASSES, epoch=ep),
                       out / "best.pt")

    model, th = load_classifier(out / "best.pt", device)
    P, Y = run_eval(model, data["test"], device)
    res = dict(n_test_crops=len(Y), thresholds=th, history=history, per_class={})
    print("\n== Tooth classifier on TEST crops (thresholds chosen on validation)")
    for k, c in enumerate(CLASSES):
        p, r, f = prf(Y[:, k], (P[:, k] >= th[c]).astype(int))
        res["per_class"][c] = dict(n_pos=int(Y[:, k].sum()), auroc=auroc(Y[:, k], P[:, k]),
                                   precision=p, recall=r, f1=f)
        a = res["per_class"][c]["auroc"]
        print(f"  {c:18s} positives {int(Y[:, k].sum()):4d}  AUROC {a if a is None else round(a, 3)}  "
              f"precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")
    Path("results").mkdir(exist_ok=True)
    json.dump(res, open("results/tooth_classifier.json", "w"), indent=2)
    print("saved results/tooth_classifier.json and", out / "best.pt")


if __name__ == "__main__":
    main()
