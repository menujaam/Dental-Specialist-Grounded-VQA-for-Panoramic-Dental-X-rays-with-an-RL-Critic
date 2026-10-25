"""Replace the critic's Reroute answer with the tooth classifier's answer (no need to redo the rollouts).

For each saved rollout question, the YOLO teeth already saved in rollouts/<split>_detections.json are cropped
and classified. The classifier then answers the diagnosis questions it can handle:
  presence ("Is there X?"), which_teeth, which_quadrant, tooth_diagnosis ("finding on tooth X"), caries_depth.
Other question types get no Reroute option (the critic cannot pick it there).

GPU job (minutes).
  python scripts/17_reroute_with_classifier.py
Writes rollouts_clf/{train,val,test}.jsonl (+ detections copied) and prints, per question type, how often
the classifier's answer is right compared with the specialist's own answer.
Then train the critic on it:
  python scripts/14_train_critic.py --rollouts rollouts_clf --out results/critic_clf
"""
import argparse
import importlib.util
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dental.common import QUADRANT_NAME, read_jsonl, write_jsonl  # noqa: E402
from dental.critic import is_correct  # noqa: E402

_spec = importlib.util.spec_from_file_location("tc", ROOT / "scripts" / "16_train_tooth_classifier.py")
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)
CLASSES = tc.CLASSES
GROUP = {"caries": ["caries", "deep caries"], "deep caries": ["deep caries"],
         "periapical lesion": ["periapical lesion"], "impacted": ["impacted"]}
HANDLED = {"presence", "which_teeth", "which_quadrant", "tooth_diagnosis", "caries_depth"}


def cap(s):
    return s[:1].upper() + s[1:]


def tooth_probs(image_path, det, model, device, tooth_conf):
    """{tooth number: {class: probability}} for the detected teeth of one X-ray (best box per tooth)."""
    best = {}
    for b, t, c in det.get("teeth", []):
        if c >= tooth_conf and (t not in best or c > best[t][1]):
            best[int(t)] = (b, c)
    if not best:
        return {}
    im = Image.open(image_path).convert("RGB")
    teeth = sorted(best)
    P = tc.predict_crops(model, [tc.crop_tooth(im, best[t][0]) for t in teeth], device)
    return {t: dict(zip(CLASSES, map(float, P[i]))) for i, t in enumerate(teeth)}


def answer(row, probs, th):
    """Classifier answer phrased like the training answers: (text, confidence) or (None, 0)."""
    qt, dx, tooth = row["qtype"], row.get("diagnosis"), row.get("tooth")
    if qt not in HANDLED or not probs:
        return None, 0.0

    def score(p, d):        # "caries" questions mean caries OR deep caries
        return max(p[c] for c in GROUP[d])

    def hit(p, d):
        return any(p[c] >= th[c] for c in GROUP[d])

    if qt in ("presence", "which_teeth", "which_quadrant"):
        teeth = sorted(t for t, p in probs.items() if hit(p, dx))
        top = max(score(p, dx) for p in probs.values())
        if qt == "presence":
            return ("Yes." if teeth else "No."), (top if teeth else 1 - top)
        if not teeth:
            return "None found.", 1 - top
        conf = float(np.mean([score(probs[t], dx) for t in teeth]))
        if qt == "which_teeth":
            return ", ".join(map(str, teeth)) + ".", conf
        qs = sorted({t // 10 for t in teeth})
        return cap(", ".join(f"quadrant {q} ({QUADRANT_NAME[q]})" for q in qs)) + ".", conf
    p = probs.get(tooth)
    if p is None:                       # tooth not detected: the classifier has nothing to look at
        return None, 0.0
    if qt == "tooth_diagnosis":         # the question implies a finding: if none passes, take the most likely
        names = [c for c in CLASSES if p[c] >= th[c]] or [max(CLASSES, key=lambda c: p[c])]
        return cap(" and ".join(sorted(names))) + ".", float(np.mean([p[c] for c in names]))
    if qt == "caries_depth":
        d = p["deep caries"]
        return ("Deep." if d >= th["deep caries"] else "Not deep."), (d if d >= th["deep caries"] else 1 - d)
    return None, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="rollouts")
    ap.add_argument("--out", default="rollouts_clf")
    ap.add_argument("--classifier", default="outputs/tooth_classifier/best.pt")
    ap.add_argument("--tooth_conf", type=float, default=0.5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, th = tc.load_classifier(args.classifier, device)
    print("thresholds (from validation):", th)
    src, dst = Path(args.rollouts), Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        if not (src / f"{split}.jsonl").exists():
            continue
        rows = read_jsonl(src / f"{split}.jsonl")
        dets = json.load(open(src / f"{split}_detections.json"))
        shutil.copy(src / f"{split}_detections.json", dst / f"{split}_detections.json")
        cache = {}
        stats = defaultdict(lambda: [0, 0, 0])   # n, specialist right, classifier right
        for r in tqdm(rows, desc=split):
            iid = r["image_id"]
            if r["qtype"] in HANDLED and iid not in cache:
                cache[iid] = tooth_probs(r["image_path"], dets.get(iid, {}), model, device, args.tooth_conf)
            text, conf = answer(r, cache.get(iid, {}), th)
            r["cands"]["sample"] = r["cands"].pop("reroute", None)      # keep the old second sample, unused
            if text is None:
                r["cands"].pop("reroute", None)
            else:
                r["cands"]["reroute"] = dict(text=text, conf_first=conf, conf_mean=conf, conf_min=conf,
                                             source="tooth_classifier")
                s = stats[r["qtype"]]
                s[0] += 1
                s[1] += is_correct(r, r["cands"]["orig"]["text"])
                s[2] += is_correct(r, text)
        write_jsonl(dst / f"{split}.jsonl", rows)
        summary[split] = {k: dict(n=v[0], specialist_acc=v[1] / v[0], classifier_acc=v[2] / v[0])
                          for k, v in stats.items() if v[0]}
        print(f"\n{split}: classifier answers vs specialist answers (same questions)")
        for k, v in sorted(summary[split].items()):
            print(f"  {k:16s} n={v['n']:5d}  specialist {v['specialist_acc']:.3f}  classifier {v['classifier_acc']:.3f}")
    Path("results").mkdir(exist_ok=True)
    json.dump(summary, open("results/reroute_classifier_vs_specialist.json", "w"), indent=2)
    print(f"\nsaved {dst}/ and results/reroute_classifier_vs_specialist.json")


if __name__ == "__main__":
    main()
