"""Collect offline rollouts for the RL critic (synopsis 4.4): for each question, the fine-tuned specialist's
greedy answer WITH token confidences, a second sampled answer (Reroute candidate), and the YOLO detections
of the image (used to build the Reground candidate).

GPU job. Outputs (per split):
  rollouts/<split>.jsonl          one line per question with cands.orig / cands.reroute (+ confidences)
  rollouts/<split>_detections.json  image_id -> {"disease": [[box, class, conf]], "teeth": [[box, tooth, conf]]}

  python scripts/13_collect_rollouts.py --splits train val test --n_train 4000 --n_val 1500
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import build_messages, load_config, read_jsonl, write_jsonl  # noqa: E402

KEEP = ["qid", "image_id", "image_path", "source", "split", "width", "height", "midline_x", "occlusal_y",
        "qtype", "level", "question", "answer", "label", "options", "diagnosis", "tooth", "gold_boxes",
        "gold_teeth", "meta", "hide_meta"]


def stratified(rows, n, seed):
    """Sample n questions, spread evenly over question types (report questions excluded)."""
    by = defaultdict(list)
    for r in rows:
        if r["qtype"] != "report":
            by[r["qtype"]].append(r)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    if n is None:
        return [r for v in by.values() for r in v]
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by.values()):
        for v in by.values():
            if i < len(v) and len(out) < n:
                out.append(v[i])
        i += 1
    return out


def generate(model, processor, rows, sample, max_new_tokens, batch_size):
    """Returns [(text, conf_first, conf_mean, conf_min)] for each row."""
    tok = processor.tokenizer
    eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id, tok.pad_token_id}
    out = []
    for b in tqdm(range(0, len(rows), batch_size), desc="sampled" if sample else "greedy"):
        batch = rows[b:b + batch_size]
        texts = [processor.apply_chat_template(build_messages(r, "meta", with_image=True),
                                               tokenize=False, add_generation_prompt=True) for r in batch]
        imgs = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        enc = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new_tokens, output_scores=True,
                               return_dict_in_generate=True, **kw)
            lp = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)
        gen = g.sequences[:, enc["input_ids"].shape[1]:]
        for i in range(len(batch)):
            ids = gen[i].tolist()
            n = len(ids)
            for k, t in enumerate(ids):              # keep tokens up to and including the first end token
                if t in eos_ids:
                    n = k + 1
                    break
            p = lp[i, :max(n, 1)].float().exp().clamp(0, 1)
            text = processor.decode(gen[i, :n], skip_special_tokens=True).strip()
            out.append((text, float(p[0]), float(p.mean()), float(p.min())))
    return out


def run_detectors(image_paths, weights, imgsz=1024):
    from ultralytics import YOLO
    dis, enum = YOLO(weights["disease"]), YOLO(weights["enumeration"])
    out = {}
    for iid, path in tqdm(image_paths.items(), desc="detectors"):
        d = dis.predict(path, imgsz=imgsz, conf=0.05, verbose=False)[0]
        e = enum.predict(path, imgsz=imgsz, conf=0.05, verbose=False)[0]
        out[iid] = {
            "disease": [[b, d.names[int(c)], float(s)] for b, c, s in
                        zip(d.boxes.xyxy.tolist(), d.boxes.cls.tolist(), d.boxes.conf.tolist())],
            "teeth": [[b, int(e.names[int(c)]), float(s)] for b, c, s in
                      zip(e.boxes.xyxy.tolist(), e.boxes.cls.tolist(), e.boxes.conf.tolist())],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    ap.add_argument("--adapter", default="outputs/dental_lora_3b/final")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=1500)
    ap.add_argument("--n_test", type=int, default=None, help="default: all test questions")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--disease_weights", default="runs/detect/outputs/dentex_disease_yolo/weights/best.pt")
    ap.add_argument("--teeth_weights", default="runs/detect/outputs/dentex_enumeration_yolo/weights/best.pt")
    ap.add_argument("--out", default="rollouts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    vqa = Path(cfg["paths"]["processed"]) / "vqa"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    name = cfg["train"]["model"]
    processor = AutoProcessor.from_pretrained(name, min_pixels=cfg["image"]["min_pixels"],
                                              max_pixels=cfg["image"]["max_pixels"])
    processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    n_for = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for split in args.splits:
        rows = stratified(read_jsonl(vqa / f"{split}.jsonl"), n_for[split], args.seed)
        print(f"{split}: {len(rows)} questions")
        greedy = generate(model, processor, rows, False, args.max_new_tokens, args.batch_size)
        sampled = generate(model, processor, rows, True, args.max_new_tokens, args.batch_size)
        out = []
        for r, g, s in zip(rows, greedy, sampled):
            rec = {k: r.get(k) for k in KEEP}
            rec["cands"] = {
                "orig": dict(text=g[0], conf_first=g[1], conf_mean=g[2], conf_min=g[3]),
                "reroute": dict(text=s[0], conf_first=s[1], conf_mean=s[2], conf_min=s[3]),
            }
            out.append(rec)
        write_jsonl(out_dir / f"{split}.jsonl", out)
        dets = run_detectors({r["image_id"]: r["image_path"] for r in rows},
                             {"disease": args.disease_weights, "enumeration": args.teeth_weights})
        json.dump(dets, open(out_dir / f"{split}_detections.json", "w"))
        print(f"saved {out_dir}/{split}.jsonl and {split}_detections.json")


if __name__ == "__main__":
    main()
