"""Compare model heatmaps with Tufts radiologist gaze maps (CC, NSS, KL).

Uses the Tufts test rows of a predictions file (from 06_predict.py).
  CPU only (box heatmaps + centre-bias baseline):
    python scripts/08_gaze_eval.py --preds preds/ft_image.jsonl --out results/gaze.json
  + VLM attention heatmaps (GPU):
    python scripts/08_gaze_eval.py --preds preds/ft_image.jsonl --attention --adapter outputs/dental_lora_3b/final
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import build_messages, load_config, parse_boxes, read_jsonl  # noqa: E402
from dental.gaze import all_metrics, boxes_heatmap, centre_bias, load_gaze, resize_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--qtypes", nargs="*", default=["ground_abnormality", "abnormality"])
    ap.add_argument("--attention", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--save_maps", help="folder to save example heatmaps (.npy)")
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = load_config(args.config)
    G = cfg["gaze"]
    rows = [r for r in read_jsonl(args.preds) if r.get("gaze_path") and r["qtype"] in args.qtypes]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("No rows with gaze maps found (are Tufts gaze paths set and in this split?).")
    print(f"{len(rows)} rows with gaze maps")

    if args.attention:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from dental.gaze import attention_heatmap
        name = args.model or cfg["train"]["model"]
        proc = AutoProcessor.from_pretrained(name, min_pixels=cfg["image"]["min_pixels"],
                                             max_pixels=cfg["image"]["max_pixels"])
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
        if args.adapter.lower() != "none":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    scores = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        size = (r["width"], r["height"])
        gaze = load_gaze(r["gaze_path"], size)
        maps = {"centre_bias": centre_bias(size)}
        pb = [b for b, _ in parse_boxes(r["pred"])]
        if pb:
            maps["pred_boxes"] = boxes_heatmap(pb, size)
        if r["gold_boxes"]:
            maps["gold_mask_boxes"] = boxes_heatmap([b for b, _ in r["gold_boxes"]], size)  # reference
        if args.attention:
            img = Image.open(r["image_path"]).convert("RGB")
            grid = attention_heatmap(model, proc, img, build_messages(r, "none"), r["pred"] or "None found.",
                                     G["last_n_layers"])
            maps["vlm_attention"] = resize_map(grid, size)
        for k, m in maps.items():
            for mk, v in all_metrics(m, gaze, G["fixation_top_percent"]).items():
                scores[k][mk].append(v)
        if args.save_maps and i < 20:
            Path(args.save_maps).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(args.save_maps) / f"{r['qid']}.npz", gaze=gaze,
                                **{k: v.astype(np.float32) for k, v in maps.items()})

    res = {k: {mk: float(np.mean(v)) for mk, v in d.items()} | {"n": len(next(iter(d.values())))}
           for k, d in scores.items()}
    print(f"{'heatmap':18s} {'CC':>7s} {'NSS':>7s} {'KL':>7s}   n")
    for k, d in res.items():
        print(f"{k:18s} {d['cc']:7.3f} {d['nss']:7.3f} {d['kl']:7.3f}   {d['n']}")
    print("(higher CC/NSS and lower KL = closer to the radiologists' gaze; compare against centre_bias)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
