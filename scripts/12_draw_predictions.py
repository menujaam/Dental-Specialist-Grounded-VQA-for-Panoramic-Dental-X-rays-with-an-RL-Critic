"""Draw the model's saved boxes (red) and the expert boxes (green) on test X-rays.

No GPU needed: it reads the predictions already saved by the evaluation.

  python scripts/12_draw_predictions.py                       # 20 examples from your fine-tuned model
  python scripts/12_draw_predictions.py --n 40 --qtype ground_diagnosis
  python scripts/12_draw_predictions.py --preds preds/zeroshot_image.jsonl --out results/boxes_zeroshot

Pictures are saved to results/boxes/ (one PNG per question), plus an index.txt listing
each picture's question, the expert answer and the model's answer.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import iou, parse_boxes  # noqa: E402

GROUND = ["ground_all", "ground_diagnosis", "ground_tooth"]


def font(size):
    for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw(row, out_path):
    im = Image.open(row["image_path"]).convert("RGB")
    d = ImageDraw.Draw(im)
    fs = max(14, im.width // 70)
    f = font(fs)
    gold = row.get("gold_boxes") or []
    pred = parse_boxes(row.get("pred"))
    for box, lab in gold:                                   # expert = green
        d.rectangle(box, outline=(0, 220, 0), width=4)
        d.text((box[0], box[3] + 2), lab, fill=(0, 220, 0), font=f)
    for box, lab in pred:                                   # model = red
        d.rectangle(box, outline=(255, 50, 50), width=4)
        d.text((box[0], max(0, box[1] - fs - 2)), lab, fill=(255, 50, 50), font=f)
    # header strip with the question and a match summary
    hits = sum(any(iou(p, g[0]) >= 0.5 for g in gold) for p, _ in pred)
    head = Image.new("RGB", (im.width, fs * 3 + 12), (20, 20, 20))
    hd = ImageDraw.Draw(head)
    hd.text((8, 4), "Q: " + row["question"], fill=(255, 255, 255), font=f)
    hd.text((8, fs + 8), f"GREEN = expert ({len(gold)})   RED = model ({len(pred)})   "
                         f"model boxes overlapping an expert box >= 50%: {hits}", fill=(220, 220, 220), font=f)
    canvas = Image.new("RGB", (im.width, im.height + head.height))
    canvas.paste(head, (0, 0))
    canvas.paste(im, (0, head.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds/dental_lora_3b_image.jsonl")
    ap.add_argument("--qtype", choices=GROUND, help="only this grounding question type")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/boxes")
    ap.add_argument("--only_with_boxes", action="store_true",
                    help="only questions where the model actually output at least one box")
    ap.add_argument("--perfect", action="store_true",
                    help="only answers where every expert box is matched (IoU>=0.5) and there are no extra boxes")
    ap.add_argument("--min_gold", type=int, default=1, help="with --perfect: at least this many expert boxes")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.preds)]
    rows = [r for r in rows if r["qtype"] in ([args.qtype] if args.qtype else GROUND) and r.get("gold_boxes")]
    if args.perfect:
        def perfect(r):
            pb = [b for b, _ in parse_boxes(r.get("pred"))]
            gold = [g[0] for g in r["gold_boxes"]]
            return (len(pb) == len(gold) >= args.min_gold
                    and all(any(iou(g, p) >= 0.5 for p in pb) for g in gold)
                    and all(any(iou(p, g) >= 0.5 for g in gold) for p in pb))
        n_all = len(rows)
        rows = [r for r in rows if perfect(r)]
        print(f"perfect answers (every expert box found, no extra boxes): {len(rows)} of {n_all}")
        rows.sort(key=lambda r: -len(r["gold_boxes"]))      # most findings first
    if args.only_with_boxes:
        n_all = len(rows)
        rows = [r for r in rows if parse_boxes(r.get("pred"))]
        print(f"model output boxes on {len(rows)} of {n_all} grounding questions")
    if not args.perfect:
        random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]
    out = Path(args.out)
    lines = []
    for i, r in enumerate(rows):
        p = out / f"{i:02d}_{r['qtype']}_{r['image_id']}.png"
        draw(r, p)
        lines += [p.name, "  Q:      " + r["question"], "  expert: " + str(r["answer"])[:300],
                  "  model:  " + str(r["pred"])[:300], ""]
        print("saved", p)
    (out / "index.txt").write_text("\n".join(lines))
    print(f"\n{len(rows)} pictures in {out}/  (open index.txt for the questions and answers)")


if __name__ == "__main__":
    main()
