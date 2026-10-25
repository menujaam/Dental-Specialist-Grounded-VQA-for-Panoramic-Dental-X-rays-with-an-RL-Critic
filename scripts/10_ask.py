"""Ask the trained dental model your own questions and see its boxes drawn on the X-ray.

Needs a GPU, so run it as a job (never on the login node):

  # 3 random test X-rays, default questions
  sbatch --job-name=ask --gres=gpu:a40:1 --time=1:00:00 --output=logs/%x-%j.out \
    --wrap='source cluster/job_header.sh && python scripts/10_ask.py --from_test 3'

  # your own question on a specific X-ray
  sbatch --job-name=ask --gres=gpu:a40:1 --time=1:00:00 --output=logs/%x-%j.out \
    --wrap='source cluster/job_header.sh && python scripts/10_ask.py --image path/to/xray.png \
            --question "Is there caries?" --question "Locate all caries and output bounding boxes."'

Answers are printed in the log; pictures with boxes are saved to results/ask/.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import load_config, read_jsonl  # noqa: E402
from dental.specialist import DentalSpecialist  # noqa: E402

DEFAULT_QUESTIONS = [
    "Is there caries in this panoramic radiograph?",
    "Which teeth have caries? Give FDI numbers.",
    "Locate all abnormal findings and label each with its diagnosis and tooth number.",
    "Write a short radiology report for this panoramic X-ray.",
    "Is tooth 29 present?",          # trick question: tooth 29 does not exist
]


def draw(image_path, boxes, gold, out_path):
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for g in gold:                    # expert boxes in green
        d.rectangle(g["box"], outline=(0, 200, 0), width=3)
        d.text((g["box"][0], g["box"][3] + 2), f"{g['diagnosis']} {g['tooth']}", fill=(0, 200, 0))
    for b in boxes:                   # model boxes in red
        d.rectangle(b["box"], outline=(255, 40, 40), width=3)
        d.text((b["box"][0], max(0, b["box"][1] - 12)), b["label"], fill=(255, 40, 40))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    ap.add_argument("--adapter", default="outputs/dental_lora_3b/final")
    ap.add_argument("--image")
    ap.add_argument("--question", action="append")
    ap.add_argument("--from_test", type=int, default=0, help="pick N random disease X-rays from the test split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with_diagnosis", help='with --from_test: only X-rays the expert marked with this, e.g. "deep caries"')
    args = ap.parse_args()
    cfg = load_config(args.config)

    items = []
    if args.from_test:
        recs = [r for r in read_jsonl(Path(cfg["paths"]["processed"]) / "splits" / "test.jsonl")
                if r["source"] == "dentex_disease"]
        if args.with_diagnosis:
            recs = [r for r in recs if any(f["diagnosis"] == args.with_diagnosis for f in r["findings"])]
            print(f"{len(recs)} test X-rays have '{args.with_diagnosis}'")
        random.Random(args.seed).shuffle(recs)
        items = [(r["image_path"], r["findings"], r["id"]) for r in recs[:args.from_test]]
    if args.image:
        items.append((args.image, [], Path(args.image).stem))
    if not items:
        sys.exit("Give --image or --from_test N")
    questions = args.question or DEFAULT_QUESTIONS

    spec = DentalSpecialist(cfg["train"]["model"], adapter=args.adapter,
                            min_pixels=cfg["image"]["min_pixels"], max_pixels=cfg["image"]["max_pixels"])
    for path, gold, name in items:
        print("=" * 90)
        print("X-ray:", path)
        if gold:
            print("Expert findings:", ", ".join(f"{g['diagnosis']} {g['tooth']}" for g in gold))
        all_boxes = []
        for q in questions:
            out = spec.answer(path, q)
            print(f"\nQ: {q}\nA: {out['answer']}")
            print(f"   confidence (first token): {out['conf_first_token']:.2f}"
                  + (f" | ACS: {out['acs_quadrant']:.2f}" if out["acs_quadrant"] is not None else ""))
            all_boxes += out["boxes_model"]
        pic = Path("results/ask") / f"{name}.png"
        draw(path, all_boxes, gold, pic)
        print(f"\nPicture with boxes (red = model, green = expert): {pic}")


if __name__ == "__main__":
    main()
