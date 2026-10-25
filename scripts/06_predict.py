"""Run the (fine-tuned or zero-shot) model on a VQA split and save predictions.

Controls (synopsis Table 5):
  --control image   normal input
  --control blind   no image at all (text + metadata only)
  --control swap    image replaced by another test image
Context ablation:
  --context meta | none

Examples
  python scripts/06_predict.py --adapter outputs/dental_lora_3b/final --control image --out preds/ft_image.jsonl
  python scripts/06_predict.py --adapter none --out preds/zeroshot_image.jsonl          # zero-shot baseline
"""
import argparse
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import build_messages, load_config, read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--control", choices=["image", "blind", "swap"], default="image")
    ap.add_argument("--context", choices=["meta", "none"], default="meta")
    ap.add_argument("--qtypes", nargs="*", help="only these question types")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_name = args.model or cfg["train"]["model"]
    E = cfg["eval"]

    rows = read_jsonl(Path(cfg["paths"]["processed"]) / "vqa" / f"{args.split}.jsonl")
    if args.qtypes:
        rows = [r for r in rows if r["qtype"] in args.qtypes]
    if args.limit:
        rows = rows[:args.limit]

    rng = random.Random(123)
    if args.control == "swap":
        by_img = {r["image_id"]: r for r in rows}
        ids = list(by_img)
        for r in rows:  # prefer a donor image with a different status label
            cands = [i for i in rng.sample(ids, min(20, len(ids))) if i != r["image_id"]]
            diff = [i for i in cands if by_img[i].get("label") != r.get("label")]
            r["_donor"] = by_img[(diff or cands or [r["image_id"]])[0]]["image_path"]

    processor = AutoProcessor.from_pretrained(model_name, min_pixels=cfg["image"]["min_pixels"],
                                              max_pixels=cfg["image"]["max_pixels"])
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16, device_map="auto")
    if args.adapter and args.adapter.lower() != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    preds = []
    for b in tqdm(range(0, len(rows), E["batch_size"])):
        batch = rows[b:b + E["batch_size"]]
        with_img = args.control != "blind"
        texts = [processor.apply_chat_template(build_messages(r, args.context, with_image=with_img),
                                               tokenize=False, add_generation_prompt=True) for r in batch]
        if with_img:
            imgs = [Image.open(r.get("_donor", r["image_path"])).convert("RGB") for r in batch]
            enc = processor(text=texts, images=imgs, padding=True, return_tensors="pt")
        else:
            enc = processor(text=texts, padding=True, return_tensors="pt")
        enc = enc.to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=E["max_new_tokens"], do_sample=False)
        dec = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, d in zip(batch, dec):
            p = {k: v for k, v in r.items() if k != "_donor"}
            p.update(pred=d.strip(), control=args.control, context=args.context,
                     adapter=args.adapter, model=model_name)
            preds.append(p)
    write_jsonl(args.out, preds)
    print(f"Saved {len(preds)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
