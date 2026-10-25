"""Zero-shot LLaVA-Med v1.5 on the dental VQA test set (synopsis baseline).

Writes predictions in the SAME format as 06_predict.py, so 07_metrics.py scores them unchanged.
Runs inside the separate `llavamed` environment (see cluster/setup_llavamed.sbatch).

  python scripts/11_predict_llavamed.py --control image --out preds/llavamed_image.jsonl
  python scripts/11_predict_llavamed.py --control blind --out preds/llavamed_blind.jsonl
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import SYSTEM_PROMPT, build_prompt, load_config, read_jsonl, write_jsonl  # noqa: E402

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    ap.add_argument("--model", default="microsoft/llava-med-v1.5-mistral-7b")
    ap.add_argument("--conv_mode", default="mistral_instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--control", choices=["image", "blind"], default="image")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    rows = read_jsonl(Path(cfg["paths"]["processed"]) / "vqa" / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[:args.limit]

    name = get_model_name_from_path(args.model)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model, None, name, device_map="auto")
    model.eval()
    conv_mode = args.conv_mode if args.conv_mode in conv_templates else "vicuna_v1"
    print(f"Loaded {args.model} | conv_mode={conv_mode} | {len(rows)} questions | control={args.control}")

    preds = []
    for r in tqdm(rows):
        # same instructions and question text as the Qwen runs (dental has no metadata)
        q = SYSTEM_PROMPT + "\n\n" + build_prompt(r["question"], r.get("meta"), "meta", r.get("hide_meta", []))
        if args.control == "image":
            q = DEFAULT_IMAGE_TOKEN + "\n" + q
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if args.control == "image":
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
            img = Image.open(r["image_path"]).convert("RGB")
            images = process_images([img], image_processor, model.config).to(model.device, dtype=torch.float16)
        else:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            images = None
        ids = ids.to(model.device)
        with torch.inference_mode():
            out = model.generate(ids, images=images, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 use_cache=True)
        # newer LLaVA code returns only new tokens; older returns prompt + answer
        if out.shape[1] > ids.shape[1] and torch.equal(out[0, :ids.shape[1]], ids[0]):
            out = out[:, ids.shape[1]:]
        ans = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        p = dict(r)
        p.update(pred=ans, control=args.control, context="meta", adapter="none", model=args.model)
        preds.append(p)
    write_jsonl(args.out, preds)
    print(f"Saved {len(preds)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
