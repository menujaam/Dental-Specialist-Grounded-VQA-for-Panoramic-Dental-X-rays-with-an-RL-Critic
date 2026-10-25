"""LoRA fine-tuning of Qwen2.5-VL on the dental VQA data.

Only the language-model projections get LoRA adapters; the vision tower stays frozen.
Loss is computed on the assistant answer tokens only.

  python scripts/05_finetune_lora.py --config configs/dental_config.yaml
  python scripts/05_finetune_lora.py --config configs/dental_config.yaml --model Qwen/Qwen2.5-VL-7B-Instruct --load_in_4bit
"""
import argparse
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dental.common import build_messages, load_config, read_jsonl  # noqa: E402

# LoRA on LLM linear layers only (anything whose path contains "visual" is excluded).
LLM_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


class VQADataset(Dataset):
    def __init__(self, rows, meta_dropout=0.0, train=True):
        self.rows = [r for r in rows if not r.get("eval_only") and r.get("answer")]
        self.meta_dropout, self.train = meta_dropout, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ctx = "none" if (self.train and random.random() < self.meta_dropout) else "meta"
        return {"messages": build_messages(r, context=ctx, with_image=True, include_answer=True),
                "image": Image.open(r["image_path"]).convert("RGB")}


class Collator:
    def __init__(self, processor):
        self.p = processor
        self.marker = processor.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self.misses = 0

    def __call__(self, batch):
        texts = [self.p.apply_chat_template(b["messages"], tokenize=False, add_generation_prompt=False)
                 for b in batch]
        enc = self.p(text=texts, images=[b["image"] for b in batch], padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        m = len(self.marker)
        for i, ids in enumerate(enc["input_ids"].tolist()):
            start = next((j + m for j in range(len(ids) - m, -1, -1) if ids[j:j + m] == self.marker), None)
            if start is None:
                labels[i, :] = -100
                self.misses += 1
            else:
                labels[i, :start] = -100   # mask system, image tokens and question
        enc["labels"] = labels
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dental_config.yaml")
    ap.add_argument("--model")
    ap.add_argument("--output_dir")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_train", type=int, default=None, help="debug: limit training rows")
    args = ap.parse_args()
    cfg = load_config(args.config)
    T = cfg["train"]
    model_name = args.model or T["model"]
    out_dir = args.output_dir or T["output_dir"]
    four_bit = args.load_in_4bit or T["load_in_4bit"]
    vqa = Path(cfg["paths"]["processed"]) / "vqa"
    random.seed(0)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'} | dtype {dtype}")

    processor = AutoProcessor.from_pretrained(model_name, min_pixels=cfg["image"]["min_pixels"],
                                              max_pixels=cfg["image"]["max_pixels"])
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2" if bf16_ok else "sdpa"
    except ImportError:
        attn = "sdpa"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype) if four_bit else None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation=attn, quantization_config=quant,
        device_map={"": 0} if torch.cuda.is_available() else None)
    model.config.use_cache = False
    if four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=T["lora_r"], lora_alpha=T["lora_alpha"],
                                             lora_dropout=T["lora_dropout"], target_modules=LLM_TARGETS,
                                             task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_rows = read_jsonl(vqa / "train.jsonl")
    random.shuffle(train_rows)
    if args.max_train:
        train_rows = train_rows[:args.max_train]
    val_rows = [r for r in read_jsonl(vqa / "val.jsonl") if not r.get("eval_only")]
    random.shuffle(val_rows)
    train_ds = VQADataset(train_rows, T["meta_dropout"], train=True)
    val_ds = VQADataset(val_rows[:T["eval_subset"]], train=False)
    print(f"train {len(train_ds)} | val {len(val_ds)}")

    targs = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=T["batch_size"],
        per_device_eval_batch_size=T["batch_size"], gradient_accumulation_steps=T["grad_accum"],
        learning_rate=T["lr"], num_train_epochs=T["epochs"], warmup_steps=50,
        lr_scheduler_type="cosine", logging_steps=10, eval_strategy="steps", eval_steps=T["eval_steps"],
        save_strategy="steps", save_steps=T["eval_steps"], save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        bf16=bf16_ok, fp16=not bf16_ok and torch.cuda.is_available(), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, dataloader_num_workers=2, report_to="none", seed=0)
    collator = Collator(processor)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator)
    trainer.train()
    final = Path(out_dir) / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    if collator.misses:
        print(f"WARNING: assistant marker not found in {collator.misses} samples (they had no loss)")
    print(f"Saved LoRA adapter -> {final}")


if __name__ == "__main__":
    main()
