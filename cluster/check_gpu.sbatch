#!/bin/bash
#SBATCH --job-name=dental_check
#SBATCH --gres=gpu:a40:1           # gpu:<model>:<count>; never set --cpus-per-task/--mem on GPU jobs
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
source cluster/job_header.sh
python -c "import torch; print(torch.cuda.get_device_name(0), 'bf16:', torch.cuda.is_bf16_supported())"
python scripts/05_finetune_lora.py --max_train 64 --output_dir outputs/debug
python scripts/06_predict.py --adapter outputs/debug/final --limit 20 --out preds/debug.jsonl
python scripts/07_metrics.py --preds preds/debug.jsonl
