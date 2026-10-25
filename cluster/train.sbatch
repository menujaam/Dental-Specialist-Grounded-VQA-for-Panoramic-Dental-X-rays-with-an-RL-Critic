#!/bin/bash
#SBATCH --job-name=dental_train
#SBATCH --gres=gpu:a40:1           # 48 GB; enough for 3B and 7B (bf16) with batch 1
#SBATCH --time=2-00:00:00            # max 3-00:00:00
#SBATCH --output=logs/%x-%j.out
source cluster/job_header.sh
# 7B example:  sbatch cluster/train.sbatch --model Qwen/Qwen2.5-VL-7B-Instruct --output_dir outputs/dental_lora_7b
python scripts/05_finetune_lora.py "$@"
