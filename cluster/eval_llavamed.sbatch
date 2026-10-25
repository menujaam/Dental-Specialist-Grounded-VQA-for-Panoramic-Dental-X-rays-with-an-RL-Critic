#!/bin/bash
#SBATCH --job-name=eval_llavamed
#SBATCH --gres=gpu:a40:1           # keep a40: LLaVA-Med's older PyTorch may not support the newest GPUs
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
# Submit from the dental_specialist folder:  sbatch cluster/eval_llavamed.sbatch
# Quick test first:                           LIMIT=20 sbatch --export=ALL cluster/eval_llavamed.sbatch
source cluster/env.sh
export TMPDIR=$P/.tmp/$SLURM_JOB_ID; mkdir -p $TMPDIR logs preds results; trap 'rm -rf $TMPDIR' EXIT
module load Miniforge3
nvidia-smi --query-gpu=name,memory.total --format=csv
L=${LIMIT:+--limit $LIMIT}

# 1) predictions with the LLaVA-Med environment
source activate $P/envs/llavamed
python scripts/11_predict_llavamed.py --control image $L --out preds/llavamed_image.jsonl
python scripts/11_predict_llavamed.py --control blind $L --out preds/llavamed_blind.jsonl
conda deactivate

# 2) scoring with the normal dental environment (same metrics as the Qwen runs)
source activate $P/envs/dental
python scripts/07_metrics.py --preds preds/llavamed_image.jsonl --blind preds/llavamed_blind.jsonl \
       --out results/llavamed.json
