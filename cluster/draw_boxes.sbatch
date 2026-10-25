#!/bin/bash
#SBATCH --job-name=draw_boxes
#SBATCH --cpus-per-task=2          # CPU-only, no GPU needed
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
# Submit from the dental_specialist folder:  sbatch cluster/draw_boxes.sbatch
source cluster/env.sh
module load Miniforge3
source activate $P/envs/dental
python scripts/12_draw_predictions.py --n 20 --out results/boxes                       # fine-tuned model
python scripts/12_draw_predictions.py --n 20 --preds preds/zeroshot_image.jsonl --out results/boxes_zeroshot   # plain Qwen
