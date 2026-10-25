#!/bin/bash
#SBATCH --job-name=error_analysis
#SBATCH --cpus-per-task=2          # CPU-only, no GPU needed
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
# Submit from the dental_specialist folder:  sbatch cluster/error_analysis.sbatch
source cluster/env.sh
module load Miniforge3
source activate $P/envs/dental
python scripts/15_error_analysis.py --preds preds/dental_lora_3b_image.jsonl --name v1
# after v2 is evaluated, also run:
# python scripts/15_error_analysis.py --preds preds/dental_lora_3b_v2_image.jsonl --name v2
