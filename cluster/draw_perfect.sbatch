#!/bin/bash
#SBATCH --job-name=draw_perfect
#SBATCH --cpus-per-task=2          # CPU-only, no GPU needed
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
# Submit from the dental_specialist folder:  sbatch cluster/draw_perfect.sbatch
source cluster/env.sh
module load Miniforge3
source activate $P/envs/dental
# Perfect answers: every expert box found (IoU >= 0.5) and no extra boxes. Most findings first.
python scripts/12_draw_predictions.py --perfect --n 30 --out results/boxes_perfect                            # all grounding types
python scripts/12_draw_predictions.py --perfect --min_gold 3 --n 20 --qtype ground_all --out results/boxes_perfect_all   # whole-X-ray answers with 3+ findings
