#!/bin/bash
#SBATCH --job-name=critic
#SBATCH --cpus-per-task=8          # CPU only: the critic is a small network over logged features
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
# Submit from the dental_specialist folder:  sbatch cluster/critic.sbatch
source cluster/env.sh
module load Miniforge3
source activate $P/envs/dental
mkdir -p results/critic
# critic trained on train-split rollouts (as in the synopsis) + accuracy-cost curve over beta
python scripts/14_train_critic.py --train_on train --beta_sweep 0 0.05 0.1 0.2 "$@"
# same critic trained on held-out validation rollouts (the specialist never saw these questions)
python scripts/14_train_critic.py --train_on val "$@"
