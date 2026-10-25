#!/bin/bash
#SBATCH --job-name=rollouts
#SBATCH --gres=gpu:a40:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
# Submit from the dental_specialist folder:  sbatch cluster/rollouts.sbatch
# Quick test first:  sbatch cluster/rollouts.sbatch --n_train 40 --n_val 40 --n_test 40 --out rollouts_test
source cluster/env.sh
export TMPDIR=$P/.tmp/$SLURM_JOB_ID; mkdir -p $TMPDIR logs; trap 'rm -rf $TMPDIR' EXIT
module load Miniforge3
source activate $P/envs/dental
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/13_collect_rollouts.py "$@"
