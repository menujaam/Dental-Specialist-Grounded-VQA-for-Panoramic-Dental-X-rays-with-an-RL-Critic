#!/bin/bash
#SBATCH --job-name=live_critic
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
# Live pipeline: specialist answers, tools + tooth classifier give their versions, the PPO critic decides.
# Submit from the dental_specialist folder, e.g.:
#   sbatch cluster/live_critic.sbatch --from_test 3 --with_diagnosis "deep caries"
#   sbatch cluster/live_critic.sbatch --image my_xrays/test1.png --question "Is there caries?"
source cluster/job_header.sh
python -u scripts/18_live_critic.py "$@"
