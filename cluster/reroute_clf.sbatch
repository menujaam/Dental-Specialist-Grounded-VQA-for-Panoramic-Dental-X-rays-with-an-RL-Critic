#!/bin/bash
#SBATCH --job-name=reroute_clf
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Submit from the dental_specialist folder:  sbatch cluster/reroute_clf.sbatch
# 1) train the tooth classifier  2) make classifier Reroute answers  3) retrain the critics with them
# (tooth crops made by an earlier run are reused, so step 1 starts straight at training)
source cluster/job_header.sh
set -e
python -u scripts/16_train_tooth_classifier.py
python -u scripts/17_reroute_with_classifier.py
python -u scripts/14_train_critic.py --rollouts rollouts_clf --out results/critic_clf
python -u scripts/14_train_critic.py --rollouts rollouts_clf --out results/critic_clf --train_on val
