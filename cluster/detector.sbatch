#!/bin/bash
#SBATCH --job-name=dental_yolo
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
source cluster/job_header.sh
python scripts/09_train_detector.py --task disease --epochs 150
python scripts/09_train_detector.py --task enumeration --epochs 150
