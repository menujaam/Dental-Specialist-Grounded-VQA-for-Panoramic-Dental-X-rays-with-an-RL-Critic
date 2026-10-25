#!/bin/bash
#SBATCH --job-name=dental_prep
#SBATCH --cpus-per-task=8          # CPU-only jobs are free (3 GiB RAM per CPU)
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
source cluster/job_header.sh
python scripts/00_inspect_dataset.py data/raw/DENTEX
python scripts/01_prepare_dentex.py
python scripts/02_prepare_tufts.py || echo "Tufts not available - continuing with DENTEX only"
python scripts/03_make_splits.py
python scripts/04_generate_vqa.py
python scripts/09_train_detector.py --prepare_only
