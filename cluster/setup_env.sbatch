#!/bin/bash
#SBATCH --job-name=dental_setup
#SBATCH --cpus-per-task=8          # CPU-only job; installs must NOT run on the login node
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
# Put this file in dental_specialist/cluster/ and submit ONCE from the project folder:
#   mkdir -p logs && sbatch cluster/setup_env.sbatch
set -e
source cluster/env.sh
export TMPDIR=$P/.tmp/$SLURM_JOB_ID; mkdir -p $TMPDIR; trap 'rm -rf $TMPDIR' EXIT
module load Miniforge3
[ -d $P/envs/dental ] || conda create -y -p $P/envs/dental python=3.11
source activate $P/envs/dental
# cu128 wheels work on every GPU model in the cluster, incl. RTX 5090 / Pro 6000 (Blackwell)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install ultralytics huggingface_hub
# model weights + dataset go to the SSD project dir
hf download Qwen/Qwen2.5-VL-3B-Instruct
# hf download Qwen/Qwen2.5-VL-7B-Instruct
hf download ibrahimhamamci/DENTEX --repo-type dataset --local-dir data/raw/DENTEX
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
echo "SETUP DONE. Check data/raw/DENTEX for .zip/.tar.gz archives to extract."
