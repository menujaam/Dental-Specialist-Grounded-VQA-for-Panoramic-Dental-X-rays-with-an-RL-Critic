#!/bin/bash
# Run ONCE on the LOGIN node, from inside the project folder:
#   cd $P/dental_specialist && bash cluster/setup_env.sh
set -e
mkdir -p logs   # Slurm needs this folder to exist before jobs start
source cluster/env.sh
module load Miniforge3
conda create -y -p $P/envs/dental python=3.11
source activate $P/envs/dental
# cu128 wheels work on every GPU in the cluster, incl. RTX 5090 / Pro 6000 (Blackwell)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install ultralytics huggingface_hub
# download model weights + dataset once, into the SSD project dir
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
# huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
huggingface-cli download ibrahimhamamci/DENTEX --repo-type dataset --local-dir data/raw/DENTEX
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
echo "Done. If data/raw/DENTEX contains .zip/.tar.gz files, unzip them there."
