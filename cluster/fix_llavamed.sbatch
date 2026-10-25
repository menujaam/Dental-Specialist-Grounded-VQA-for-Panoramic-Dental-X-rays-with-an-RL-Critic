#!/bin/bash
#SBATCH --job-name=fix_llavamed
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
set -e
source cluster/env.sh
module load Miniforge3
source activate $P/envs/llavamed
pip uninstall -y bitsandbytes || true
pip install "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
python -c "import torch, transformers; from llava.model.builder import load_pretrained_model; print('torch', torch.__version__, '| transformers', transformers.__version__, '| LLaVA import OK')"
