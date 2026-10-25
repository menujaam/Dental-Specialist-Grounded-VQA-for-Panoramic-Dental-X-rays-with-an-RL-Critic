#!/bin/bash
#SBATCH --job-name=llavamed_setup
#SBATCH --cpus-per-task=8          # CPU-only job (installs must not run on the login node)
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
# Submit once from the dental_specialist folder:  sbatch cluster/setup_llavamed.sbatch
# Creates a SEPARATE environment, so the working `dental` environment is not touched.
set -e
source cluster/env.sh
export TMPDIR=$P/.tmp/$SLURM_JOB_ID; mkdir -p $TMPDIR; trap 'rm -rf $TMPDIR' EXIT
module load Miniforge3
[ -d $P/envs/llavamed ] || conda create -y -p $P/envs/llavamed python=3.10
source activate $P/envs/llavamed
mkdir -p $P/src
[ -d $P/src/LLaVA-Med ] || git clone https://github.com/microsoft/LLaVA-Med.git $P/src/LLaVA-Med
cd $P/src/LLaVA-Med
pip install --upgrade pip
pip install -e .                  # installs LLaVA-Med with the library versions it was built for
pip install pyyaml tqdm protobuf sentencepiece
hf download microsoft/llava-med-v1.5-mistral-7b || huggingface-cli download microsoft/llava-med-v1.5-mistral-7b
python -c "import torch, transformers, llava; print('torch', torch.__version__, '| transformers', transformers.__version__, '| llava OK')"
echo "LLAVAMED SETUP DONE"
