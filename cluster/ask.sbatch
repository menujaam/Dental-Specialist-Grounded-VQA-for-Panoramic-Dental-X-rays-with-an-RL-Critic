#!/bin/bash
#SBATCH --job-name=ask
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
# Submit from the dental_specialist folder, e.g.:
#   sbatch cluster/ask.sbatch --from_test 3
#   sbatch cluster/ask.sbatch --from_test 2 --question "Which teeth have deep caries?" --question "Is tooth 48 present?"
source cluster/env.sh
module load Miniforge3
source activate $P/envs/dental
python scripts/10_ask.py "$@"
