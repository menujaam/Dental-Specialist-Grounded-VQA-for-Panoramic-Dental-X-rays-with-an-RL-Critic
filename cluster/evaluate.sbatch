#!/bin/bash
#SBATCH --job-name=dental_eval
#SBATCH --gres=gpu:a40:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
source cluster/job_header.sh
A=${ADAPTER:-outputs/dental_lora_3b/final}
N=$(basename $(dirname $A))
python scripts/06_predict.py --adapter $A --control image --out preds/${N}_image.jsonl
python scripts/06_predict.py --adapter $A --control blind --out preds/${N}_blind.jsonl
python scripts/06_predict.py --adapter $A --control swap  --out preds/${N}_swap.jsonl
python scripts/06_predict.py --adapter none               --out preds/zeroshot_image.jsonl
python scripts/06_predict.py --adapter none --control blind --out preds/zeroshot_blind.jsonl
python scripts/07_metrics.py --preds preds/${N}_image.jsonl --blind preds/${N}_blind.jsonl \
       --swap preds/${N}_swap.jsonl --out results/${N}.json
python scripts/07_metrics.py --preds preds/zeroshot_image.jsonl --blind preds/zeroshot_blind.jsonl \
       --out results/zeroshot.json
python scripts/08_gaze_eval.py --preds preds/${N}_image.jsonl --attention --adapter $A \
       --save_maps results/gaze_maps --out results/${N}_gaze.json || echo "gaze eval skipped (no Tufts gaze data)"
