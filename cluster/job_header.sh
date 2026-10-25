# Sourced at the start of every job.
source cluster/env.sh
export TMPDIR=$P/.tmp/$SLURM_JOB_ID          # per-job scratch, removed at exit
mkdir -p $TMPDIR logs
trap 'rm -rf $TMPDIR' EXIT
module load Miniforge3
source activate $P/envs/dental
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "(CPU job)"
