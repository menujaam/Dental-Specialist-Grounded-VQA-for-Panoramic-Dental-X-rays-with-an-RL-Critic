# ============================================================================
# EDIT THIS ONE LINE: your SSD project directory (created with `storagemgr`).
export P=/projects/dentalvqamenujaa
# ============================================================================
# Caches and scratch on the SSD project dir (required by the NTU EEE storage guide)
export TMPDIR=$P/.tmp
export PIP_CACHE_DIR=$P/.tmp/pip CONDA_PKGS_DIRS=$P/.tmp/conda
export HF_HOME=$P/.tmp/hf TRITON_CACHE_DIR=$P/.tmp/triton XDG_CACHE_HOME=$P/.tmp/cache
export TOKENIZERS_PARALLELISM=false
mkdir -p $TMPDIR
