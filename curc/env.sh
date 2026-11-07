# Shared paths for running HydroAgent on CU Research Computing (Alpine).
# Sourced by curc/setup.sbatch and curc/eval.sbatch. Override any of these
# in your shell before `sbatch` to relocate things.

: "${USER:=$(id -un)}"   # unset under `srun bash -c`

# Data, model weights, and caches are re-downloadable, so they live on scratch
# (fast GPFS, 9.5 TB quota, purged after 90 days without access).
export HYDRO_ROOT="${HYDRO_ROOT:-/scratch/alpine/$USER/hydroagent}"
export HYDROLLM_DATA_ROOT="${HYDROLLM_DATA_ROOT:-$HYDRO_ROOT/data}"
export HYDRO_MODELS="${HYDRO_MODELS:-$HYDRO_ROOT/models}"

# The conda env (python + sglang + EF5 binary) lives on /projects so it is
# never purged.
export HYDRO_ENV="${HYDRO_ENV:-/projects/$USER/software/anaconda/envs/hydroagent}"
export EF5_EXECUTABLE="${EF5_EXECUTABLE:-$HYDRO_ENV/bin/ef5}"

# HF dataset commit pinned in modal_app/images.py (train_image).
export HYDRO_DATA_SHA="${HYDRO_DATA_SHA:-221a411777e82c788a951e8c2ebf2393ae034785}"

# Keep big caches out of $HOME (2 GB quota). HF_HUB_CACHE (not HF_HOME) so the
# login token in ~/.cache/huggingface/token is still found.
export HF_HUB_CACHE="$HYDRO_ROOT/cache/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_CACHE_DIR="$HYDRO_ROOT/cache/pip"
export TRITON_CACHE_DIR="$HYDRO_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$HYDRO_ROOT/cache/inductor"
export FLASHINFER_WORKSPACE_BASE="$HYDRO_ROOT/cache"
export OUTLINES_CACHE_DIR="$HYDRO_ROOT/cache/outlines"

activate_hydro_env() {
    # The miniforge module exists on acompile but not on the GPU nodes, and the
    # env is self-contained, so put it first on PATH instead of `conda activate`.
    export CONDA_PREFIX="$HYDRO_ENV"
    export PATH="$HYDRO_ENV/bin:$PATH"
    # ~/.local/lib/python3.11 holds a truncated sympy that shadows the env's.
    export PYTHONNOUSERSITE=1
    hash -r
}
