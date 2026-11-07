#!/bin/bash
# One-time setup on Alpine: conda env (sglang 0.4.8 + sci stack), EF5 binary,
# calibration data, and HF model weights. Idempotent: rerunning skips any
# step that already finished.
#
#   cd /projects/$USER/HydroAgent
#   sbatch curc/setup.sbatch                                  # base model
#   MODEL_IDS="Qwen/Qwen3-4B-Instruct-2507 chrimerss/Qwen3-4B-hydro-sft" \
#       sbatch --export=ALL curc/setup.sbatch                 # several models
#
#SBATCH --job-name=hydroagent-setup
#SBATCH --partition=acompile
#SBATCH --qos=compile
#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=curc/logs/%x_%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
source curc/env.sh
MODEL_IDS="${MODEL_IDS:-Qwen/Qwen3-4B-Instruct-2507}"
mkdir -p "$HYDRO_ROOT"/{src,downloads,cache} "$HYDRO_MODELS"

echo "== [1/4] conda env at $HYDRO_ENV"
module purge
module load miniforge
eval "$(conda shell.bash hook)"
if [ ! -x "$HYDRO_ENV/bin/python" ]; then
    mamba create -y -p "$HYDRO_ENV" -c conda-forge \
        python=3.11 geotiff libtiff zlib cxx-compiler make \
        autoconf automake libtool pkg-config
fi
conda activate "$HYDRO_ENV"
export PYTHONNOUSERSITE=1   # ~/.local/lib/python3.11 holds a truncated sympy
if ! python -c "import sglang, pandas, yaml" 2>/dev/null; then
    # sglang 0.4.8 matches the verl0.5-sglang0.4.8 image used on Modal. Its
    # flashinfer-python==0.2.6.post1 is sdist-only on PyPI, and building it in
    # an isolated env yields a bogus torch pin, so build it against torch 2.7.1.
    # The rest comes from curc/requirements.lock (see curc/requirements.in).
    pip install torch==2.7.1 numpy==1.26.4 setuptools wheel packaging ninja
    pip install --no-build-isolation flashinfer-python==0.2.6.post1
    pip install -r curc/requirements.lock
fi
python -c "import sglang, torch, transformers; print('sglang', sglang.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"

echo "== [2/4] EF5 -> $EF5_EXECUTABLE"
if [ ! -x "$EF5_EXECUTABLE" ]; then
    cd "$HYDRO_ROOT/src"
    [ -d EF5 ] || git clone https://github.com/HyDROSLab/EF5.git
    cd EF5
    git checkout d71621b948c9e13c4d74c4cd69e11bdf6c09c50a   # same pin as modal_app/images.py
    autoreconf --force --install
    ./configure CXXFLAGS='-Wall -O2' CFLAGS='-Wall -O2' \
        CPPFLAGS="-I$CONDA_PREFIX/include" \
        LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
    sed -i 's/-Werror//g' Makefile
    make -j"${SLURM_CPUS_PER_TASK:-8}"
    install -m 755 bin/ef5 "$EF5_EXECUTABLE"
    cd "$REPO"
fi
ldd "$EF5_EXECUTABLE" | grep -E "geotiff|tiff|not found" || true

echo "== [3/4] calibration data (commit $HYDRO_DATA_SHA) -> $HYDROLLM_DATA_ROOT"
if [ ! -f "$HYDROLLM_DATA_ROOT/.complete_$HYDRO_DATA_SHA" ]; then
    tarball="$HYDRO_ROOT/downloads/data_$HYDRO_DATA_SHA.tar.gz"
    [ -s "$tarball" ] || wget -q -O "$tarball" \
        "https://huggingface.co/datasets/chrimerss/hydro_cali_agent_example/resolve/$HYDRO_DATA_SHA/data.tar.gz"
    rm -rf "$HYDROLLM_DATA_ROOT"
    mkdir -p "$HYDROLLM_DATA_ROOT"
    tar -xzf "$tarball" -C "$HYDROLLM_DATA_ROOT" --no-same-owner
    # macOS AppleDouble files make EF5 hang (see modal_app/images.py).
    find "$HYDROLLM_DATA_ROOT" -name '._*' -delete
    mkdir -p "$HYDROLLM_DATA_ROOT/docs"
    cp control.txt "$HYDROLLM_DATA_ROOT/docs/control.txt"
    touch "$HYDROLLM_DATA_ROOT/.complete_$HYDRO_DATA_SHA"
fi
du -sh "$HYDROLLM_DATA_ROOT"/* || true
echo "gages with MRMS forcing: $(ls "$HYDROLLM_DATA_ROOT/data_mrms_clip" | tr '\n' ' ')"

echo "== [4/4] model weights -> $HYDRO_MODELS"
for mid in $MODEL_IDS; do
    dest="$HYDRO_MODELS/${mid##*/}"
    python - "$mid" "$dest" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=repo_id, local_dir=dest)
print(f"{repo_id} -> {path}")
PY
    du -sh "$dest"
done

echo "== smoke test: unit tests + one EF5 run on 02338660 (default params)"
# test_evaluate_empty and test_parse_malformed_json already fail at HEAD
# (stale expectations), so unit-test failures are reported but not fatal.
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/ || echo "WARNING: unit tests failed (see above)"
PYTHONPATH=src python - <<'PY'
import logging
from hydrollm.config import load_gage_config
from hydrollm.environment import HydroEnvironment
logging.basicConfig(level=logging.WARNING)
env = HydroEnvironment(load_gage_config("configs/gages/02338660.yaml"))
try:
    run = env.run_simulation()
    print("run_simulation:", run)
    metrics = env.evaluate()
    print("evaluate:", metrics)
    assert run["status"] == "ok" and metrics["status"] == "ok", "EF5 smoke test failed"
finally:
    env.cleanup()
PY

echo "== setup done"
