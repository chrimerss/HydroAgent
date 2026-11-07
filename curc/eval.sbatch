#!/bin/bash
# Multi-turn calibration inference on one Hopper GPU (Alpine has no H100
# partition; ah200 = H200 141 GB, same sm_90 architecture).
#
#   cd /projects/$USER/HydroAgent
#   sbatch curc/eval.sbatch                                   # base Qwen3-4B, test gages
#   MODEL=$HYDRO_MODELS/Qwen3-4B-hydro-sft GAGES=02338660 RUN_TAG=sft \
#       sbatch --export=ALL curc/eval.sbatch
#
# Results: results/<RUN_KIND>/<model dir name>/<gage>.json + _summary.json
#
#SBATCH --job-name=hydroagent-eval
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --account=ucb-general
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=curc/logs/%x_%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
source curc/env.sh
activate_hydro_env

MODEL="${MODEL:-$HYDRO_MODELS/Qwen3-4B-Instruct-2507}"
GAGES="${GAGES:-02338660,01403060,06279500,07144100}"
RUN_KIND="${RUN_KIND:-baseline}"
MAX_TURNS="${MAX_TURNS:-100}"
OUT_DIR="${OUT_DIR:-$REPO/results}"

for f in "$EF5_EXECUTABLE" "$MODEL/config.json" "$HYDROLLM_DATA_ROOT/docs/control.txt"; do
    [ -e "$f" ] || { echo "missing $f — run curc/setup.sbatch first" >&2; exit 1; }
done

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} model=$MODEL gages=$GAGES max_turns=$MAX_TURNS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python curc/eval.py \
    --model "$MODEL" \
    --gages "$GAGES" \
    --run-kind "$RUN_KIND" \
    --max-turns "$MAX_TURNS" \
    --out-dir "$OUT_DIR" \
    ${RUN_TAG:+--run-tag "$RUN_TAG"} \
    ${EXTRA_ARGS:-}
