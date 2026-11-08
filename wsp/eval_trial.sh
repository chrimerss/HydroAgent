#!/bin/bash
# Freeze a finished trial's edit, build its matched random control, and run
# both on the held-out gages with the baseline's protocol (curc/eval.py,
# greedy, 100 turns, sequential). Run inside the trial's GPU allocation:
#
#   setsid srun --jobid=<alloc> --overlap --ntasks=1 --cpus-per-task=16 \
#       --gres=gpu:h200_3g.71gb:1 bash wsp/eval_trial.sh trial_2 2 > <log> 2>&1 &
#
# Results: results/wsp/<trial>/ and results/wsp/<trial>_control/
set -euo pipefail
T="$1"; SEED="$2"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source curc/env.sh
activate_hydro_env
export PYTHONPATH="$REPO"
R="$HYDRO_ROOT/wsp_runs"
GAGES=02338660,01403060,06279500,07144100

[ -f "$R/$T/final_ckpt/.complete" ] || python wsp/finalize.py --trial "$T" --seed "$SEED"

for tag in "$T" "${T}_control"; do
    model="$R/$tag/final_ckpt"
    summary="results/wsp/$tag/_summary.json"
    if [ -f "$summary" ] && python -c "import json,sys; sys.exit(json.load(open('$summary'))['n_gages'] != 4)"; then
        echo "$tag already evaluated"; continue
    fi
    echo "== evaluating $tag ($model)"
    python curc/eval.py --model "$model" --gages "$GAGES" --run-kind wsp --run-tag "$tag" \
        --max-turns 100 --out-dir results
done
echo "== held-out evaluation of $T done"
