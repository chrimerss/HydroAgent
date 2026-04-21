#!/bin/bash
# Run baseline evaluation for all 3 Qwen2.5 models on Modal
# Usage: bash scripts/run_baseline.sh

set -e

echo "=== HydroLLM Baseline Evaluation ==="
echo "Running all 3 Qwen2.5 models on gage 02338660..."
echo ""

# Option 1: Run all models in a single Modal function (sequential)
modal run modal_app/baseline.py --all \
    --gage-config configs/gages/02338660.yaml \
    --max-turns 10

echo ""
echo "=== Baseline evaluation complete ==="
echo "Results saved to Modal Volume 'hydrollm-results'"
echo "Download with: modal volume get hydrollm-results baseline_report_02338660.md"
