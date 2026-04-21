"""Modal entrypoint for evaluating trained HydroLLM models.

Compares trained models against base models on calibration tasks.

Usage:
    modal run modal_app/eval.py --model-path /checkpoints/qwen2.5-7b/final
    modal run modal_app/eval.py --model-path /checkpoints/qwen2.5-72b/final
"""

from __future__ import annotations

import json
import modal

from modal_app.images import eval_image

app = modal.App("hydrollm-eval")
checkpoint_vol = modal.Volume.from_name("hydrollm-checkpoints")
results_vol = modal.Volume.from_name("hydrollm-results", create_if_missing=True)


@app.function(
    image=eval_image,
    gpu="H100:2",
    timeout=7200,
    volumes={
        "/checkpoints": checkpoint_vol,
        "/results": results_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    memory=65536,
)
def evaluate_model(
    model_path: str,
    gage_config: str = "configs/gages/02338660.yaml",
    max_turns: int = 10,
):
    """Evaluate a trained model on a calibration task.

    Args:
        model_path: Path to trained model checkpoint (on Modal Volume).
        gage_config: Path to gage YAML config.
        max_turns: Maximum calibration turns.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hydrollm.eval")

    from hydrollm.config import load_gage_config
    from hydrollm.baseline import run_baseline

    gcfg = load_gage_config(f"/app/{gage_config}")

    logger.info("Evaluating trained model: %s", model_path)
    result = run_baseline(
        model_name=model_path,
        gage_config=gcfg,
        max_turns=max_turns,
    )

    # Save results
    model_name = model_path.rstrip("/").split("/")[-2]
    output_path = f"/results/eval_{model_name}_{gcfg.gage_id}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    results_vol.commit()

    logger.info("Best NSE: %s (target: %s)", result["best_nse"], gcfg.target_nse)
    return result


@app.local_entrypoint()
def main(
    model_path: str,
    gage_config: str = "configs/gages/02338660.yaml",
    max_turns: int = 10,
):
    """Local entrypoint for model evaluation."""
    result = evaluate_model.remote(
        model_path=model_path,
        gage_config=gage_config,
        max_turns=max_turns,
    )
    print(json.dumps(result, indent=2, default=str))
