"""Modal entrypoint for baseline inference evaluation.

Runs base Qwen2.5 models (without RL training) on the calibration task
to establish performance floor before GRPO training.

Usage:
    modal run modal_app/baseline.py --model-config configs/models/qwen2.5_7b.yaml
    modal run modal_app/baseline.py --model-config configs/models/qwen2.5_72b.yaml
    modal run modal_app/baseline.py --all  # Run all 3 models
"""

from __future__ import annotations

import json
import modal

from modal_app.images import eval_image

app = modal.App("hydrollm-baseline")

# Persistent volume for storing baseline results
vol = modal.Volume.from_name("hydrollm-results", create_if_missing=True)


@app.function(
    image=eval_image,
    gpu="H100:2",  # 2 GPUs for inference (even 72B fits with vLLM quantization)
    timeout=7200,  # 2 hours
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    memory=65536,
)
def run_baseline_eval(
    model_config: str = "configs/models/qwen2.5_7b.yaml",
    gage_config: str = "configs/gages/02338660.yaml",
    max_turns: int = 10,
):
    """Run baseline evaluation for a single model on a single gage.

    Args:
        model_config: Path to model YAML config.
        gage_config: Path to gage YAML config.
        max_turns: Maximum calibration turns.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hydrollm.baseline")

    from hydrollm.config import load_model_config, load_gage_config
    from hydrollm.baseline import run_baseline

    mcfg = load_model_config(f"/app/{model_config}")
    gcfg = load_gage_config(f"/app/{gage_config}")

    logger.info("=" * 60)
    logger.info("Baseline Evaluation")
    logger.info("Model: %s", mcfg.model_id)
    logger.info("Gage: %s", gcfg.gage_id)
    logger.info("Max turns: %d", max_turns)
    logger.info("=" * 60)

    result = run_baseline(
        model_name=mcfg.model_id,
        gage_config=gcfg,
        max_turns=max_turns,
    )

    # Save results
    output_path = f"/results/baseline_{mcfg.name}_{gcfg.gage_id}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    vol.commit()

    logger.info("Results saved to %s", output_path)
    logger.info("Best NSE: %s", result["best_nse"])
    logger.info("Target met: %s", result["target_met"])

    return result


@app.function(
    image=eval_image,
    gpu="H100:4",
    timeout=14400,  # 4 hours for all models
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    memory=65536,
)
def run_all_baselines(
    gage_config: str = "configs/gages/02338660.yaml",
    max_turns: int = 10,
):
    """Run baseline evaluation for all 3 Qwen2.5 models.

    Args:
        gage_config: Path to gage YAML config.
        max_turns: Maximum calibration turns.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hydrollm.baseline")

    from hydrollm.config import load_gage_config
    from hydrollm.baseline import run_baseline, format_baseline_report

    gcfg = load_gage_config(f"/app/{gage_config}")

    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ]

    all_results = []
    for model_name in models:
        logger.info("Running baseline for %s...", model_name)
        try:
            result = run_baseline(
                model_name=model_name,
                gage_config=gcfg,
                max_turns=max_turns,
            )
            all_results.append(result)
            logger.info("%s: Best NSE = %s", model_name, result["best_nse"])
        except Exception as e:
            logger.error("Failed for %s: %s", model_name, e)
            all_results.append({
                "model": model_name,
                "gage_id": gcfg.gage_id,
                "error": str(e),
            })

    # Save all results
    with open(f"/results/baseline_all_{gcfg.gage_id}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate report
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        report = format_baseline_report(valid_results)
        with open(f"/results/baseline_report_{gcfg.gage_id}.md", "w") as f:
            f.write(report)
        logger.info("\n%s", report)

    vol.commit()
    return all_results


@app.local_entrypoint()
def main(
    model_config: str = "",
    gage_config: str = "configs/gages/02338660.yaml",
    all: bool = False,
    max_turns: int = 10,
):
    """Local entrypoint for baseline evaluation."""
    if all:
        results = run_all_baselines.remote(
            gage_config=gage_config,
            max_turns=max_turns,
        )
    else:
        if not model_config:
            model_config = "configs/models/qwen2.5_7b.yaml"
        results = run_baseline_eval.remote(
            model_config=model_config,
            gage_config=gage_config,
            max_turns=max_turns,
        )
    print(json.dumps(results, indent=2, default=str))
