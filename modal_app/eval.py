"""Evaluate a verl GRPO checkpoint OR a stock HuggingFace model with SGLang.

Two run kinds:
  - **experiment** — pass `--checkpoint-path /checkpoints/.../global_step_<N>`.
    The FSDP shards under `actor/` are merged once via `verl.model_merger`
    into a `hf_merged/` directory (cached on the Modal volume), then SGLang
    loads from there.
  - **baseline**   — pass `--model-id Qwen/Qwen3-4B-Instruct-2507` (or any HF
    repo id). SGLang downloads + loads from HF hub directly. No merge step.

Default behaviour: run a multi-turn calibration loop on **every gage in
`configs/gages/` that is NOT in `train_config.yaml`** (the held-out set —
typically just `02338660` plus any swap-outs). Override with
`--gage-config <path>` to eval a single gage, or
`--gage-configs <a.yaml,b.yaml,...>` to eval an explicit list.

Output layout in the `hydrollm-results` Modal Volume:

    /results/<run_kind>/<run_tag>/<gage_id>.json
    /results/<run_kind>/<run_tag>/_summary.json

Where `run_tag` is `global_step_<N>` for experiments or the HF model name
for baselines. Per-gage files contain the full turn trace; `_summary.json`
collects best/final NSE per gage for quick comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from modal_app.images import train_image  # has SGLang + verl + EF5

app = modal.App("hydrollm-eval")

results_vol = modal.Volume.from_name("hydrollm-results", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("hydrollm-checkpoints", create_if_missing=True)


def _resolve_checkpoint(checkpoint_path: str) -> str:
    """If a directory was given without a global_step_N suffix, pick the latest."""
    p = Path(checkpoint_path)
    if (p / "actor").is_dir():
        return str(p)
    latest = p / "latest_checkpointed_iteration.txt"
    if latest.exists():
        step = latest.read_text().strip()
        return str(p / f"global_step_{step}")
    raise FileNotFoundError(
        f"Could not resolve verl checkpoint at {checkpoint_path}; expected "
        f"either an `actor/` subdir or `latest_checkpointed_iteration.txt`."
    )


def _select_eval_gages(gage_configs: str | None, train_config: str) -> list[str]:
    """Return the list of gage YAML paths to evaluate.

    Resolution rules (relative to /workspace):
      - explicit `gage_configs` (comma-separated paths) takes priority.
      - otherwise: every YAML in `configs/gages/` that is NOT in
        `train_config.gage_configs`.
    """
    repo_root = Path("/workspace")
    if gage_configs:
        return [p.strip() for p in gage_configs.split(",") if p.strip()]

    train_yaml = repo_root / train_config
    train_set: set[str] = set()
    if train_yaml.exists():
        import yaml
        with train_yaml.open() as f:
            tcfg = yaml.safe_load(f) or {}
        for path in tcfg.get("gage_configs", []) or []:
            train_set.add(Path(path).name)  # match by filename

    all_gages = sorted((repo_root / "configs" / "gages").glob("*.yaml"))
    selected = [
        str(p.relative_to(repo_root)) for p in all_gages
        if p.name not in train_set
    ]
    return selected


@app.function(
    image=train_image,
    gpu="H100:1",
    timeout=14400,  # 4h — multi-gage runs need more headroom
    volumes={
        "/results": results_vol,
        "/checkpoints": checkpoints_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    memory=65536,
    cpu=16.0,
)
def run_inference(
    checkpoint_path: str | None = None,
    model_id: str | None = None,
    gage_config: str | None = None,        # legacy single-gage path
    gage_configs: str | None = None,       # comma-separated explicit list
    train_config: str = "configs/train_config.yaml",
    max_turns: int = 100,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    parse_retry_limit: int = 3,
    harness: str = "v2",
):
    """Evaluate a verl checkpoint OR HF model on the held-out gage set.

    The session loop is hydrollm.session.run_session (shared with
    curc/eval.py); `harness` selects "v2" (fixed, default) or "v1" (original).

    Pass exactly one of:
        - `checkpoint_path`: a verl GRPO checkpoint dir (FSDP shards are
          merged to HF format on first run, then cached).
        - `model_id`: a HuggingFace model id (e.g. `Qwen/Qwen3-4B-Instruct-2507`).

    Gage selection (in priority order):
        - `gage_config`: legacy single-gage path (e.g. `configs/gages/02338660.yaml`).
        - `gage_configs`: comma-separated list of YAML paths.
        - default: every `configs/gages/*.yaml` not in `train_config.gage_configs`.
    """
    import logging
    import os
    import subprocess

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hydrollm.eval")

    os.chdir("/workspace")
    sys.path.insert(0, "/workspace/src")

    from hydrollm.config import load_gage_config
    from hydrollm.session import run_session

    if checkpoint_path and model_id:
        raise ValueError("Pass either --checkpoint-path OR --model-id, not both.")
    if not checkpoint_path and not model_id:
        checkpoint_path = "/checkpoints/qwen3-4b-grpo"

    # 1) Resolve which gages to evaluate.
    if gage_config:
        gage_paths = [gage_config]
    else:
        gage_paths = _select_eval_gages(gage_configs, train_config)
    if not gage_paths:
        raise RuntimeError("No gages selected for evaluation.")
    logger.info("Evaluating %d gage(s): %s", len(gage_paths), gage_paths)

    # 2) Resolve the model path SGLang will load.
    if model_id:
        run_kind = "baseline"
        run_tag = model_id.split("/")[-1]
        sglang_model_path = model_id
        logger.info("Baseline HF model: %s", model_id)
    else:
        run_kind = "experiment"
        ckpt_dir = _resolve_checkpoint(checkpoint_path)
        actor_dir = f"{ckpt_dir}/actor"
        merged_dir = f"{ckpt_dir}/hf_merged"
        logger.info("Checkpoint: %s", ckpt_dir)
        logger.info("Actor:      %s", actor_dir)
        logger.info("Merged HF:  %s", merged_dir)

        safetensors_marker = Path(merged_dir) / "model.safetensors.index.json"
        single_safetensors = Path(merged_dir) / "model.safetensors"
        if not (safetensors_marker.exists() or single_safetensors.exists()):
            logger.info("Merging FSDP shards via `python -m verl.model_merger` ...")
            Path(merged_dir).mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "python", "-m", "verl.model_merger", "merge",
                    "--backend", "fsdp",
                    "--local_dir", actor_dir,
                    "--target_dir", merged_dir,
                ],
                check=True,
            )
            checkpoints_vol.commit()
        else:
            logger.info("Reusing existing merged HF model at %s", merged_dir)
        run_tag = Path(ckpt_dir).name  # e.g. global_step_50
        sglang_model_path = merged_dir

    # 3) Output dir: /results/<run_kind>/<run_tag>/
    out_dir = Path("/results") / run_kind / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) Boot SGLang once and reuse across gages.
    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        sglang_model_path, trust_remote_code=True, extra_special_tokens={},
    )
    logger.info("Starting SGLang engine ...")
    engine = sgl.Engine(
        model_path=sglang_model_path,
        tokenizer_path=sglang_model_path,
        trust_remote_code=True,
        tool_call_parser="hermes",
        mem_fraction_static=0.85,
        log_level="warning",
    )

    sampling_params = {
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "stop": ["<|im_end|>"],
    }

    def generate(prompt: str) -> str:
        return engine.generate([prompt], sampling_params)[0]["text"]

    summary: list[dict] = []

    try:
        for gpath in gage_paths:
            full_path = gpath if gpath.startswith("/") else f"/workspace/{gpath}"
            try:
                gcfg = load_gage_config(full_path)
            except Exception as e:
                logger.error("Failed to load %s: %s — skipping", gpath, e)
                continue

            logger.info("=" * 60)
            logger.info("HydroLLM SGLang Eval — gage %s", gcfg.gage_id)
            logger.info("Run kind=%s  Run tag=%s  Max turns=%d", run_kind, run_tag, max_turns)
            logger.info("=" * 60)

            record = run_session(
                generate, tokenizer, gcfg, harness=harness, max_turns=max_turns,
                max_tokens=max_tokens, context_len=8192, parse_retry_limit=parse_retry_limit,
            )
            elapsed = record["elapsed_seconds"]
            result = {
                "run_kind": run_kind,
                "run_tag": run_tag,
                "model_id": model_id,
                "checkpoint_path": checkpoint_path,
                "sglang_model_path": sglang_model_path,
                "temperature": temperature,
                **record,
            }

            out_path = out_dir / f"{gcfg.gage_id}.json"
            out_path.write_text(json.dumps(result, indent=2, default=str))
            results_vol.commit()
            summary.append({
                "gage_id": gcfg.gage_id,
                "best_nse": result["best_nse"],
                "final_nse": result["final_nse"],
                "num_simulation_runs": result["num_simulation_runs"],
                "elapsed_seconds": result["elapsed_seconds"],
                "target_met": result["target_met"],
                "path": str(out_path),
            })

            logger.info(
                "[%s] best_nse=%s  sim_runs=%d  elapsed=%.1fs",
                gcfg.gage_id, result["best_nse"],
                result["num_simulation_runs"], elapsed,
            )

    finally:
        try:
            engine.shutdown()
        except Exception:
            pass

    summary_doc = {
        "run_kind": run_kind,
        "run_tag": run_tag,
        "model_id": model_id,
        "checkpoint_path": checkpoint_path,
        "n_gages": len(summary),
        "results": summary,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary_doc, indent=2, default=str))
    results_vol.commit()

    logger.info("=" * 60)
    logger.info("Eval done. %d gage(s) → %s", len(summary), out_dir)
    for r in summary:
        logger.info("  %s  best_nse=%s  sim_runs=%d", r["gage_id"], r["best_nse"], r["num_simulation_runs"])
    return summary_doc


def _download_results_to_host(summary: dict, local_root: str = "results") -> None:
    """Pull per-gage JSON + _summary.json from the Modal volume to the host.

    Mirrors the volume layout: `<local_root>/<run_kind>/<run_tag>/<gage_id>.json`.
    """
    vol = modal.Volume.from_name("hydrollm-results")
    run_kind = summary.get("run_kind", "unknown")
    run_tag = summary.get("run_tag", "unknown")
    out_dir = Path(local_root) / run_kind / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def _pull(remote_rel: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("wb") as f:
            for chunk in vol.read_file(remote_rel):
                f.write(chunk)

    # Per-gage JSONs.
    for entry in summary.get("results", []):
        remote_abs = entry.get("path", "")
        rel = remote_abs.lstrip("/")
        if rel.startswith("results/"):
            rel = rel[len("results/"):]
        if not rel:
            continue
        gage_id = entry.get("gage_id", "result")
        _pull(rel, out_dir / f"{gage_id}.json")

    # Roll-up summary.
    summary_rel = f"{run_kind}/{run_tag}/_summary.json"
    _pull(summary_rel, out_dir / "_summary.json")

    print(f"Saved {len(summary.get('results', []))} per-gage JSON(s) + _summary.json to {out_dir}/")


@app.local_entrypoint()
def main(
    checkpoint_path: str = "",
    model_id: str = "",
    gage_config: str = "",
    gage_configs: str = "",
    train_config: str = "configs/train_config.yaml",
    max_turns: int = 50,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    parse_retry_limit: int = 3,
    download_dir: str = "results",
    harness: str = "v2",
):
    """Run an SGLang eval on the held-out gage set.

    Each per-gage JSON contains the full trajectory in `turn_details` (every
    set_parameters / run_simulation / evaluate call with the NSE value at
    each evaluate). After the remote run returns, all per-gage JSONs plus
    `_summary.json` are downloaded into `./<download_dir>/<run_kind>/<run_tag>/`
    on the host.

    Examples:
        # Default: latest verl checkpoint, all non-training gages
        modal run modal_app/eval.py

        # Specific verl step, all non-training gages
        modal run modal_app/eval.py \
            --checkpoint-path /checkpoints/qwen3-4b-grpo-anchored/global_step_50

        # Baseline HF model, all non-training gages
        modal run modal_app/eval.py --model-id Qwen/Qwen3-4B-Instruct-2507

        # Single gage override (e.g. the canonical test gage)
        modal run modal_app/eval.py --gage-config configs/gages/02338660.yaml
    """
    summary = run_inference.remote(
        checkpoint_path=checkpoint_path or None,
        model_id=model_id or None,
        gage_config=gage_config or None,
        gage_configs=gage_configs or None,
        train_config=train_config,
        max_turns=max_turns,
        temperature=temperature,
        max_tokens=max_tokens,
        parse_retry_limit=parse_retry_limit,
        harness=harness,
    )
    print(json.dumps(summary, indent=2, default=str))

    try:
        _download_results_to_host(summary, local_root=download_dir)
    except Exception as e:
        print(f"WARNING: result download failed ({e}); pull manually with "
              f"`modal volume get hydrollm-results <run_kind>/<run_tag>/`.")
