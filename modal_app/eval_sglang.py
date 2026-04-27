"""Evaluate a verl GRPO checkpoint with SGLang inference.

Pipeline:
  1. Mount the `hydrollm-checkpoints` volume.
  2. Run verl's `scripts/model_merger.py` to consolidate the FSDP shards
     under `global_step_<N>/actor/` into a single HF-format model dir.
  3. Boot SGLang's local Engine, hand it the merged model + Hermes tool
     parser, and run a manual multi-turn calibration loop on a single
     gage (default 02338660 — the held-out test gage).

Usage:
    # Eval the latest checkpoint of the active GRPO run:
    modal run modal_app/eval_sglang.py

    # Pin to a specific step or different run:
    modal run modal_app/eval_sglang.py \
        --checkpoint-path /checkpoints/qwen3-4b-grpo/global_step_10 \
        --gage-config configs/gages/02338660.yaml \
        --max-turns 50
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from modal_app.images import train_image  # has SGLang + verl + EF5

app = modal.App("hydrollm-eval-sglang")

results_vol = modal.Volume.from_name("hydrollm-results", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("hydrollm-checkpoints", create_if_missing=True)


_EVALUATE_METRIC_KEYS = (
    "NSE", "CC", "KGE",
    "sim_peak", "obs_peak", "peak_ratio",
    "lag_hours_sim_minus_obs", "num_points",
)


def _compact_tool_result(tool_name: str, data: dict) -> str:
    status = data.get("status", "error")
    if status == "error":
        return json.dumps({"status": "error", "message": data.get("message", "")})
    if tool_name == "set_parameters":
        return json.dumps({"status": "ok", "params": data.get("validated_params", {})})
    if tool_name == "run_simulation":
        return json.dumps({
            "status": "ok",
            "run": data.get("run_number"),
            "message": "Simulation complete. Call evaluate to see metrics.",
        })
    if tool_name == "evaluate":
        payload = {"status": "ok"}
        for key in _EVALUATE_METRIC_KEYS:
            if key in data:
                payload[key] = data[key]
        return json.dumps(payload)
    return json.dumps({"status": status})


def _resolve_checkpoint(checkpoint_path: str) -> str:
    """If a directory was given without a global_step_N suffix, pick the latest."""
    p = Path(checkpoint_path)
    if (p / "actor").is_dir():
        return str(p)
    # User passed the parent run dir; pick latest_checkpointed_iteration.txt
    latest = p / "latest_checkpointed_iteration.txt"
    if latest.exists():
        step = latest.read_text().strip()
        return str(p / f"global_step_{step}")
    raise FileNotFoundError(
        f"Could not resolve verl checkpoint at {checkpoint_path}; expected "
        f"either an `actor/` subdir or `latest_checkpointed_iteration.txt`."
    )


@app.function(
    image=train_image,
    gpu="H100:1",
    timeout=7200,  # 2h
    volumes={
        "/results": results_vol,
        "/checkpoints": checkpoints_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    memory=65536,
    cpu=16.0,
)
def run_inference(
    checkpoint_path: str = "/checkpoints/qwen3-4b-grpo",
    gage_config: str = "configs/gages/02338660.yaml",
    max_turns: int = 50,
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    """Evaluate a verl checkpoint on a single gage with SGLang."""
    import logging
    import os
    import subprocess
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hydrollm.eval_sglang")

    os.chdir("/workspace")
    sys.path.insert(0, "/workspace/src")

    from hydrollm.config import load_gage_config
    from hydrollm.environment import HydroEnvironment
    from hydrollm.prompts import build_messages
    from hydrollm.tools import HYDRO_TOOLS, ToolExecutor, parse_tool_calls

    # 1) Resolve the verl checkpoint dir.
    ckpt_dir = _resolve_checkpoint(checkpoint_path)
    actor_dir = f"{ckpt_dir}/actor"
    merged_dir = f"{ckpt_dir}/hf_merged"
    logger.info("Checkpoint: %s", ckpt_dir)
    logger.info("Actor:      %s", actor_dir)
    logger.info("Merged HF:  %s", merged_dir)

    # 2) Merge FSDP shards → single HF-format model dir (idempotent: if
    #    `merged_dir` already has a model.safetensors, skip).
    safetensors_marker = Path(merged_dir) / "model.safetensors.index.json"
    single_safetensors = Path(merged_dir) / "model.safetensors"
    if not (safetensors_marker.exists() or single_safetensors.exists()):
        logger.info("Merging FSDP shards via `python -m verl.model_merger` ...")
        Path(merged_dir).mkdir(parents=True, exist_ok=True)
        # verl 0.5 ships the merger as a module, not a script.
        subprocess.run(
            [
                "python", "-m", "verl.model_merger", "merge",
                "--backend", "fsdp",
                "--local_dir", actor_dir,
                "--target_dir", merged_dir,
            ],
            check=True,
        )
        checkpoints_vol.commit()  # persist merged model alongside the shards
    else:
        logger.info("Reusing existing merged HF model at %s", merged_dir)

    # 3) Load gage config + build conversation.
    gcfg = load_gage_config(f"/workspace/{gage_config}")
    env = HydroEnvironment(gcfg)
    executor = ToolExecutor(env)
    messages = build_messages(gcfg)

    logger.info("=" * 60)
    logger.info("HydroLLM SGLang Eval")
    logger.info("Checkpoint: %s", ckpt_dir)
    logger.info("Gage:       %s", gcfg.gage_id)
    logger.info("Max turns:  %d", max_turns)
    logger.info("=" * 60)

    # 4) Boot SGLang local engine with Hermes tool parser (matches train).
    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        merged_dir, trust_remote_code=True, extra_special_tokens={},
    )

    logger.info("Starting SGLang engine ...")
    engine = sgl.Engine(
        model_path=merged_dir,
        tokenizer_path=merged_dir,
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

    parameter_trajectory = []
    valid_tool_calls = 0
    invalid_tool_calls = 0
    turn_details = []
    start = time.time()

    max_prompt_tokens = 8192 - max_tokens

    try:
        for turn in range(max_turns):
            logger.info("Turn %d/%d", turn + 1, max_turns)

            prompt = tokenizer.apply_chat_template(
                messages, tools=HYDRO_TOOLS, tokenize=False, add_generation_prompt=True,
            )
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            while prompt_len > max_prompt_tokens and len(messages) > 2:
                messages.pop(1)
                prompt = tokenizer.apply_chat_template(
                    messages, tools=HYDRO_TOOLS, tokenize=False, add_generation_prompt=True,
                )
                prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

            outputs = engine.generate([prompt], sampling_params)
            assistant_message = outputs[0]["text"]
            tool_calls = parse_tool_calls(assistant_message)

            if not tool_calls:
                messages.append({"role": "assistant", "content": assistant_message})
                logger.info("No tool calls in turn %d, ending.", turn + 1)
                turn_details.append({
                    "turn": turn + 1,
                    "action": "no_tool_call",
                    "response_preview": assistant_message[:500],
                })
                break

            import re
            import uuid

            reasoning = re.sub(r"<tool_call>.*?</tool_call>", "", assistant_message, flags=re.DOTALL).strip()
            json_pattern = r'\{"name"\s*:\s*"\w+"\s*,\s*"arguments"\s*:\s*\{.*?\}\}'
            reasoning = re.sub(json_pattern, "", reasoning, flags=re.DOTALL).strip()

            openai_tool_calls = []
            for call in tool_calls:
                cid = f"call_{uuid.uuid4().hex[:8]}"
                call["id"] = cid
                args_dict = call["arguments"] if isinstance(call["arguments"], dict) else json.loads(call["arguments"])
                openai_tool_calls.append({
                    "id": cid,
                    "type": "function",
                    "function": {"name": call["name"], "arguments": args_dict},
                })

            messages.append({
                "role": "assistant",
                "content": reasoning,
                "tool_calls": openai_tool_calls,
            })

            for call in tool_calls:
                tool_name = call["name"]
                tool_args = call["arguments"]
                logger.info("Executing %s ...", tool_name)
                result_str = executor.execute(tool_name, tool_args)
                result_data = json.loads(result_str)

                if result_data.get("status") == "error":
                    invalid_tool_calls += 1
                else:
                    valid_tool_calls += 1
                if tool_name == "set_parameters" and result_data.get("status") == "ok":
                    parameter_trajectory.append(result_data.get("validated_params", {}))

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": tool_name,
                    "content": _compact_tool_result(tool_name, result_data),
                })

                entry = {"turn": turn + 1, "tool": tool_name}
                if tool_name == "evaluate" and result_data.get("status") == "ok":
                    for k in _EVALUATE_METRIC_KEYS:
                        if k in result_data:
                            entry[k] = result_data[k]
                turn_details.append(entry)
    finally:
        elapsed = time.time() - start
        try:
            engine.shutdown()
        except Exception:
            pass
        env.cleanup()

    nse_hist = env.nse_history
    best_nse = max(nse_hist) if nse_hist else None
    step_tag = Path(ckpt_dir).name  # e.g. global_step_10

    result = {
        "checkpoint": ckpt_dir,
        "step": step_tag,
        "gage_id": gcfg.gage_id,
        "final_nse": nse_hist[-1] if nse_hist else None,
        "best_nse": round(best_nse, 4) if best_nse is not None else None,
        "nse_history": [round(n, 4) for n in nse_hist],
        "num_turns": len(turn_details),
        "num_simulation_runs": env.run_count,
        "valid_tool_calls": valid_tool_calls,
        "invalid_tool_calls": invalid_tool_calls,
        "parameter_trajectory": parameter_trajectory,
        "target_nse": gcfg.target_nse,
        "target_met": best_nse is not None and best_nse > gcfg.target_nse,
        "elapsed_seconds": round(elapsed, 1),
        "turn_details": turn_details,
    }

    output_path = f"/results/sglang_{step_tag}_{gcfg.gage_id}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    results_vol.commit()

    logger.info("-" * 60)
    logger.info("Step:        %s", step_tag)
    logger.info("Gage:        %s", gcfg.gage_id)
    logger.info("Best NSE:    %s", result["best_nse"])
    logger.info("Target NSE:  %s", gcfg.target_nse)
    logger.info("Target met:  %s", result["target_met"])
    logger.info("Turns:       %d", result["num_turns"])
    logger.info("Sim runs:    %d", result["num_simulation_runs"])
    logger.info("Elapsed:     %.1fs", elapsed)
    logger.info("Saved to:    %s", output_path)
    return result


@app.local_entrypoint()
def main(
    checkpoint_path: str = "/checkpoints/qwen3-4b-grpo",
    gage_config: str = "configs/gages/02338660.yaml",
    max_turns: int = 50,
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    result = run_inference.remote(
        checkpoint_path=checkpoint_path,
        gage_config=gage_config,
        max_turns=max_turns,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(json.dumps(result, indent=2, default=str))
