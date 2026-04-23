"""Modal entrypoint for HydroLLM inference evaluation.

Unified inference script: pass any model ID and the script determines
whether it's a baseline (base Qwen3-8B) or experiment (SFT/RL fine-tuned).

Usage:
    # Baseline — raw Qwen3-8B
    modal run modal_app/eval.py --model-id Qwen/Qwen3-8B

    # Experiment — SFT distilled model
    modal run modal_app/eval.py --model-id chrimerss/Qwen-3-8B-hydro-distill

    # Experiment — RL fine-tuned model
    modal run modal_app/eval.py --model-id chrimerss/Qwen-3-8B-hydroLLM

    # Custom gage + turns
    modal run modal_app/eval.py --model-id Qwen/Qwen3-8B \
        --gage-config configs/gages/02338660.yaml --max-turns 15
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import modal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from modal_app.images import eval_image

app = modal.App("hydrollm-eval")

results_vol = modal.Volume.from_name("hydrollm-results", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("hydrollm-checkpoints", create_if_missing=True)

# Models treated as "baseline" (no fine-tuning applied)
BASELINE_MODELS = {
    "Qwen/Qwen3-8B",
}


def classify_model(model_id: str) -> str:
    """Classify a model as 'baseline' or 'experiment'."""
    if model_id in BASELINE_MODELS:
        return "baseline"
    return "experiment"


def _compact_tool_result(tool_name: str, data: dict) -> str:
    """Return a short summary of tool output for the conversation context."""
    status = data.get("status", "error")
    if status == "error":
        return json.dumps({"status": "error", "message": data.get("message", "")})

    if tool_name == "set_parameters":
        params = data.get("validated_params", {})
        return json.dumps({"status": "ok", "params": params})

    if tool_name == "run_simulation":
        return json.dumps({"status": "ok", "run": data.get("run_number")})

    if tool_name == "evaluate":
        return json.dumps({
            "status": "ok",
            "nse": data.get("nse") or data.get("current_nse"),
            "best_nse": data.get("best_nse"),
            "target_met": data.get("target_met"),
        })

    return json.dumps({"status": status})


@app.function(
    image=eval_image,
    gpu="H100:1",
    timeout=7200,  # 2 hours
    volumes={
        "/results": results_vol,
        "/checkpoints": checkpoints_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    memory=65536,
)
def run_inference(
    model_id: str = "Qwen/Qwen3-8B",
    gage_config: str = "configs/gages/02338660.yaml",
    max_turns: int = 100,
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    """Run model inference on a calibration task.

    Automatically classifies the model as baseline or experiment
    based on the model ID and saves results accordingly.

    Args:
        model_id: HuggingFace model ID (base or fine-tuned).
        gage_config: Path to gage YAML config.
        max_turns: Maximum calibration turns.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens per generation.
    """
    import logging
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hydrollm.eval")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from hydrollm.config import load_gage_config
    from hydrollm.environment import HydroEnvironment
    from hydrollm.prompts import build_messages
    from hydrollm.tools import HYDRO_TOOLS, ToolExecutor, parse_tool_calls

    # Classify model
    run_type = classify_model(model_id)
    model_short = model_id.split("/")[-1]

    gcfg = load_gage_config(f"/app/{gage_config}")

    logger.info("=" * 60)
    logger.info("HydroLLM Inference Evaluation")
    logger.info("Model:     %s", model_id)
    logger.info("Run type:  %s", run_type.upper())
    logger.info("Gage:      %s", gcfg.gage_id)
    logger.info("Max turns: %d", max_turns)
    logger.info("=" * 60)

    # Load tokenizer + model
    logger.info("Loading model %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        extra_special_tokens={},  # Workaround for Qwen tokenizer save bug (list vs dict)
    )

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=8192,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
    )

    # Initialize environment and conversation
    env = HydroEnvironment(gcfg)
    executor = ToolExecutor(env)
    messages = build_messages(gcfg)

    parameter_trajectory = []
    valid_tool_calls = 0
    invalid_tool_calls = 0
    turn_details = []
    start_time = time.time()

    max_prompt_tokens = 8192 - max_tokens  # reserve room for the completion

    try:
        for turn in range(max_turns):
            logger.info("Turn %d/%d", turn + 1, max_turns)

            # Apply chat template with tool definitions
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=HYDRO_TOOLS,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Trim oldest mid-conversation turns if prompt exceeds context budget
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            while prompt_len > max_prompt_tokens and len(messages) > 2:
                messages.pop(1)  # drop oldest turn after system message
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tools=HYDRO_TOOLS,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

            # Generate
            outputs = llm.generate(prompt, sampling_params)
            assistant_message = outputs[0].outputs[0].text

            messages.append({"role": "assistant", "content": assistant_message})

            # Parse tool calls
            tool_calls = parse_tool_calls(assistant_message)

            if not tool_calls:
                logger.info("No tool calls in turn %d, ending conversation", turn + 1)
                turn_details.append({
                    "turn": turn + 1,
                    "action": "no_tool_call",
                    "response_preview": assistant_message[:200],
                })
                break

            # Execute each tool call
            for call in tool_calls:
                tool_name = call["name"]
                tool_args = call["arguments"]

                logger.info("Executing tool: %s", tool_name)
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
                    "name": tool_name,
                    "content": _compact_tool_result(tool_name, result_data),
                })

                turn_details.append({
                    "turn": turn + 1,
                    "tool": tool_name,
                    "nse": result_data.get("nse"),
                })

    finally:
        elapsed = time.time() - start_time
        env.cleanup()

    nse_hist = env.nse_history
    best_nse = max(nse_hist) if nse_hist else None

    result = {
        "model": model_id,
        "run_type": run_type,
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

    # Save results
    output_path = f"/results/{run_type}_{model_short}_{gcfg.gage_id}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    results_vol.commit()

    logger.info("-" * 60)
    logger.info("Run type:    %s", run_type.upper())
    logger.info("Best NSE:    %s", result["best_nse"])
    logger.info("Target NSE:  %s", gcfg.target_nse)
    logger.info("Target met:  %s", result["target_met"])
    logger.info("Turns:       %d", result["num_turns"])
    logger.info("Elapsed:     %.1fs", elapsed)
    logger.info("Saved to:    %s", output_path)
    logger.info("-" * 60)

    return result


@app.local_entrypoint()
def main(
    model_id: str = "Qwen/Qwen3-8B",
    gage_config: str = "configs/gages/02338660.yaml",
    max_turns: int = 100,
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    """Run inference evaluation on Modal.

    Examples:
        # Baseline
        modal run modal_app/eval.py --model-id Qwen/Qwen3-8B

        # SFT experiment
        modal run modal_app/eval.py --model-id chrimerss/Qwen-3-8B-hydro-distill

        # RL experiment
        modal run modal_app/eval.py --model-id chrimerss/Qwen-3-8B-hydroLLM
    """
    result = run_inference.remote(
        model_id=model_id,
        gage_config=gage_config,
        max_turns=max_turns,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(json.dumps(result, indent=2, default=str))
