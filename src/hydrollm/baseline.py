"""Baseline inference evaluator.

Runs each base Qwen2.5 model (pre-training) on the calibration task
to establish the performance floor. Records NSE, parameter trajectories,
and tool call validity.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from hydrollm.config import GageConfig, ModelConfig
from hydrollm.environment import HydroEnvironment
from hydrollm.prompts import build_messages
from hydrollm.tools import HYDRO_TOOLS, ToolExecutor, parse_tool_calls

logger = logging.getLogger(__name__)


def run_baseline(
    model_name: str,
    gage_config: GageConfig,
    max_turns: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Run a base model on hydrologic calibration without any RL training.

    This function:
    1. Loads the model via vLLM (offline mode)
    2. Uses the tokenizer's chat template to inject tool definitions
    3. Executes the conversation loop (model generates → tools execute → repeat)
    4. Records all metrics for baseline analysis

    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B-Instruct").
        gage_config: Gage configuration for the calibration task.
        max_turns: Maximum number of turns (each turn = model response + tool execution).
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens per generation.

    Returns:
        Dict with baseline results including NSE, trajectories, and metrics.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    logger.info("Loading model %s for baseline evaluation...", model_name)

    # Load tokenizer separately to apply chat template with tools
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],  # Qwen2.5 end-of-turn token
    )

    # Initialize environment and conversation
    env = HydroEnvironment(gage_config)
    executor = ToolExecutor(env)
    messages = build_messages(gage_config)

    parameter_trajectory = []
    valid_tool_calls = 0
    invalid_tool_calls = 0
    turn_details = []
    start_time = time.time()

    try:
        for turn in range(max_turns):
            logger.info("Turn %d/%d", turn + 1, max_turns)

            # Apply chat template with tool definitions
            # Qwen2.5-Instruct tokenizers support `tools` in apply_chat_template
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=HYDRO_TOOLS,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Generate model response using offline LLM API
            outputs = llm.generate(prompt, sampling_params)
            assistant_message = outputs[0].outputs[0].text

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": assistant_message})

            # Parse tool calls from the response
            tool_calls = parse_tool_calls(assistant_message)

            if not tool_calls:
                # Model didn't make a tool call — check if it's done
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

                # Check validity
                result_data = json.loads(result_str)
                if result_data.get("status") == "error":
                    invalid_tool_calls += 1
                else:
                    valid_tool_calls += 1

                # Record parameter trajectory
                if tool_name == "set_parameters" and result_data.get("status") == "ok":
                    parameter_trajectory.append(result_data.get("validated_params", {}))

                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": result_str,
                })

                turn_details.append({
                    "turn": turn + 1,
                    "tool": tool_name,
                    "nse": result_data.get("nse"),
                })

    finally:
        elapsed = time.time() - start_time
        eval_result = env.evaluate()
        env.cleanup()

    return {
        "model": model_name,
        "gage_id": gage_config.gage_id,
        "final_nse": eval_result["current_nse"],
        "best_nse": eval_result["best_nse"],
        "nse_history": eval_result["nse_history"],
        "num_turns": len(turn_details),
        "num_simulation_runs": eval_result["num_runs"],
        "valid_tool_calls": valid_tool_calls,
        "invalid_tool_calls": invalid_tool_calls,
        "parameter_trajectory": parameter_trajectory,
        "target_met": eval_result["target_met"],
        "elapsed_seconds": round(elapsed, 1),
        "turn_details": turn_details,
    }


def format_baseline_report(results: list[dict[str, Any]]) -> str:
    """Format baseline results into a readable markdown report.

    Args:
        results: List of result dicts from run_baseline().

    Returns:
        Markdown-formatted report string.
    """
    lines = ["# HydroLLM Baseline Evaluation Report\n"]

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Model | Gage | Best NSE | Turns | Valid Calls | Target Met | Time (s) |")
    lines.append("|-------|------|----------|-------|-------------|------------|----------|")
    for r in results:
        model_short = r["model"].split("/")[-1]
        target = "✅" if r["target_met"] else "❌"
        lines.append(
            f"| {model_short} | {r['gage_id']} | {r['best_nse']} | "
            f"{r['num_turns']} | {r['valid_tool_calls']} | {target} | "
            f"{r['elapsed_seconds']} |"
        )

    # Detailed results per model
    lines.append("\n## Detailed Results\n")
    for r in results:
        model_short = r["model"].split("/")[-1]
        lines.append(f"### {model_short} on gage {r['gage_id']}\n")
        lines.append(f"- **Best NSE**: {r['best_nse']}")
        lines.append(f"- **NSE History**: {r['nse_history']}")
        lines.append(f"- **Simulation Runs**: {r['num_simulation_runs']}")
        lines.append(f"- **Valid/Invalid Tool Calls**: {r['valid_tool_calls']}/{r['invalid_tool_calls']}")
        lines.append(f"- **Time**: {r['elapsed_seconds']}s")
        if r["parameter_trajectory"]:
            lines.append(f"- **Final Parameters**: {json.dumps(r['parameter_trajectory'][-1], indent=2)}")
        lines.append("")

    return "\n".join(lines)
