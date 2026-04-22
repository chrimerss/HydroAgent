"""Trajectory-level reward functions for multi-turn GRPO training.

The reward signal comes from the NSE (Nash-Sutcliffe Efficiency) score
computed after the agent runs the EF5/CREST hydrologic simulation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from hydrollm.config import GageConfig, load_gage_config
from hydrollm.environment import HydroEnvironment
from hydrollm.tools import ToolExecutor, parse_tool_calls

logger = logging.getLogger(__name__)


def extract_nse_history(trajectory: list[dict]) -> list[float]:
    """Extract NSE values from a multi-turn trajectory.

    Scans the trajectory for tool results that contain NSE scores.

    Args:
        trajectory: List of message dicts from the conversation history.

    Returns:
        List of NSE float values found in tool results.
    """
    nse_values = []
    for msg in trajectory:
        content = msg.get("content", "")
        if isinstance(content, str) and '"nse"' in content:
            try:
                data = json.loads(content)
                if "nse" in data and isinstance(data["nse"], (int, float)):
                    nse_values.append(float(data["nse"]))
            except (json.JSONDecodeError, TypeError):
                # Try regex fallback
                match = re.search(r'"nse"\s*:\s*([-\d.]+)', content)
                if match:
                    try:
                        nse_values.append(float(match.group(1)))
                    except ValueError:
                        pass
    return nse_values


def count_invalid_tool_calls(trajectory: list[dict]) -> int:
    """Count tool result messages that indicate errors."""
    count = 0
    for msg in trajectory:
        content = msg.get("content", "")
        if isinstance(content, str) and '"status": "error"' in content:
            count += 1
    return count


def hydro_trajectory_reward(
    completions: list[str],
    trajectory_inputs: list[list[dict]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Compute trajectory-level rewards for multi-turn GRPO.

    Reward components:
        1. Best NSE achieved (primary signal, clipped to [-1, 1])
        2. +0.2 improvement bonus if NSE improved from first to last run
        3. +0.5 target bonus if NSE > 0.8075
        4. -0.5 per invalid tool call (format penalty)
        5. -0.02 per simulation run (efficiency incentive)

    Args:
        completions: List of final completion strings from each rollout.
        trajectory_inputs: List of full conversation histories for each rollout.
            Each is a list of message dicts with "role" and "content" keys.

    Returns:
        List of float rewards, one per rollout.
    """
    rewards = []

    if trajectory_inputs is None:
        trajectory_inputs = [[] for _ in completions]

    for completion, trajectory in zip(completions, trajectory_inputs):
        nse_history = extract_nse_history(trajectory)

        # No valid NSE produced → harsh penalty
        if not nse_history:
            rewards.append(-1.0)
            continue

        best_nse = max(nse_history)

        # Primary reward: best NSE, clipped to [-1, 1]
        reward = max(min(best_nse, 1.0), -1.0)

        # Improvement bonus: did the agent learn from its mistakes?
        if len(nse_history) > 1 and nse_history[-1] > nse_history[0]:
            reward += 0.2

        # Target bonus: exceeding the calibration target
        if best_nse > 0.8075:
            reward += 0.5

        # Format penalty: invalid tool calls
        n_invalid = count_invalid_tool_calls(trajectory)
        reward -= n_invalid * 0.5

        # Efficiency penalty: encourage fewer simulation runs
        reward -= len(nse_history) * 0.02

        rewards.append(reward)

    return rewards


# ---------------------------------------------------------------------------
# Online reward: actually runs EF5 during GRPO rollouts
# ---------------------------------------------------------------------------

def make_online_reward(gage_config: GageConfig):
    """Create an online reward function that executes EF5 simulations.

    This function factory returns a reward function that:
    1. Creates a HydroEnvironment sandbox for each rollout
    2. Parses tool calls from the model's completion
    3. Executes the full multi-turn interaction
    4. Returns the trajectory-level reward

    Usage in GRPOTrainer:
        reward_fn = make_online_reward(gage_cfg)
        trainer = GRPOTrainer(reward_funcs=[reward_fn], ...)
    """

    def online_reward(completions: list, **kwargs: Any) -> list[float]:
        rewards = []
        for completion in completions:
            # TRL may pass completions as:
            #   - a string (plain text)
            #   - a list of message dicts (chat format: [{"role": ..., "content": ...}])
            if isinstance(completion, list):
                # Chat format: concatenate all content fields
                text = "\n".join(
                    msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    for msg in completion
                    if (isinstance(msg, dict) and msg.get("content")) or not isinstance(msg, dict)
                )
            else:
                text = str(completion)

            env = HydroEnvironment(gage_config)
            try:
                tool_calls = parse_tool_calls(text)
                if not tool_calls:
                    rewards.append(-1.0)
                    continue

                executor = ToolExecutor(env)
                for call in tool_calls:
                    executor.execute(call["name"], call["arguments"])

                # Use best NSE as reward
                if env.nse_history:
                    best_nse = max(env.nse_history)
                    reward = max(min(best_nse, 1.0), -1.0)
                    if best_nse > gage_config.target_nse:
                        reward += 0.5
                    reward -= len(env.nse_history) * 0.02
                    rewards.append(reward)
                else:
                    rewards.append(-1.0)
            finally:
                env.cleanup()

        return rewards

    return online_reward
