"""verl-compatible custom reward function for hydrologic calibration.

verl's `compute_score` is invoked once per rollout, after the trajectory
is complete. Per-turn rewards (ΔNSE, invalid-tool penalties) are already
returned by the tools in `verl_tools`; this function only adds the
trajectory-level terminal bonus tied to the *best* NSE achieved.

Signature follows verl's convention:
    compute_score(data_source, solution_str, ground_truth, extra_info=None)
"""

from __future__ import annotations

import json
import re
from typing import Any


_TARGET_BONUS = 0.5
_BEST_NSE_WEIGHT = 1.0
# Per-evaluate INCENTIVE (not penalty) to encourage long-horizon iteration.
# Earlier we used a -0.02 penalty per evaluate call; that taught the model
# to exit after 1 round. The user wants long-horizon calibration, so we
# pay a small bonus for each completed iteration. Capped via max_assistant_turns.
_PER_EVALUATE_BONUS = 0.02
# Reward strict monotonic improvement: each evaluate that beats the prior
# best earns this. This explicitly trains "iterate until you can't improve".
_IMPROVEMENT_BONUS = 0.1


def _extract_nse_history(text: str) -> list[float]:
    """Pull every NSE value mentioned in tool result blobs.

    `solution_str` from verl is the rendered trajectory text — system +
    user + assistant turns + tool results, joined.
    """
    nse_values: list[float] = []
    # Try JSON-block scan first.
    for match in re.finditer(r"\{[^{}]*\"NSE\"[^{}]*\}", text):
        snippet = match.group(0)
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        nse = data.get("NSE")
        if isinstance(nse, (int, float)) and nse > -998:
            nse_values.append(float(nse))
    # Fallback: regex for "NSE": value (handles the common case).
    if not nse_values:
        for match in re.finditer(r'"NSE"\s*:\s*([-\d.eE+]+)', text):
            try:
                v = float(match.group(1))
            except ValueError:
                continue
            if v > -998:
                nse_values.append(v)
    return nse_values


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Trajectory-level reward.

    Args:
        data_source: dataset tag (we set this to "hydrollm").
        solution_str: rendered trajectory text.
        ground_truth: dict with at least "target_nse".
        extra_info: optional metadata (gage_id, etc).

    Returns:
        Float reward to add to the per-turn rewards already accumulated.
    """
    target_nse = 0.8075
    if isinstance(ground_truth, dict):
        target_nse = float(ground_truth.get("target_nse", target_nse))
    elif isinstance(ground_truth, (int, float)):
        target_nse = float(ground_truth)

    history = _extract_nse_history(solution_str or "")
    if not history:
        # Agent never produced a valid evaluate() result.
        return -1.0

    best = max(history)
    score = _BEST_NSE_WEIGHT * max(min(best, 1.0), -1.0)
    if best > target_nse:
        score += _TARGET_BONUS

    # Reward iteration. Per-evaluate bonus for sustained engagement.
    score += _PER_EVALUATE_BONUS * len(history)
    # Plus an extra bonus for each evaluate that improved on the running best.
    running_best = float("-inf")
    n_improvements = 0
    for nse in history:
        if nse > running_best:
            n_improvements += 1
            running_best = nse
    # The very first evaluate is "improvement vs -inf" — don't double-count
    # by subtracting one (or treat the trajectory must have at least one).
    score += _IMPROVEMENT_BONUS * max(0, n_improvements - 1)

    return float(score)
