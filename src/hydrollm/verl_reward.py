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
_NUM_RUNS_PENALTY = 0.02


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
    score -= _NUM_RUNS_PENALTY * len(history)
    return float(score)
