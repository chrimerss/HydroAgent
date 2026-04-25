"""System and user prompt templates for hydrologic model calibration.

These prompts are used to construct the conversation that the LLM
receives during both baseline evaluation and GRPO training.
"""

from __future__ import annotations

from hydrollm.config import GageConfig, PARAMETER_RANGES, TUNABLE_PARAMETERS


SYSTEM_PROMPT = """\
You are an expert hydrologic model calibration scientist. You calibrate \
the EF5/CREST distributed hydrologic model by calling tools — never by \
narrating, never by inventing numbers, never by writing tables of \
"hypothetical" results.

You have three tools: `set_parameters`, `run_simulation`, `evaluate`.

CRITICAL FORMAT — every tool invocation MUST use exactly this XML+JSON syntax \
(NOT Python-style `func(arg=val)`, NOT markdown):

<tool_call>{"name": "<tool_name>", "arguments": {"<key>": <value>, ...}}</tool_call>

Calling rules:
- `set_parameters` requires all 11 keys (wm, b, im, ke, fc, under, leaki, \
alpha, beta, alpha0, iwu) as numeric values.
- `run_simulation` and `evaluate` take no arguments: `"arguments": {}`.

Protocol: each iteration is set_parameters → run_simulation → evaluate. \
After each evaluate result, write 1–2 sentences of diagnosis, then begin \
the next iteration with another set_parameters tool_call. Adjust one \
process at a time (peak, volume, timing). Stop when NSE > target.\
"""


def build_user_prompt(gage_config: GageConfig) -> str:
    """Build the user prompt for a specific gage.

    Includes watershed metadata, parameter ranges, and objective.
    """
    # Build parameter table
    param_lines = []
    for param in TUNABLE_PARAMETERS:
        lo, hi = PARAMETER_RANGES[param]
        param_lines.append(f"  - {param}: [{lo}, {hi}]")
    param_table = "\n".join(param_lines)

    return f"""\
Calibrate the CREST hydrologic model for USGS gage {gage_config.gage_id}.

Watershed Information:
- Gage ID: {gage_config.gage_id}
- Drainage area: {gage_config.basin_area} km²
- Location: ({gage_config.lat}°N, {gage_config.lon}°E)
- Evaluation period: {_format_time(gage_config.time_begin)} to \
{_format_time(gage_config.time_end)}

Objective: Achieve NSE > {gage_config.target_nse}

Tunable Parameters (name: [min, max]):
{param_table}

Fixed Parameters (do not change):
  - th: 10.0 (channel initiation threshold)
  - isu: 0.0 (initial interflow storage)

Begin by calling set_parameters.\
"""


def build_messages(gage_config: GageConfig) -> list[dict[str, str]]:
    """Build the initial message list for a calibration session.

    Returns a list of message dicts suitable for chat template formatting.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(gage_config)},
    ]


def _format_time(time_str: str) -> str:
    """Format YYYYMMDDHHMM to a readable date string."""
    if len(time_str) >= 8:
        return f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]}"
    return time_str
