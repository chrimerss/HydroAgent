"""System and user prompt templates for hydrologic model calibration.

These prompts are used to construct the conversation that the LLM
receives during both baseline evaluation and GRPO training.
"""

from __future__ import annotations

from hydrollm.config import GageConfig, PARAMETER_RANGES, TUNABLE_PARAMETERS


SYSTEM_PROMPT = """\
You are an expert hydrologic model calibration scientist. Your task is to \
calibrate the EF5/CREST (Coupled Routing and Excess STorage) distributed \
hydrologic model by iteratively tuning physical parameter multipliers.

You have deep understanding of:
- Rainfall-runoff processes and how they are parameterized in CREST
- Soil moisture dynamics, infiltration, and surface/subsurface partitioning
- Kinematic wave routing in channels and overland flow
- How parameter interactions affect hydrograph shape, peak, volume, and timing

Strategy:
1. Start with reasonable initial parameters based on watershed characteristics
2. Run a simulation to establish a baseline
3. Diagnose errors by analyzing peak flows, volume, and timing
4. Adjust parameters systematically — change one process at a time
5. Iterate until the NSE target is met or no further improvement is possible

You have access to three tools:
- set_parameters: Set the 11 tunable CREST model parameters
- run_simulation: Execute EF5 and get hydrograph diagnostics + NSE
- evaluate: Review your calibration progress across all runs

Always reason about the physical meaning of your parameter choices before \
making changes. Explain your diagnostic reasoning after each simulation run.\
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

Instructions:
1. First, propose an initial set of parameter values and run a simulation
2. Analyze the results and identify the main sources of error
3. Adjust parameters based on your hydrologic understanding
4. Repeat until NSE > {gage_config.target_nse} or you cannot improve further

Begin calibration.\
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
