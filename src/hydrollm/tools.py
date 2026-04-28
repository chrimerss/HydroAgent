"""Multi-turn tool definitions for hydrologic model calibration.

Tools are defined as JSON schemas compatible with Qwen2.5's Hermes
tool-calling template. The tool executor maps tool calls from the LLM
to HydroEnvironment methods.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from hydrollm.environment import HydroEnvironment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Definitions (Hermes / OpenAI function-calling format)
# ---------------------------------------------------------------------------

HYDRO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_parameters",
            "description": (
                "Set CREST hydrologic model parameter multipliers for calibration. "
                "Each parameter scales a spatially distributed grid that represents "
                "a physical property of the watershed. After setting parameters, you "
                "must call run_simulation to execute the model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "wm": {
                        "type": "number",
                        "description": (
                            "Soil water storage capacity multiplier [0.1, 10.0]. "
                            "Higher values increase total water holding capacity, "
                            "reducing runoff for small events."
                        ),
                    },
                    "b": {
                        "type": "number",
                        "description": (
                            "Variable infiltration curve shape [0.000001, 3.0]. "
                            "Controls spatial variability of soil moisture capacity. "
                            "Higher values produce more uneven infiltration."
                        ),
                    },
                    "im": {
                        "type": "number",
                        "description": (
                            "Impervious area fraction [0.0, 1.0]. "
                            "Fraction of the watershed that generates direct runoff "
                            "regardless of soil moisture."
                        ),
                    },
                    "ke": {
                        "type": "number",
                        "description": (
                            "Evapotranspiration scaling factor [0.8, 1.2]. "
                            "Multiplier on potential ET. Values >1 increase ET losses."
                        ),
                    },
                    "fc": {
                        "type": "number",
                        "description": (
                            "Saturated hydraulic conductivity multiplier [0.1, 2.0]. "
                            "Controls the rate of soil drainage. Higher values allow "
                            "faster drainage, reducing surface runoff."
                        ),
                    },
                    "under": {
                        "type": "number",
                        "description": (
                            "Interflow velocity multiplier [0.1, 10.0]. "
                            "Controls how fast subsurface lateral flow reaches the channel."
                        ),
                    },
                    "leaki": {
                        "type": "number",
                        "description": (
                            "Interflow leakage rate multiplier [0.1, 10.0]. "
                            "Controls how much interflow leaks to deeper groundwater."
                        ),
                    },
                    "alpha": {
                        "type": "number",
                        "description": (
                            "Channel routing coefficient [0.1, 3.0] in Q = alpha * A^beta. "
                            "Controls the relationship between cross-sectional area and discharge."
                        ),
                    },
                    "beta": {
                        "type": "number",
                        "description": (
                            "Channel routing exponent [0.1, 3.0] in Q = alpha * A^beta. "
                            "Controls the nonlinearity of the stage-discharge relationship."
                        ),
                    },
                    "alpha0": {
                        "type": "number",
                        "description": (
                            "Overland (non-channel) routing parameter [0.0, 3.0]. "
                            "Controls the speed of overland flow before it reaches a channel."
                        ),
                    },
                    "iwu": {
                        "type": "number",
                        "description": (
                            "Initial soil moisture [0.1, 100.0]. "
                            "Sets the starting soil water content at the beginning "
                            "of the simulation period."
                        ),
                    },
                },
                "required": [
                    "wm", "b", "im", "ke", "fc",
                    "under", "leaki",
                    "alpha", "beta", "alpha0",
                    "iwu",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": (
                "Execute the EF5/CREST simulation with the currently set parameters. "
                "Runs the hydrologic model and produces an output CSV file with "
                "simulated discharge time series. You must call set_parameters first. "
                "This tool does NOT return any performance metrics — it only confirms "
                "the simulation completed. You must call evaluate afterwards to see "
                "NSE, KGE, correlation, peak magnitudes, and timing error."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate",
            "description": (
                "Compute a full metric suite from the most recent simulation output. "
                "Returns NSE (Nash-Sutcliffe Efficiency), CC (Pearson correlation), "
                "KGE (Kling-Gupta Efficiency), simulated and observed peak discharge, "
                "the peak ratio (sim/obs), and the peak-timing lag in hours "
                "(positive = simulation peaks late). Call this after run_simulation "
                "— it is the only tool that exposes performance metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Executes tool calls against a HydroEnvironment instance.

    Maps tool function names from LLM output to environment methods.
    """

    def __init__(self, environment: HydroEnvironment):
        self.env = environment
        self._dispatch = {
            "set_parameters": self._handle_set_parameters,
            "run_simulation": self._handle_run_simulation,
            "evaluate": self._handle_evaluate,
        }

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return the result as a JSON string.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Parsed arguments dict from the tool call.

        Returns:
            JSON string with the tool result.
        """
        handler = self._dispatch.get(tool_name)
        if handler is None:
            return json.dumps({
                "status": "error",
                "message": f"Unknown tool: {tool_name}. "
                f"Available tools: {list(self._dispatch.keys())}",
            })

        try:
            result = handler(arguments)
            return json.dumps(result, default=str)
        except Exception as e:
            logger.exception("Tool execution error for %s", tool_name)
            return json.dumps({
                "status": "error",
                "message": f"Tool execution failed: {str(e)}",
            })

    def _handle_set_parameters(self, arguments: dict[str, Any]) -> dict:
        return self.env.set_parameters(arguments)

    def _handle_run_simulation(self, arguments: dict[str, Any]) -> dict:
        return self.env.run_simulation()

    def _handle_evaluate(self, arguments: dict[str, Any]) -> dict:
        return self.env.evaluate()


# ---------------------------------------------------------------------------
# Tool call parsing utilities
# ---------------------------------------------------------------------------

def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from model output text.

    Handles the Hermes-style <tool_call> format used by Qwen2.5:

        <tool_call>
        {"name": "set_parameters", "arguments": {"wm": 2.5, ...}}
        </tool_call>

    Also handles the standard JSON function_call format.

    Returns a list of dicts with 'name' and 'arguments' keys.
    """
    import re

    tool_calls = []

    # Pattern 1: Hermes XML-style tool calls
    hermes_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    for match in re.finditer(hermes_pattern, text, re.DOTALL):
        snippet = match.group(1)
        try:
            call_data = json.loads(snippet)
            name = call_data.get("name", "")
            arguments = call_data.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            tool_calls.append({"name": name, "arguments": arguments})
            continue
        except json.JSONDecodeError:
            pass

        # Lenient fallback for the common Qwen3 typo on empty-args tools:
        # `{"name": "run_simulation", "arguments: {}}` (missing closing
        # quote on `arguments`). Extract `name` via regex; default args = {}.
        name_m = re.search(r'"name"\s*:\s*"([^"\\]+)"', snippet)
        if not name_m:
            logger.warning("Failed to parse tool call JSON: %s", snippet[:200])
            continue
        repaired_snippet = re.sub(r'"arguments\s*:\s*\{', '"arguments": {', snippet)
        try:
            call_data = json.loads(repaired_snippet)
            arguments = call_data.get("arguments", {})
        except json.JSONDecodeError:
            arguments = {}
            args_m = re.search(r"'?arguments'?\s*:?\s*(\{[^}]*\})", snippet, re.DOTALL)
            if args_m:
                try:
                    arguments = json.loads(args_m.group(1))
                except json.JSONDecodeError:
                    pass
        tool_calls.append({"name": name_m.group(1), "arguments": arguments})
        logger.warning(
            "Repaired malformed tool_call JSON (extracted name=%s, args=%s): %s",
            name_m.group(1), arguments, snippet[:200],
        )

    # Pattern 2: Direct JSON function calls (fallback)
    if not tool_calls:
        json_pattern = r'\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                name = match.group(1)
                arguments = json.loads(match.group(2))
                tool_calls.append({"name": name, "arguments": arguments})
            except json.JSONDecodeError:
                logger.warning("Failed to parse function call: %s", match.group(0)[:200])

    return tool_calls
