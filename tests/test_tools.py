"""Unit tests for tool definitions and parsing."""

import json
import pytest

from hydrollm.tools import HYDRO_TOOLS, ToolExecutor, parse_tool_calls


class TestToolDefinitions:
    """Verify tool definitions are well-formed."""

    def test_three_tools_defined(self):
        assert len(HYDRO_TOOLS) == 3

    def test_tool_names(self):
        names = {t["function"]["name"] for t in HYDRO_TOOLS}
        assert names == {"set_parameters", "run_simulation", "evaluate"}

    def test_set_parameters_has_required_fields(self):
        set_params = next(
            t for t in HYDRO_TOOLS if t["function"]["name"] == "set_parameters"
        )
        props = set_params["function"]["parameters"]["properties"]
        required = set_params["function"]["parameters"]["required"]

        # All 11 tunable parameters should be in properties
        expected_params = [
            "wm", "b", "im", "ke", "fc", "under", "leaki",
            "alpha", "beta", "alpha0", "iwu",
        ]
        for param in expected_params:
            assert param in props, f"Missing property: {param}"
            assert param in required, f"Missing required: {param}"

    def test_all_tools_have_descriptions(self):
        for tool in HYDRO_TOOLS:
            assert "description" in tool["function"]
            assert len(tool["function"]["description"]) > 20

    def test_tools_are_valid_json(self):
        # Should be serializable
        serialized = json.dumps(HYDRO_TOOLS)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 3


class TestToolCallParsing:
    """Test parsing tool calls from model output."""

    def test_parse_hermes_format(self):
        text = """Let me set the initial parameters.

<tool_call>
{"name": "set_parameters", "arguments": {"wm": 2.5, "b": 0.8, "im": 0.05, "ke": 1.0, "fc": 1.0, "under": 1.0, "leaki": 1.0, "alpha": 1.0, "beta": 1.0, "alpha0": 0.0, "iwu": 25.0}}
</tool_call>"""

        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "set_parameters"
        assert calls[0]["arguments"]["wm"] == 2.5
        assert calls[0]["arguments"]["b"] == 0.8

    def test_parse_run_simulation(self):
        text = """Now let me run the simulation.

<tool_call>
{"name": "run_simulation", "arguments": {}}
</tool_call>"""

        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "run_simulation"

    def test_parse_multiple_calls(self):
        text = """Setting parameters and running.

<tool_call>
{"name": "set_parameters", "arguments": {"wm": 3.0, "b": 1.0, "im": 0.1, "ke": 1.0, "fc": 1.0, "under": 1.0, "leaki": 1.0, "alpha": 1.0, "beta": 1.0, "alpha0": 0.0, "iwu": 30.0}}
</tool_call>

<tool_call>
{"name": "run_simulation", "arguments": {}}
</tool_call>"""

        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "set_parameters"
        assert calls[1]["name"] == "run_simulation"

    def test_parse_no_tool_calls(self):
        text = "I think the calibration is complete. The NSE is 0.85."
        calls = parse_tool_calls(text)
        assert len(calls) == 0

    def test_parse_malformed_json(self):
        text = """<tool_call>
{"name": "set_parameters", "arguments": {invalid json}}
</tool_call>"""
        calls = parse_tool_calls(text)
        assert len(calls) == 0  # Gracefully handles parse error

    def test_parse_json_fallback_format(self):
        text = '{"name": "run_simulation", "arguments": {}}'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "run_simulation"


class TestToolExecutor:
    """Test ToolExecutor dispatching."""

    def test_unknown_tool(self):
        from unittest.mock import MagicMock
        env = MagicMock()
        executor = ToolExecutor(env)

        result = json.loads(executor.execute("nonexistent_tool", {}))
        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]

    def test_set_parameters_dispatch(self):
        from unittest.mock import MagicMock
        env = MagicMock()
        env.set_parameters.return_value = {"status": "ok", "validated_params": {"wm": 2.0}}

        executor = ToolExecutor(env)
        result = json.loads(executor.execute("set_parameters", {"wm": 2.0}))
        assert result["status"] == "ok"
        env.set_parameters.assert_called_once_with({"wm": 2.0})

    def test_run_simulation_dispatch(self):
        from unittest.mock import MagicMock
        env = MagicMock()
        env.run_simulation.return_value = {"status": "ok", "nse": 0.75}

        executor = ToolExecutor(env)
        result = json.loads(executor.execute("run_simulation", {}))
        assert result["nse"] == 0.75
        env.run_simulation.assert_called_once()
