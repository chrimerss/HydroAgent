"""Tests for the calibration session harness (hydrollm.session), with a fake
model, environment and tokenizer, so no GPU or EF5 is needed."""

import json

import pytest

import hydrollm.session as session
from hydrollm.config import GageConfig
from hydrollm.prompts import STOP_NSE, SYSTEM_PROMPT, SYSTEM_PROMPT_V1, build_messages

PARAMS = {"wm": 1.0, "b": 1e-6, "im": 0.1, "ke": 1.0, "fc": 1.0, "under": 1.0,
          "leaki": 1.0, "alpha": 1.0, "beta": 1.0, "alpha0": 0.5, "iwu": 20.0}


def call(name, args=None):
    return '<tool_call>\n' + json.dumps({"name": name, "arguments": args or {}}) + '\n</tool_call>'


class FakeEnv:
    def __init__(self, gcfg):
        self.nse_history, self.run_count = [], 0

    def cleanup(self):
        pass


class FakeExecutor:
    nse_values: list = []

    def __init__(self, env):
        self.env = env

    def execute(self, name, args):
        if name == "set_parameters":
            return json.dumps({"status": "ok", "validated_params": {k: min(v, 2.0) for k, v in args.items()}})
        if name == "run_simulation":
            self.env.run_count += 1
            return json.dumps({"status": "ok", "run_number": self.env.run_count})
        nse = self.nse_values[len(self.env.nse_history)]
        self.env.nse_history.append(nse)
        return json.dumps({"status": "ok", "NSE": nse})


class FakeTokenizer:
    def apply_chat_template(self, msgs, tools=None, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"<{m['role']}>{m.get('content', '')}" for m in msgs)

    def encode(self, text, add_special_tokens=False):
        return text.split()


@pytest.fixture
def gcfg():
    return GageConfig(gage_id="00000000", lon=0.0, lat=0.0, basin_area=1.0, obs_dir="/x",
                      control_template="/x", time_begin="201801010000", time_end="201801020000")


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(session, "HydroEnvironment", FakeEnv)
    monkeypatch.setattr(session, "ToolExecutor", FakeExecutor)


def scripted(outputs):
    it = iter(outputs)
    seen = []

    def generate(prompt):
        seen.append(prompt)
        return next(it)
    return generate, seen


def test_v2_ends_session_when_model_stops_after_threshold(gcfg):
    FakeExecutor.nse_values = [STOP_NSE + 0.05]
    gen, _ = scripted([call("set_parameters", PARAMS), call("run_simulation"), call("evaluate"),
                       "Calibration complete."])
    rec = session.run_session(gen, FakeTokenizer(), gcfg, harness="v2", max_turns=10)
    assert rec["end_reason"] == "stopped_after_threshold"
    assert rec["num_turns"] == 4 and rec["turn_details"][-1]["action"] == "finished"


def test_v1_keeps_hinting_after_threshold(gcfg):
    FakeExecutor.nse_values = [STOP_NSE + 0.05]
    gen, seen = scripted([call("set_parameters", PARAMS), call("run_simulation"), call("evaluate"),
                          "Done.", "Done.", "Done.", "Done."])
    rec = session.run_session(gen, FakeTokenizer(), gcfg, harness="v1", max_turns=10)
    assert rec["end_reason"] == "parse_failures"
    assert "Re-emit your tool call exactly" in seen[4]


def test_v2_hint_names_next_step(gcfg):
    FakeExecutor.nse_values = [0.1]
    gen, seen = scripted([call("set_parameters", PARAMS), "Now I will run it.",
                          call("run_simulation"), "ok", call("evaluate"), "Hmm.", "x", "y", "z"])
    session.run_session(gen, FakeTokenizer(), gcfg, harness="v2", max_turns=9)
    assert "next step is run_simulation" in seen[2]
    assert "next step is evaluate" in seen[4]
    assert "best NSE so far is 0.100" in seen[6] and "call set_parameters" in seen[6]


def test_v2_reports_clamped_values_v1_does_not(gcfg):
    FakeExecutor.nse_values = []
    for harness, expect in (("v2", True), ("v1", False)):
        gen, seen = scripted([call("set_parameters", dict(PARAMS, fc=2.5)), call("run_simulation")])
        rec = session.run_session(gen, FakeTokenizer(), gcfg, harness=harness, max_turns=2)
        # Only v2 tells the model; both record the clamp for analysis.
        assert ('"clamped"' in seen[1]) is expect
        assert rec["turn_details"][0].get("clamped") == ["fc"]


def test_clamped_values():
    out = session.clamped_values({"fc": 2.5, "wm": 1.0, "b": 0.0, "bogus": 3})
    assert set(out) == {"fc", "b"}
    assert out["fc"] == {"requested": 2.5, "used": 2.0, "range": [0.1, 2.0]}


def test_build_messages_harness_prompts(gcfg):
    assert build_messages(gcfg, harness="v1")[0]["content"] == SYSTEM_PROMPT_V1
    assert build_messages(gcfg)[0]["content"] == SYSTEM_PROMPT
    assert "Stop when NSE > target." in SYSTEM_PROMPT_V1
