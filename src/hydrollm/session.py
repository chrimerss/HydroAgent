"""One multi-turn calibration session: the evaluation harness.

Shared by curc/eval.py (CURC), modal_app/eval.py (Modal) and wsp/fwd.py, so
every entry point runs the same protocol. `generate(prompt) -> text` wraps the
inference backend.

Two harness versions (`harness=`):

v1  The original harness, kept so results can be compared like for like:
    the system prompt says "Stop when NSE > target." without giving a target;
    when the context overflows the oldest message is dropped first, which is
    the user's task message; a turn without a tool call gets "re-emit your
    tool call exactly"; set_parameters reports only the values it used.

v2  (default) The fixed harness:
    - the system prompt says "Stop when NSE > 0.8." (prompts.STOP_NSE);
    - the task message is always kept and whole old exchanges are dropped
      (prompts.fit_context);
    - the retry hint names the next protocol step instead of asking for the
      previous call again;
    - set_parameters reports which requested values were outside their range
      and what was used instead;
    - a turn without a tool call after the best NSE has passed 0.8 ends the
      session (end_reason "stopped_after_threshold").
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from hydrollm.config import PARAMETER_RANGES, GageConfig
from hydrollm.environment import HydroEnvironment
from hydrollm.prompts import STOP_NSE, build_messages, fit_context
from hydrollm.tools import HYDRO_TOOLS, ToolExecutor, parse_tool_calls

logger = logging.getLogger("hydrollm.eval")

HARNESSES = ("v1", "v2")

EVALUATE_METRIC_KEYS = (
    "NSE", "CC", "KGE",
    "sim_peak", "obs_peak", "peak_ratio",
    "lag_hours_sim_minus_obs", "num_points",
)

_TOOL_CALL_FORMAT = (
    "<tool_call>\n"
    '{"name": "<tool_name>", "arguments": {<json_arguments>}}\n'
    "</tool_call>"
)

RETRY_HINT_V1 = (
    "Your previous response could not be parsed as a valid tool call. "
    "Re-emit your tool call exactly in this format and nothing else:\n" + _TOOL_CALL_FORMAT
)

_NEXT_STEP = {
    None: "Start the calibration: call set_parameters with all 11 parameters.",
    "set_parameters": "The parameters are set; the next step is run_simulation.",
    "run_simulation": "The simulation has finished; the next step is evaluate.",
}


def retry_hint(harness: str, last_ok_tool: str | None, best_nse: float | None) -> str:
    """The user message sent after a turn without a parseable tool call."""
    if harness == "v1":
        return RETRY_HINT_V1
    if last_ok_tool == "evaluate":
        best = "none yet" if best_nse is None else f"{best_nse:.3f}"
        step = (f"The best NSE so far is {best}, which is not above {STOP_NSE}, so keep "
                "calibrating: choose new parameter values based on the last evaluate result "
                "and call set_parameters.")
    else:
        step = _NEXT_STEP.get(last_ok_tool, _NEXT_STEP[None])
    return (f"Your previous reply contained no tool call. {step} "
            f"Reply with exactly one tool call in this format:\n{_TOOL_CALL_FORMAT}")


def clamped_values(requested: dict) -> dict:
    """Requested parameter values outside their allowed range, with the value used."""
    out = {}
    for key, value in requested.items():
        if key not in PARAMETER_RANGES:
            continue
        try:
            req = float(value)
        except (TypeError, ValueError):
            continue
        lo, hi = PARAMETER_RANGES[key]
        if req < lo or req > hi:
            out[key] = {"requested": req, "used": min(max(req, lo), hi), "range": [lo, hi]}
    return out


def compact_tool_result(tool_name: str, data: dict, harness: str = "v2",
                        clamped: dict | None = None) -> str:
    """The tool message the model sees (a compact view of the tool's result)."""
    status = data.get("status", "error")
    if status == "error":
        return json.dumps({"status": "error", "message": data.get("message", "")})
    if tool_name == "set_parameters":
        payload = {"status": "ok", "params": data.get("validated_params", {})}
        if harness == "v2" and clamped:
            payload["clamped"] = clamped
            payload["note"] = ("Some requested values were outside their allowed range and were "
                               "set to the nearest bound; the simulation will use the values in "
                               "'params'.")
        return json.dumps(payload)
    if tool_name == "run_simulation":
        return json.dumps({
            "status": "ok",
            "run": data.get("run_number"),
            "message": "Simulation complete. Call evaluate to see metrics.",
        })
    if tool_name == "evaluate":
        payload = {"status": "ok"}
        for key in EVALUATE_METRIC_KEYS:
            if key in data:
                payload[key] = data[key]
        return json.dumps(payload)
    return json.dumps({"status": status})


def _fit_context_v1(messages, render, count_tokens, max_tokens):
    """Original trimming: drop messages[1] (the user task message goes first)."""
    prompt = render(messages)
    while count_tokens(prompt) > max_tokens and len(messages) > 2:
        messages.pop(1)
        prompt = render(messages)
    return prompt


def run_session(generate, tokenizer, gcfg: GageConfig, *, harness: str = "v2",
                max_turns: int = 100, max_tokens: int = 2048, context_len: int = 8192,
                parse_retry_limit: int = 3, transcript: list | None = None) -> dict:
    """Run one calibration session and return its record.

    If `transcript` is a list, every turn's exact prompt and completion are
    appended to it.
    """
    if harness not in HARNESSES:
        raise ValueError(f"harness must be one of {HARNESSES}, got {harness!r}")
    env = HydroEnvironment(gcfg)
    executor = ToolExecutor(env)
    messages = build_messages(gcfg, harness=harness)
    max_prompt_tokens = context_len - max_tokens

    def render(msgs):
        return tokenizer.apply_chat_template(
            msgs, tools=HYDRO_TOOLS, tokenize=False, add_generation_prompt=True,
        )

    def count_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    trim = _fit_context_v1 if harness == "v1" else fit_context
    parameter_trajectory: list[dict] = []
    turn_details: list[dict] = []
    valid_tool_calls = invalid_tool_calls = parse_failures = 0
    last_ok_tool = None
    end_reason = "max_turns"
    start = time.time()

    try:
        for turn in range(max_turns):
            logger.info("[%s] Turn %d/%d", gcfg.gage_id, turn + 1, max_turns)
            prompt = trim(messages, render, count_tokens, max_prompt_tokens)
            assistant_message = generate(prompt)
            tool_calls = parse_tool_calls(assistant_message)
            if transcript is not None:
                transcript.append({"turn": turn + 1, "prompt": prompt,
                                   "output": assistant_message, "parsed": bool(tool_calls)})

            if not tool_calls:
                messages.append({"role": "assistant", "content": assistant_message})
                best = max(env.nse_history) if env.nse_history else None
                if harness == "v2" and best is not None and best > STOP_NSE:
                    turn_details.append({"turn": turn + 1, "action": "finished",
                                         "response_preview": assistant_message[:500]})
                    end_reason = "stopped_after_threshold"
                    logger.info("[%s] Stopped after best NSE %.4f > %s; ending.",
                                gcfg.gage_id, best, STOP_NSE)
                    break
                turn_details.append({"turn": turn + 1, "action": "parse_failure",
                                     "response_preview": assistant_message[:500]})
                parse_failures += 1
                if parse_failures > parse_retry_limit:
                    end_reason = "parse_failures"
                    logger.info("[%s] Parse failed %d times (>%d), ending.",
                                gcfg.gage_id, parse_failures, parse_retry_limit)
                    break
                logger.info("[%s] Parse failed (%d/%d) — appending retry hint.",
                            gcfg.gage_id, parse_failures, parse_retry_limit)
                messages.append({"role": "user",
                                 "content": retry_hint(harness, last_ok_tool, best)})
                continue
            parse_failures = 0

            reasoning = re.sub(r"<tool_call>.*?</tool_call>", "", assistant_message,
                               flags=re.DOTALL).strip()
            json_pattern = r'\{"name"\s*:\s*"\w+"\s*,\s*"arguments"\s*:\s*\{.*?\}\}'
            reasoning = re.sub(json_pattern, "", reasoning, flags=re.DOTALL).strip()

            openai_tool_calls = []
            for call in tool_calls:
                cid = f"call_{uuid.uuid4().hex[:8]}"
                call["id"] = cid
                if not isinstance(call["arguments"], dict):
                    call["arguments"] = json.loads(call["arguments"])
                openai_tool_calls.append({
                    "id": cid,
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                })
            messages.append({"role": "assistant", "content": reasoning,
                             "tool_calls": openai_tool_calls})

            for call in tool_calls:
                tool_name = call["name"]
                logger.info("[%s] Executing %s ...", gcfg.gage_id, tool_name)
                result_data = json.loads(executor.execute(tool_name, call["arguments"]))
                ok = result_data.get("status") != "error"
                if ok:
                    valid_tool_calls += 1
                    last_ok_tool = tool_name
                else:
                    invalid_tool_calls += 1
                clamped = None
                if tool_name == "set_parameters" and result_data.get("status") == "ok":
                    parameter_trajectory.append(result_data.get("validated_params", {}))
                    clamped = clamped_values(call["arguments"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": tool_name,
                    "content": compact_tool_result(tool_name, result_data, harness, clamped),
                })
                entry = {"turn": turn + 1, "tool": tool_name}
                if clamped:
                    entry["clamped"] = sorted(clamped)
                if tool_name == "evaluate" and result_data.get("status") == "ok":
                    for key in EVALUATE_METRIC_KEYS:
                        if key in result_data:
                            entry[key] = result_data[key]
                    logger.info("[%s] NSE=%s", gcfg.gage_id, result_data.get("NSE"))
                turn_details.append(entry)
    finally:
        elapsed = time.time() - start
        env.cleanup()

    nse_hist = env.nse_history
    best_nse = max(nse_hist) if nse_hist else None
    return {
        "harness": harness,
        "gage_id": gcfg.gage_id,
        "final_nse": nse_hist[-1] if nse_hist else None,
        "best_nse": round(best_nse, 4) if best_nse is not None else None,
        "nse_history": [round(n, 4) for n in nse_hist],
        "num_turns": len(turn_details),
        "num_simulation_runs": env.run_count,
        "valid_tool_calls": valid_tool_calls,
        "invalid_tool_calls": invalid_tool_calls,
        "parse_failures_total": parse_failures,
        "parameter_trajectory": parameter_trajectory,
        "target_nse": gcfg.target_nse,
        "target_met": best_nse is not None and best_nse > gcfg.target_nse,
        "end_reason": end_reason,
        "elapsed_seconds": round(elapsed, 1),
        "turn_details": turn_details,
    }
