"""Tests for prompt construction and context trimming."""

from hydrollm.prompts import SYSTEM_PROMPT, fit_context


def _render(msgs):
    return "|".join(f"{m['role']}:{m['content']}" for m in msgs)


def _session():
    return [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "TASK"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "hint"},
        {"role": "assistant", "content": "a3"},
        {"role": "tool", "content": "t3"},
    ]


def test_fit_context_noop_when_it_fits():
    msgs = _session()
    prompt = fit_context(msgs, _render, len, 10_000)
    assert msgs == _session()
    assert prompt == _render(_session())


def test_fit_context_keeps_system_and_task_and_drops_whole_exchanges():
    msgs = _session()
    full = len(_render(_session()))
    prompt = fit_context(msgs, _render, len, full - 1)
    # The oldest exchange (a1 + its tool result) goes first, together.
    assert [m["content"] for m in msgs] == ["S", "TASK", "a2", "hint", "a3", "t3"]
    assert prompt == _render(msgs)


def test_fit_context_never_drops_the_task_message():
    msgs = _session()
    fit_context(msgs, _render, len, 1)
    assert [m["content"] for m in msgs[:2]] == ["S", "TASK"]
    assert len(msgs) == 3  # system, task, and the last remaining message


def test_fit_context_no_orphaned_tool_results():
    msgs = _session()
    fit_context(msgs, _render, len, len(_render(_session())) - 1)
    assert msgs[2]["role"] == "assistant"


def test_system_prompt_has_explicit_stop_threshold():
    assert "Stop when NSE > 0.8." in SYSTEM_PROMPT
    assert "target" not in SYSTEM_PROMPT
