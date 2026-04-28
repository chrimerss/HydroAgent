#!/usr/bin/env python3
"""Convert GPT-5 calibration histories to long-horizon multi-turn SFT JSONL.

Each `calibration_history.json` becomes ONE long sequential conversation:

    system  → user  →
        round_1: assistant_set → tool_set → assistant_run → tool_run →
                 assistant_eval → tool_eval (NSE feedback) →
        round_2: assistant_set → tool_set → assistant_run → tool_run →
                 assistant_eval → tool_eval (NSE feedback) →
        ...
        round_N: same →
        final assistant summary.

Per round we use the BEST candidate (by `best_candidate_index`) so the
trajectory shows monotonic-ish improvement. The goal of this format is to
teach the model long-horizon iterative calibration — not isolated single
runs.

The user prompt mirrors `src/hydrollm/prompts.py:build_user_prompt` and
deliberately does NOT include a `target_nse`; the objective is to
"maximize NSE as much as possible".

Usage:
    python scripts/prepare_sft_data.py
    python scripts/prepare_sft_data.py --output data/sft_train.jsonl
    python scripts/prepare_sft_data.py --upload  # upload to HF dataset
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hydrollm.config import (  # noqa: E402
    PARAMETER_RANGES,
    TUNABLE_PARAMETERS,
    DEFAULT_PARAMETERS,
)
from hydrollm.prompts import SYSTEM_PROMPT  # noqa: E402


# Core 11 parameters the tool expects (exclude th/isu which are fixed).
TOOL_PARAMS = ["wm", "b", "im", "ke", "fc", "under", "leaki",
               "alpha", "beta", "alpha0", "iwu"]

GAGE_RE = re.compile(r"_(\d{8})_")


def extract_gage_id(dirname: str) -> str:
    m = GAGE_RE.search(dirname)
    return m.group(1) if m else "unknown"


def build_user_prompt(gage_id: str) -> str:
    """User prompt without `target_nse` — open-ended NSE maximization."""
    param_lines = "\n".join(
        f"  - {p}: [{PARAMETER_RANGES[p][0]}, {PARAMETER_RANGES[p][1]}]"
        for p in TUNABLE_PARAMETERS
    )
    return (
        f"Calibrate the CREST hydrologic model for USGS gage {gage_id}.\n\n"
        f"Objective: Maximize NSE as much as possible. Iterate as many "
        f"calibration rounds as needed — keep adjusting parameters as long "
        f"as you can find further improvements.\n\n"
        f"Tunable Parameters (name: [min, max]):\n{param_lines}\n\n"
        f"Fixed Parameters (do not change):\n"
        f"  - th: 10.0 (channel initiation threshold)\n"
        f"  - isu: 0.0 (initial interflow storage)\n\n"
        f"Begin by calling set_parameters."
    )


def _hermes(name: str, arguments: dict) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


def _params_for_tool(params: dict) -> dict:
    return {k: round(float(params[k]), 6) for k in TOOL_PARAMS if k in params}


def _validated_params(params: dict) -> dict:
    """Clamp + format params for the tool reply (matches HydroEnvironment)."""
    out = dict(DEFAULT_PARAMETERS)
    for k, v in params.items():
        if k in PARAMETER_RANGES:
            lo, hi = PARAMETER_RANGES[k]
            out[k] = max(lo, min(hi, float(v)))
    return {k: round(float(v), 6) for k, v in out.items()}


def _evaluate_payload(candidate: dict, nse_running: list[float]) -> dict:
    """Build a JSON evaluate-tool result mirroring HydroEnvironment.evaluate."""
    full = candidate.get("full_metrics", {}) or {}
    nse = full.get("NSE", candidate.get("metrics", {}).get("NSE"))
    if not isinstance(nse, (int, float)):
        nse = -999.0
    payload = {
        "status": "ok",
        "message": "Metrics calculated.",
        "NSE": round(float(nse), 4),
        "CC": round(float(full.get("CC", -999)), 4),
        "KGE": round(float(full.get("KGE", -999)), 4),
        "sim_peak": round(float(full.get("sim_peak", 0)), 4),
        "obs_peak": round(float(full.get("obs_peak", 0)), 4),
        "peak_ratio": round(float(full.get("peak_ratio", 0)), 4),
        "lag_hours_sim_minus_obs": round(
            float(full.get("lag_hours_sim_minus_obs", 0)), 4
        ),
    }
    return payload


def _round_reasoning(round_data: dict, cand: dict, prev_nse: float | None) -> str:
    """Synthesize an assistant reasoning paragraph for this round."""
    cand_idx = cand.get("candidate_index", 0)
    refined = (round_data.get("refined_candidates") or [])
    proposals = (round_data.get("proposals") or [])
    rationale = ""
    if cand_idx < len(refined) and refined[cand_idx].get("rationale"):
        rationale = refined[cand_idx]["rationale"]
    elif cand_idx < len(proposals) and proposals[cand_idx].get("goal"):
        rationale = proposals[cand_idx]["goal"]
    elif round_data.get("rationale"):
        rationale = round_data["rationale"]

    parts = []
    if prev_nse is None:
        parts.append(
            "Starting calibration. I'll propose an initial set of parameters "
            "based on watershed characteristics and standard CREST defaults."
        )
    else:
        parts.append(
            f"Previous best NSE = {prev_nse:.4f}. Adjusting parameters to "
            f"target the dominant residual error."
        )
    if rationale:
        parts.append(f"Rationale: {rationale.strip()}")
    return "\n\n".join(parts)


def _round_diagnosis(payload: dict, prev_nse: float | None) -> str:
    """Synthesize a short post-evaluate diagnosis."""
    nse = payload["NSE"]
    pr = payload["peak_ratio"]
    lag = payload["lag_hours_sim_minus_obs"]
    bits = [f"NSE = {nse:.4f}."]
    if pr < 0.7:
        bits.append(
            f"Peaks underestimated (peak_ratio={pr:.2f}); increase runoff "
            "generation (raise im or lower wm) or relax routing attenuation."
        )
    elif pr > 1.3:
        bits.append(
            f"Peaks overestimated (peak_ratio={pr:.2f}); raise storage (wm) "
            "or strengthen routing attenuation."
        )
    else:
        bits.append(f"Peak magnitude reasonable (peak_ratio={pr:.2f}).")
    if abs(lag) > 6:
        when = "late" if lag > 0 else "early"
        bits.append(
            f"Timing offset {abs(lag):.0f}h {when}; tune routing (alpha, beta, alpha0)."
        )
    if prev_nse is not None:
        delta = nse - prev_nse
        if delta > 0:
            bits.append(f"Improved by {delta:.4f}; keep this direction.")
        elif delta < -0.05:
            bits.append(
                f"Regressed by {abs(delta):.4f}; revert and try a different lever."
            )
    return " ".join(bits)


def _final_summary(best_nse: float, best_params: dict) -> str:
    if best_nse > 0.7:
        verdict = "Strong calibration."
    elif best_nse > 0.4:
        verdict = "Moderate calibration."
    elif best_nse > 0:
        verdict = "Weak calibration — further refinement recommended."
    else:
        verdict = "Calibration unsuccessful — strategy needs to be revisited."
    pretty = ", ".join(f"{k}={v:.4f}" for k, v in sorted(best_params.items()))
    return (
        f"Best NSE achieved: {best_nse:.4f}. {verdict} "
        f"Final parameter set: {pretty}."
    )


def trajectory_to_messages(cal_history: dict, gage_id: str) -> list[dict] | None:
    """Convert one calibration_history.json into a single multi-turn SFT example.

    Returns:
        list of message dicts in OpenAI/Hermes format, or None if the
        trajectory has no usable rounds.
    """
    rounds = cal_history.get("rounds") or []
    if not rounds:
        return None

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(gage_id)},
    ]

    nse_running: list[float] = []
    best_nse = float("-inf")
    best_params: dict = {}
    n_rounds_used = 0

    for round_data in rounds:
        candidates = round_data.get("candidates") or []
        if not candidates:
            continue
        best_idx = round_data.get("best_candidate_index", 0)
        if not (0 <= best_idx < len(candidates)):
            best_idx = 0
        cand = candidates[best_idx]
        params = cand.get("params") or {}
        if not params:
            continue

        # 1) reasoning + set_parameters
        prev_nse = nse_running[-1] if nse_running else None
        reasoning = _round_reasoning(round_data, cand, prev_nse)
        tool_args = _params_for_tool(params)
        messages.append({
            "role": "assistant",
            "content": f"{reasoning}\n\n{_hermes('set_parameters', tool_args)}",
        })
        messages.append({
            "role": "tool",
            "name": "set_parameters",
            "content": json.dumps({
                "status": "ok",
                "validated_params": _validated_params(params),
            }),
        })

        # 2) run_simulation
        messages.append({
            "role": "assistant",
            "content": f"Running EF5 simulation.\n\n{_hermes('run_simulation', {})}",
        })
        messages.append({
            "role": "tool",
            "name": "run_simulation",
            "content": json.dumps({
                "status": "ok",
                "message": "Simulation completed successfully.",
            }),
        })

        # 3) evaluate
        eval_payload = _evaluate_payload(cand, nse_running)
        messages.append({
            "role": "assistant",
            "content": f"Computing metrics.\n\n{_hermes('evaluate', {})}",
        })
        messages.append({
            "role": "tool",
            "name": "evaluate",
            "content": json.dumps(eval_payload),
        })

        # 4) diagnosis (assistant reflects on the result)
        messages.append({
            "role": "assistant",
            "content": _round_diagnosis(eval_payload, prev_nse),
        })

        nse_running.append(eval_payload["NSE"])
        if eval_payload["NSE"] > best_nse:
            best_nse = eval_payload["NSE"]
            best_params = dict(tool_args)
        n_rounds_used += 1

    if n_rounds_used == 0:
        return None

    # Final summary turn so the model learns to recognize when it's done.
    messages.append({"role": "assistant", "content": _final_summary(best_nse, best_params)})

    return messages


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="data/sets_for_SFT_RL",
                   help="Directory containing per-trajectory experiment dirs.")
    p.add_argument("--output", default="data/sft_train.jsonl")
    p.add_argument("--upload", action="store_true",
                   help="Upload to chrimerss/hydro_cali_agent_example after writing.")
    p.add_argument("--hf-repo", default="chrimerss/hydro_cali_agent_example")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    in_dir = repo / args.input_dir
    out_path = repo / args.output

    files = sorted(in_dir.rglob("calibration_history.json"))
    print(f"Found {len(files)} calibration histories in {in_dir}")

    examples: list[dict] = []
    skipped = 0
    for f in files:
        gid = extract_gage_id(f.parent.parent.name)
        try:
            cal = json.loads(f.read_text())
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            skipped += 1
            continue
        msgs = trajectory_to_messages(cal, gid)
        if msgs is None:
            skipped += 1
            continue
        # Capture best NSE across the trajectory for quality weighting.
        # Walk the messages and pull the NSE values out of evaluate tool replies.
        nse_seq: list[float] = []
        for m in msgs:
            if m["role"] == "tool" and m.get("name") == "evaluate":
                try:
                    payload = json.loads(m["content"])
                    if isinstance(payload.get("NSE"), (int, float)):
                        nse_seq.append(float(payload["NSE"]))
                except Exception:
                    pass
        best_nse = max(nse_seq) if nse_seq else -1.0
        # weight in [0.1, 1.0]
        weight = max(0.1, min(1.0, (best_nse + 1.0) / 2.0))
        examples.append({
            "messages": msgs,
            "metadata": {
                "gage_id": gid,
                "exp_dir": f.parent.parent.name,
                "n_rounds": len(nse_seq),
                "best_nse": round(best_nse, 4),
                "weight": round(weight, 4),
            },
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fp:
        for ex in examples:
            fp.write(json.dumps(ex) + "\n")

    nse_vals = [e["metadata"]["best_nse"] for e in examples]
    n_rounds = [e["metadata"]["n_rounds"] for e in examples]
    print("=" * 60)
    print(f"Wrote {len(examples)} trajectories to {out_path}")
    print(f"Skipped: {skipped}")
    if nse_vals:
        print(
            f"best_nse: min={min(nse_vals):.3f}  median={sorted(nse_vals)[len(nse_vals)//2]:.3f}  max={max(nse_vals):.3f}"
        )
    if n_rounds:
        print(
            f"rounds:   min={min(n_rounds)}  median={sorted(n_rounds)[len(n_rounds)//2]}  max={max(n_rounds)}"
        )

    if args.upload and examples:
        from huggingface_hub import HfApi
        import os
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="sft_train.jsonl",
            repo_id=args.hf_repo,
            repo_type="dataset",
            commit_message="Sequential long-horizon SFT data (no target_nse, NSE-max objective)",
        )
        print(f"Uploaded to {args.hf_repo}/sft_train.jsonl")


if __name__ == "__main__":
    main()
