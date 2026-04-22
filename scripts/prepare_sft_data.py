#!/usr/bin/env python3
"""Convert GPT-4o calibration histories to multi-turn chat JSONL for SFT.

Scans all calibration_history.json files in sets_for_SFT_RL/, converts each
round+candidate into a multi-turn tool-calling conversation that matches the
HydroLLM tool schema (set_parameters → run_simulation → evaluate).

Includes ALL rounds and candidates (including unsuccessful ones) to give the
model a rich corpus that covers both good and bad parameter choices.

Quality weighting: each example gets a `weight` field based on the best NSE
achieved in that trajectory, so the training can up-weight successful episodes.

Usage:
    python scripts/prepare_sft_data.py
    python scripts/prepare_sft_data.py --output data/sft_train.jsonl
    python scripts/prepare_sft_data.py --upload  # Upload to HuggingFace
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

PARAMETER_RANGES = {
    "wm": (0.1, 10.0),
    "b": (0.000001, 3.0),
    "im": (0.0, 1.0),
    "ke": (0.8, 1.2),
    "fc": (0.1, 2.0),
    "under": (0.1, 10.0),
    "leaki": (0.1, 10.0),
    "alpha": (0.1, 3.0),
    "beta": (0.1, 3.0),
    "alpha0": (0.0, 3.0),
    "iwu": (0.1, 100.0),
    "th": (10.0, 10.0),
    "isu": (0.0, 0.0),
}

TUNABLE_PARAMETERS = [k for k, (lo, hi) in PARAMETER_RANGES.items() if lo != hi]

# Core 11 parameters the tool expects (exclude th, isu which are fixed)
TOOL_PARAMS = ["wm", "b", "im", "ke", "fc", "under", "leaki",
               "alpha", "beta", "alpha0", "iwu"]


def extract_gage_id(dirname: str) -> str:
    """Extract gage ID from experiment directory name.

    Examples:
        conus_batch_test_exp_001_01400500_2018_... → 01400500
        exp_001_02338660_2018_... → 02338660
        debug_03076500_2018_... → 03076500
    """
    # Pattern: look for 8-digit USGS gage ID
    match = re.search(r'_(\d{8})_', dirname)
    if match:
        return match.group(1)
    # Fallback: look for any 7-8 digit sequence
    match = re.search(r'(\d{7,8})', dirname)
    if match:
        return match.group(1)
    return "unknown"


def build_user_prompt(gage_id: str) -> str:
    """Build the user prompt for a specific gage (simplified from prompts.py)."""
    param_lines = []
    for param in TUNABLE_PARAMETERS:
        lo, hi = PARAMETER_RANGES[param]
        param_lines.append(f"  - {param}: [{lo}, {hi}]")
    param_table = "\n".join(param_lines)

    return f"""\
Calibrate the CREST hydrologic model for USGS gage {gage_id}.

Objective: Achieve NSE > 0.8075

Tunable Parameters (name: [min, max]):
{param_table}

Fixed Parameters (do not change):
  - th: 10.0 (channel initiation threshold)
  - isu: 0.0 (initial interflow storage)

Instructions:
1. First, propose an initial set of parameter values and run a simulation
2. Analyze the results and identify the main sources of error
3. Adjust parameters based on your hydrologic understanding
4. Repeat until NSE > 0.8075 or you cannot improve further

Begin calibration.\
"""


def build_tool_call(name: str, arguments: dict) -> str:
    """Format a tool call in Hermes/Qwen style."""
    call_data = {"name": name, "arguments": arguments}
    return f"<tool_call>\n{json.dumps(call_data)}\n</tool_call>"


def build_set_parameters_args(params: dict) -> dict:
    """Extract the 11 tunable parameters for set_parameters tool call."""
    return {k: params[k] for k in TOOL_PARAMS if k in params}


def build_sim_result(candidate: dict) -> dict:
    """Build a simulated run_simulation result from candidate metrics."""
    metrics = candidate.get("metrics", {})
    full_metrics = candidate.get("full_metrics", {})

    # Use full_metrics NSE if available, otherwise aggregate
    nse = full_metrics.get("NSE", metrics.get("NSE", -1.0))

    return {
        "status": "ok",
        "nse": round(nse, 4),
        "peak_sim_m3s": round(full_metrics.get("sim_peak", 0), 2),
        "peak_obs_m3s": round(full_metrics.get("obs_peak", 0), 2),
        "volume_ratio": round(
            full_metrics.get("peak_ratio", metrics.get("peak_ratio", 0)), 3
        ),
        "timing_error_hours": abs(int(
            full_metrics.get("lag_hours_sim_minus_obs",
                             metrics.get("lag_hours", 0))
        )),
    }


def build_eval_result(nse_history: list[float], current_params: dict,
                       target_nse: float = 0.8075) -> dict:
    """Build an evaluate tool result."""
    best_nse = max(nse_history) if nse_history else None
    return {
        "current_nse": round(nse_history[-1], 4) if nse_history else None,
        "best_nse": round(best_nse, 4) if best_nse is not None else None,
        "nse_history": [round(n, 4) for n in nse_history],
        "num_runs": len(nse_history),
        "target_nse": target_nse,
        "target_met": best_nse is not None and best_nse > target_nse,
        "current_params": {k: round(v, 6) for k, v in current_params.items()},
    }


def synthesize_reasoning(
    candidate: dict,
    round_data: dict,
    round_idx: int,
    cand_idx: int,
    nse_history: list[float],
    is_first: bool,
) -> str:
    """Synthesize assistant reasoning text from calibration data.

    Combines proposal goals, rationale, and diagnostic analysis.
    """
    parts = []

    # Try to find the matching proposal/refined candidate
    proposals = round_data.get("proposals", [])
    refined = round_data.get("refined_candidates", [])

    # Match refined candidate to this index if possible
    if cand_idx < len(refined):
        r = refined[cand_idx]
        if r.get("rationale"):
            parts.append(f"**Rationale**: {r['rationale']}")
    elif cand_idx < len(proposals):
        p = proposals[cand_idx]
        if p.get("goal"):
            parts.append(f"**Strategy**: {p['goal']}")

    # Add context based on position
    if is_first:
        parts.insert(0, f"I'll begin calibrating by proposing an initial parameter set for round {round_idx}.")
    else:
        # Refer to previous results
        if nse_history:
            prev_nse = nse_history[-1]
            parts.insert(0,
                f"The previous simulation achieved NSE = {prev_nse:.4f}. "
                f"Let me adjust parameters based on the hydrograph diagnostics."
            )

    if not parts:
        parts.append(
            f"Proposing parameter set {cand_idx} for round {round_idx} "
            f"based on watershed analysis."
        )

    return "\n\n".join(parts)


def synthesize_analysis(candidate: dict, nse_history: list[float]) -> str:
    """Synthesize assistant analysis after seeing simulation results."""
    metrics = candidate.get("metrics", {})
    full_metrics = candidate.get("full_metrics", {})

    nse = full_metrics.get("NSE", metrics.get("NSE", -1.0))
    peak_ratio = full_metrics.get("peak_ratio", metrics.get("peak_ratio", 0))
    lag = full_metrics.get("lag_hours_sim_minus_obs", metrics.get("lag_hours", 0))

    parts = [f"NSE = {nse:.4f}."]

    # Peak analysis
    if peak_ratio < 0.7:
        parts.append(f"Peak ratio is {peak_ratio:.2f} — simulated peaks are too low. "
                     "Need to increase runoff generation (lower wm, higher b or im) "
                     "or reduce routing attenuation.")
    elif peak_ratio > 1.3:
        parts.append(f"Peak ratio is {peak_ratio:.2f} — simulated peaks are too high. "
                     "Need to increase infiltration/storage (higher wm, lower im) "
                     "or increase routing attenuation.")
    else:
        parts.append(f"Peak ratio is {peak_ratio:.2f} — within reasonable range.")

    # Timing analysis
    if abs(lag) > 10:
        direction = "late" if lag > 0 else "early"
        parts.append(f"Timing error: {abs(lag):.0f} hours {direction}. "
                     "Need to adjust routing parameters (alpha, beta, alpha0).")
    elif abs(lag) > 3:
        direction = "late" if lag > 0 else "early"
        parts.append(f"Minor timing offset: {abs(lag):.0f} hours {direction}.")
    else:
        parts.append("Timing is well-aligned.")

    # Progress
    if len(nse_history) > 1:
        improvement = nse_history[-1] - nse_history[-2]
        if improvement > 0:
            parts.append(f"NSE improved by {improvement:.4f} from previous run.")
        elif improvement < -0.1:
            parts.append(f"NSE decreased by {abs(improvement):.4f}. "
                         "The parameter change was too aggressive.")

    # Target check
    if nse > 0.8075:
        parts.append("✅ Target NSE > 0.8075 achieved!")
    elif nse > 0.6:
        parts.append("Getting close to the target. Fine-tuning needed.")
    elif nse > 0.3:
        parts.append("Moderate performance. Significant adjustments still needed.")
    elif nse > 0:
        parts.append("Below target. Need substantial parameter changes.")
    else:
        parts.append("Negative NSE — model is performing worse than the mean. "
                     "Fundamental parameter rethinking needed.")

    return " ".join(parts)


def convert_calibration_to_conversations(
    cal_history: dict,
    gage_id: str,
    exp_dir: str,
) -> list[dict]:
    """Convert a single calibration history to list of conversation dicts.

    Each conversation covers the full multi-round calibration trajectory,
    including ALL candidates per round.

    Returns a list of conversation dicts, each with:
        - messages: list of {role, content} dicts
        - metadata: {gage_id, exp_dir, best_nse, weight}
    """
    rounds = cal_history.get("rounds", [])
    best_info = cal_history.get("best") or {}
    best_metrics = best_info.get("metrics") or {}
    best_overall_nse = best_metrics.get("NSE", -1.0)

    conversations = []

    # Strategy: create one conversation per round, showing all candidates
    # This gives the model exposure to multiple parameter proposals and
    # their outcomes within each calibration round
    nse_history_global = []

    for round_data in rounds:
        round_idx = round_data.get("round_index", 0)
        candidates = round_data.get("candidates", [])

        if not candidates:
            continue

        for cand in candidates:
            cand_idx = cand.get("candidate_index", 0)
            params = cand.get("params", {})
            if not params:
                continue

            # Build a single-trajectory conversation for this candidate
            messages = []

            # System prompt
            messages.append({"role": "system", "content": SYSTEM_PROMPT})

            # User prompt
            messages.append({"role": "user", "content": build_user_prompt(gage_id)})

            # If we have prior rounds' best result, include context
            is_first = (round_idx <= 1 and cand_idx == 0)

            # Assistant reasoning + tool call
            reasoning = synthesize_reasoning(
                cand, round_data, round_idx, cand_idx,
                nse_history_global, is_first
            )

            tool_args = build_set_parameters_args(params)
            tool_call_text = build_tool_call("set_parameters", tool_args)

            messages.append({
                "role": "assistant",
                "content": f"{reasoning}\n\n{tool_call_text}",
            })

            # Tool result: set_parameters response
            validated_params = {k: round(v, 6) for k, v in params.items()
                               if k in PARAMETER_RANGES}
            messages.append({
                "role": "tool",
                "name": "set_parameters",
                "content": json.dumps({
                    "status": "ok",
                    "validated_params": validated_params,
                }),
            })

            # Assistant calls run_simulation
            messages.append({
                "role": "assistant",
                "content": f"Parameters set. Running EF5/CREST simulation...\n\n"
                           f"{build_tool_call('run_simulation', {})}",
            })

            # Tool result: simulation results
            sim_result = build_sim_result(cand)
            messages.append({
                "role": "tool",
                "name": "run_simulation",
                "content": json.dumps(sim_result),
            })

            # Track NSE
            cand_nse = sim_result["nse"]
            local_nse_history = list(nse_history_global) + [cand_nse]

            # Assistant analysis
            analysis = synthesize_analysis(cand, local_nse_history)
            messages.append({
                "role": "assistant",
                "content": analysis,
            })

            # Optionally call evaluate at end
            eval_result = build_eval_result(
                local_nse_history, params
            )
            messages.append({
                "role": "assistant",
                "content": f"Let me check the overall calibration progress.\n\n"
                           f"{build_tool_call('evaluate', {})}",
            })
            messages.append({
                "role": "tool",
                "name": "evaluate",
                "content": json.dumps(eval_result),
            })

            # Final assistant summary
            if cand_nse > 0.8075:
                final_msg = (
                    f"Calibration target achieved! Best NSE = {cand_nse:.4f} "
                    f"exceeds the target of 0.8075. The optimized parameters are: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in sorted(tool_args.items()))
                )
            elif cand_nse > 0.5:
                final_msg = (
                    f"Good progress with NSE = {cand_nse:.4f}. "
                    f"Further refinement of parameters is recommended, "
                    f"focusing on peak timing and volume balance."
                )
            else:
                final_msg = (
                    f"Current NSE = {cand_nse:.4f} is below target. "
                    f"Need to revisit the parameter strategy — "
                    f"consider adjusting the process that dominates the error."
                )
            messages.append({"role": "assistant", "content": final_msg})

            # Compute quality weight
            # Scale: NSE in [-1, 1] → weight in [0.1, 1.0]
            weight = max(0.1, min(1.0, (cand_nse + 1.0) / 2.0))

            conversations.append({
                "messages": messages,
                "metadata": {
                    "gage_id": gage_id,
                    "exp_dir": exp_dir,
                    "round_index": round_idx,
                    "candidate_index": cand_idx,
                    "nse": cand_nse,
                    "best_nse": best_overall_nse,
                    "weight": round(weight, 4),
                },
            })

        # Update global NSE history with best candidate from this round
        best_idx = round_data.get("best_candidate_index", 0)
        if best_idx < len(candidates):
            best_cand = candidates[best_idx]
            best_cand_nse = best_cand.get("full_metrics", best_cand.get("metrics", {})).get("NSE", -1.0)
            nse_history_global.append(best_cand_nse)

    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Convert calibration histories to SFT training data"
    )
    parser.add_argument(
        "--input-dir", type=str,
        default="sets_for_SFT_RL",
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/sft_train.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload dataset to HuggingFace after preparation",
    )
    parser.add_argument(
        "--hf-repo", type=str,
        default="chrimerss/hydro_cali_agent_example",
        help="HuggingFace dataset repo ID for upload",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_dir = project_root / args.input_dir
    output_path = project_root / args.output

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    # Find all calibration_history.json files
    cal_files = sorted(input_dir.rglob("calibration_history.json"))
    print(f"Found {len(cal_files)} calibration history files")

    # Process all files
    all_conversations = []
    stats = {
        "total_files": len(cal_files),
        "total_examples": 0,
        "gage_ids": set(),
        "nse_values": [],
        "weights": [],
        "skipped": 0,
        "errors": 0,
    }

    for cal_file in cal_files:
        exp_dir = cal_file.parent.parent.name
        gage_id = extract_gage_id(exp_dir)
        stats["gage_ids"].add(gage_id)

        try:
            with open(cal_file) as f:
                cal_history = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ERROR reading {cal_file}: {e}")
            stats["errors"] += 1
            continue

        conversations = convert_calibration_to_conversations(
            cal_history, gage_id, exp_dir
        )

        if not conversations:
            stats["skipped"] += 1
            continue

        all_conversations.extend(conversations)
        for conv in conversations:
            meta = conv["metadata"]
            stats["nse_values"].append(meta["nse"])
            stats["weights"].append(meta["weight"])

    stats["total_examples"] = len(all_conversations)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for conv in all_conversations:
            f.write(json.dumps(conv) + "\n")

    # Print statistics
    print(f"\n{'='*60}")
    print(f"SFT Data Preparation Complete")
    print(f"{'='*60}")
    print(f"Input files:       {stats['total_files']}")
    print(f"Unique gages:      {len(stats['gage_ids'])}")
    print(f"Total examples:    {stats['total_examples']}")
    print(f"Skipped (empty):   {stats['skipped']}")
    print(f"Errors:            {stats['errors']}")

    if stats["nse_values"]:
        import numpy as np
        nse_arr = np.array(stats["nse_values"])
        print(f"\nNSE Distribution:")
        print(f"  Min:    {nse_arr.min():.4f}")
        print(f"  25th:   {np.percentile(nse_arr, 25):.4f}")
        print(f"  Median: {np.median(nse_arr):.4f}")
        print(f"  75th:   {np.percentile(nse_arr, 75):.4f}")
        print(f"  Max:    {nse_arr.max():.4f}")
        print(f"  NSE > 0:    {np.sum(nse_arr > 0)}/{len(nse_arr)}")
        print(f"  NSE > 0.5:  {np.sum(nse_arr > 0.5)}/{len(nse_arr)}")
        print(f"  NSE > 0.8:  {np.sum(nse_arr > 0.8)}/{len(nse_arr)}")

        weight_arr = np.array(stats["weights"])
        print(f"\nWeight Distribution:")
        print(f"  Min:    {weight_arr.min():.4f}")
        print(f"  Median: {np.median(weight_arr):.4f}")
        print(f"  Max:    {weight_arr.max():.4f}")

    # Count tokens (rough estimate)
    total_chars = sum(
        sum(len(m.get("content", "")) for m in conv["messages"])
        for conv in all_conversations
    )
    avg_chars = total_chars / max(len(all_conversations), 1)
    print(f"\nApprox. total chars: {total_chars:,}")
    print(f"Avg chars/example:  {avg_chars:,.0f}")
    print(f"Approx avg tokens:  {avg_chars / 4:,.0f}")

    print(f"\nGage IDs: {sorted(stats['gage_ids'])}")
    print(f"\nOutput written to: {output_path}")

    # Upload to HuggingFace if requested
    if args.upload:
        print(f"\nUploading to HuggingFace: {args.hf_repo}")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(output_path),
                path_in_repo="sft_train.jsonl",
                repo_id=args.hf_repo,
                repo_type="dataset",
            )
            print(f"✓ Uploaded to https://huggingface.co/datasets/{args.hf_repo}")
        except Exception as e:
            print(f"Upload failed: {e}")
            print("You can upload manually later with:")
            print(f"  huggingface-cli upload {args.hf_repo} {output_path} sft_train.jsonl --repo-type dataset")


if __name__ == "__main__":
    main()
