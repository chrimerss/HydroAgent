"""Compare baseline, WSP-edited and matched-random-control models on the held-out gages.

    python wsp/analyze.py [--json out.json]

Primary (pre-registered) metric: mean over gages of best NSE clipped to [-1, 1]
(as in src/hydrollm/verl_reward.py). Also: raw best NSE, simulations, turns
without a parseable tool call, invalid tool calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GAGES = ["02338660", "01403060", "06279500", "07144100"]
BASE = REPO / "results" / "baseline" / "Qwen3-4B-Instruct-2507"
WSP = REPO / "results" / "wsp"


def load_run(d: Path) -> dict | None:
    rows = {}
    for g in GAGES:
        p = d / f"{g}.json"
        if not p.exists():
            return None
        r = json.loads(p.read_text())
        rows[g] = {
            "best": r["best_nse"],
            "clipped": float(np.clip(r["best_nse"], -1, 1)) if r["best_nse"] is not None else -1.0,
            "sims": r["num_simulation_runs"],
            "no_call_turns": sum(1 for t in r["turn_details"] if t.get("action") == "parse_failure"),
            "invalid": r["invalid_tool_calls"],
            "evals": len(r["nse_history"]),
        }
    return rows


def summarize(rows: dict) -> dict:
    return {
        "clipped_mean": float(np.mean([rows[g]["clipped"] for g in GAGES])),
        "raw_mean": float(np.mean([rows[g]["best"] for g in GAGES])),
        "sims": float(np.mean([rows[g]["sims"] for g in GAGES])),
        "no_call_turns": float(np.mean([rows[g]["no_call_turns"] for g in GAGES])),
        "invalid": float(np.mean([rows[g]["invalid"] for g in GAGES])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()

    base = load_run(BASE)
    runs = {"baseline": base}
    for k in range(1, 6):
        for tag in (f"trial_{k}", f"trial_{k}_control"):
            r = load_run(WSP / tag)
            if r is not None:
                runs[tag] = r

    print("Per-gage best NSE (raw; clipped mean uses [-1, 1])")
    print(f"{'model':18s}" + "".join(f"{g:>10s}" for g in GAGES) + f"{'clipped':>10s}{'sims':>7s}{'nocall':>8s}{'invalid':>8s}")
    sums = {}
    for tag, rows in runs.items():
        s = summarize(rows)
        sums[tag] = s
        print(f"{tag:18s}" + "".join(f"{rows[g]['best']:>10.3f}" for g in GAGES)
              + f"{s['clipped_mean']:>10.3f}{s['sims']:>7.1f}{s['no_call_turns']:>8.1f}{s['invalid']:>8.1f}")

    b = sums["baseline"]["clipped_mean"]
    d_edit, d_ctrl, d_pair = [], [], []
    print("\nClipped-mean change vs baseline")
    for k in range(1, 6):
        e, c = sums.get(f"trial_{k}"), sums.get(f"trial_{k}_control")
        line = f"  trial_{k}: edit {e['clipped_mean'] - b:+.3f}" if e else f"  trial_{k}: edit pending"
        if e:
            d_edit.append(e["clipped_mean"] - b)
        if c:
            d_ctrl.append(c["clipped_mean"] - b)
            line += f" | control {c['clipped_mean'] - b:+.3f}"
            if e:
                d_pair.append(e["clipped_mean"] - c["clipped_mean"])
                line += f" | edit - control {e['clipped_mean'] - c['clipped_mean']:+.3f}"
        print(line)

    def ms(x):
        x = np.array(x, float)
        if x.size == 0:
            return "n=0"
        sd = x.std(ddof=1) if x.size > 1 else float("nan")
        return f"mean {x.mean():+.3f}, sd {sd:.3f}, se {sd / np.sqrt(x.size):.3f}, n={x.size}, positive {int((x > 0).sum())}/{x.size}"

    print("\nAcross trials:")
    print(f"  edit - baseline:    {ms(d_edit)}")
    print(f"  control - baseline: {ms(d_ctrl)}")
    print(f"  edit - control:     {ms(d_pair)}")

    edits = [t for t in runs if t.startswith("trial_") and not t.endswith("control")]
    ctrls = [t for t in runs if t.endswith("control")]
    for name, group in (("edits", edits), ("controls", ctrls)):
        if group:
            print(f"\nMean process metrics, {name} ({len(group)}) vs baseline:")
            for m in ("sims", "no_call_turns", "invalid"):
                print(f"  {m:14s} {np.mean([sums[t][m] for t in group]):6.1f}  (baseline {sums['baseline'][m]:.1f})")

    if a.json:
        Path(a.json).write_text(json.dumps({"runs": runs, "summary": sums}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
