"""Compare harness v1 and v2 on sampled held-out runs.

    python curc/compare_harness.py [--root results/sampled] [--seeds 1-5]

Expects <root>/<harness>_s<seed>/<gage>.json from curc/eval.py. Each seed is a
full replicate of the held-out evaluation, so the per-seed clipped mean (mean
over gages of best NSE clipped to [-1, 1]) gives 5 independent values per
harness; the difference is tested with Welch's t-test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
GAGES = ["02338660", "01403060", "06279500", "07144100"]


def load(root: Path, harness: str, seeds: list[int]) -> dict:
    runs = {}
    for s in seeds:
        d = root / f"{harness}_s{s}"
        rows = {}
        for g in GAGES:
            p = d / f"{g}.json"
            if p.exists():
                r = json.loads(p.read_text())
                rows[g] = {
                    "best": r["best_nse"],
                    "clipped": float(np.clip(r["best_nse"], -1, 1)) if r["best_nse"] is not None else -1.0,
                    "sims": r["num_simulation_runs"],
                    "no_call": sum(1 for t in r["turn_details"] if t.get("action") == "parse_failure"),
                    "end": r.get("end_reason", "?"),
                    "above_stop": r["best_nse"] is not None and r["best_nse"] > 0.8,
                }
        if len(rows) == len(GAGES):
            runs[s] = rows
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results" / "sampled"))
    ap.add_argument("--seeds", default="1-5")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    root = Path(a.root)

    per_seed = {}
    for h in ("v1", "v2"):
        runs = load(root, h, seeds)
        print(f"\n== harness {h}: {len(runs)} complete seed(s) {sorted(runs)}")
        if not runs:
            continue
        print(f"  {'gage':9s} {'best NSE mean ± sd':>20s} {'clipped mean':>13s} {'sims':>6s} {'no-call':>8s} {'>0.8':>5s}  end reasons")
        for g in GAGES:
            rows = [runs[s][g] for s in runs]
            best = np.array([r["best"] for r in rows], float)
            ends = {}
            for r in rows:
                ends[r["end"]] = ends.get(r["end"], 0) + 1
            print(f"  {g:9s} {best.mean():>10.3f} ± {best.std(ddof=1) if len(best) > 1 else float('nan'):<7.3f} "
                  f"{np.mean([r['clipped'] for r in rows]):>13.3f} {np.mean([r['sims'] for r in rows]):>6.1f} "
                  f"{np.mean([r['no_call'] for r in rows]):>8.1f} {sum(r['above_stop'] for r in rows):>3d}/{len(rows)}  {ends}")
        per_seed[h] = np.array([np.mean([runs[s][g]["clipped"] for g in GAGES]) for s in sorted(runs)])
        x = per_seed[h]
        print(f"  clipped mean per seed: {np.round(x, 3).tolist()}  ->  {x.mean():.3f} ± {x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else float('nan'):.3f} (SE)")

    if all(h in per_seed and per_seed[h].size > 1 for h in ("v1", "v2")):
        d = per_seed["v2"].mean() - per_seed["v1"].mean()
        t = stats.ttest_ind(per_seed["v2"], per_seed["v1"], equal_var=False)
        print(f"\nv2 - v1 clipped mean: {d:+.3f}  (Welch t = {t.statistic:.2f}, p = {t.pvalue:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
