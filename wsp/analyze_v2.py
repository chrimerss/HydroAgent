"""Weight edits vs the unedited model under harness v2 (sampled, 5 seeds).

    python wsp/analyze_v2.py

Edited models: results/wsp_v2/trial_<k>_s<seed>/<gage>.json; the unedited
model: results/sampled/v2_s<seed>/<gage>.json. Both come from curc/eval.py
with the same protocol (harness v2, temperature 0.7, top-p 0.8, top-k 20,
engine seed 1-5, one session at a time). Each seed is a full replicate of the
held-out set, so each model has 5 per-seed clipped means (mean over gages of
best NSE clipped to [-1, 1]); models are compared with Welch's t-test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
GAGES = ["02338660", "01403060", "06279500", "07144100"]
TAGS = ["base"] + [f"trial_{k}" for k in range(1, 6)]


def run_dir(tag: str, k: int) -> Path:
    if tag == "base":
        return REPO / "results" / "sampled" / f"v2_s{k}"
    return REPO / "results" / "wsp_v2" / f"{tag}_s{k}"


def load(tag: str, n: int = 5) -> list[dict] | None:
    reps = []
    for k in range(1, n + 1):
        rows = {}
        for g in GAGES:
            p = run_dir(tag, k) / f"{g}.json"
            if not p.exists():
                return None
            r = json.loads(p.read_text())
            rows[g] = {
                "best": r["best_nse"], "clipped": float(np.clip(r["best_nse"], -1, 1)),
                "sims": r["num_simulation_runs"],
                "no_call": sum(1 for t in r["turn_details"] if t.get("action") == "parse_failure"),
                "above": r["best_nse"] > 0.8, "end": r["end_reason"],
            }
        reps.append(rows)
    return reps


def rep_means(reps):
    return np.array([np.mean([rep[g]["clipped"] for g in GAGES]) for rep in reps])


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    data = {t: load(t) for t in TAGS}
    data = {t: v for t, v in data.items() if v is not None}
    if "base" not in data:
        print("baseline not complete yet:", sorted(data))
        return 1
    b = rep_means(data["base"])

    print(f"{'model':9s} " + " ".join(f"{g:>9s}" for g in GAGES)
          + f" {'clipped':>8s} {'± SE':>6s} {'vs base':>8s} {'p':>6s} {'sims':>5s} {'nocall':>6s} {'>0.8':>5s}")
    pooled = []
    for t, reps in data.items():
        m = rep_means(reps)
        per_gage = [np.mean([rep[g]["best"] for rep in reps]) for g in GAGES]
        sims = np.mean([rep[g]["sims"] for rep in reps for g in GAGES])
        nc = np.mean([rep[g]["no_call"] for rep in reps for g in GAGES])
        above = sum(rep[g]["above"] for rep in reps for g in GAGES)
        if t == "base":
            diff, p = "", ""
        else:
            pooled.extend(m.tolist())
            diff = f"{m.mean() - b.mean():+.3f}"
            p = f"{stats.ttest_ind(m, b, equal_var=False).pvalue:.3f}"
        print(f"{t:9s} " + " ".join(f"{x:>9.3f}" for x in per_gage)
              + f" {m.mean():>8.3f} {m.std(ddof=1) / np.sqrt(m.size):>6.3f} {diff:>8s} {p:>6s} {sims:>5.1f} {nc:>6.1f} {above:>3d}/20")

    if pooled:
        pooled = np.array(pooled)
        t = stats.ttest_ind(pooled, b, equal_var=False)
        print(f"\nall edits pooled ({pooled.size} replicates): {pooled.mean():.3f} vs base {b.mean():.3f}: "
              f"{pooled.mean() - b.mean():+.3f} (Welch p = {t.pvalue:.3f}); "
              f"edits above base: {sum(rep_means(data[x]).mean() > b.mean() for x in data if x != 'base')}/{len(data) - 1}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
