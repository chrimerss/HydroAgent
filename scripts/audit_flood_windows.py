"""Audit obs CSVs to score every 60-day window per gage by flood-event quality.

For each candidate gage (excluding 02338660), slide a 60-day window across
the period intersected with MRMS coverage (2017-10-01 → 2019-12-31). Score
each window by:
  - peak / median (must be >> 1)
  - rise length: hours from start of rise to peak (>= 12h)
  - recession length: hours from peak back to (peak + median)/2 (>= 24h)
  - peak must NOT be within 5 days of either window edge (so the hydrograph
    has visible context on both sides).

Print best window per gage; user (or follow-up code) picks the top 10.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

GAUGE_DIR = Path("/Users/allen/Documents/Python/hydroGPT/data/gauge")
MRMS_BEGIN = pd.Timestamp("2017-10-01", tz="UTC")
MRMS_END = pd.Timestamp("2019-12-31 23:00", tz="UTC")
WINDOW_DAYS = 60
EDGE_BUFFER_DAYS = 5
STRIDE_DAYS = 5  # how far apart to start candidate windows

# Excluded: 02338660 reserved for testing.
CANDIDATES = [
    "01403060", "02294781", "02312000", "06279500", "07144100",
    "07195430", "11043000", "11152000", "11179000", "11376000",
    "11383500", "14207500", "14301000",
]


def score_window(df: pd.DataFrame) -> dict | None:
    """Return scoring dict for a window or None if invalid."""
    q = df["discharge_m3s"].to_numpy()
    if len(q) < 24 * 7:  # need at least a week of data
        return None
    if np.isnan(q).any():
        # Allow up to 10% NaN
        if np.isnan(q).sum() / len(q) > 0.1:
            return None
        q = pd.Series(q).interpolate().to_numpy()
    if np.nanmax(q) <= 0:
        return None

    median = float(np.nanmedian(q))
    if median <= 0:
        median = max(float(np.nanmean(q)), 1e-3)
    peak = float(np.nanmax(q))
    peak_idx = int(np.nanargmax(q))

    # Edge buffer: peak must be at least 5 days from either edge
    edge_h = EDGE_BUFFER_DAYS * 24
    if peak_idx < edge_h or peak_idx > len(q) - edge_h:
        return None

    # Rising limb: walk back from peak until value drops below midpoint
    midpoint = (peak + median) / 2.0
    rise_start = peak_idx
    while rise_start > 0 and q[rise_start] > midpoint:
        rise_start -= 1
    rise_h = peak_idx - rise_start

    # Recession: walk forward until value drops below midpoint
    rec_end = peak_idx
    while rec_end < len(q) - 1 and q[rec_end] > midpoint:
        rec_end += 1
    rec_h = rec_end - peak_idx

    if rise_h < 6 or rec_h < 12:
        return None

    peak_ratio = peak / median
    return {
        "peak": peak,
        "median": median,
        "peak_ratio": peak_ratio,
        "rise_h": rise_h,
        "rec_h": rec_h,
        # Composite score: emphasize rise+recession quality plus magnitude
        "score": float(np.log10(peak_ratio + 1) * np.sqrt(rise_h * rec_h)),
    }


def best_window_for_gage(gage_id: str) -> dict | None:
    csv = GAUGE_DIR / f"USGS_{gage_id}_1h_UTC.csv"
    if not csv.exists():
        print(f"  [skip] no obs file: {csv}")
        return None
    df = pd.read_csv(csv, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.loc[(df.index >= MRMS_BEGIN) & (df.index <= MRMS_END)]
    if len(df) < WINDOW_DAYS * 24:
        print(f"  [skip] insufficient obs in MRMS range")
        return None

    best = None
    start = MRMS_BEGIN
    end_limit = MRMS_END - pd.Timedelta(days=WINDOW_DAYS)
    while start <= end_limit:
        window = df.loc[start : start + pd.Timedelta(days=WINDOW_DAYS) - pd.Timedelta(hours=1)]
        if len(window) >= WINDOW_DAYS * 24 * 0.8:  # at least 80% coverage
            scored = score_window(window)
            if scored is not None:
                scored["start"] = start
                scored["end"] = window.index[-1]
                if best is None or scored["score"] > best["score"]:
                    best = scored
        start += pd.Timedelta(days=STRIDE_DAYS)
    return best


def main() -> None:
    rows = []
    for g in CANDIDATES:
        print(f"=== {g} ===")
        b = best_window_for_gage(g)
        if b is None:
            continue
        print(f"  best window: {b['start']} → {b['end']}")
        print(f"    peak={b['peak']:.1f} m3/s, median={b['median']:.2f}, ratio={b['peak_ratio']:.1f}")
        print(f"    rise={b['rise_h']}h, recession={b['rec_h']}h, score={b['score']:.2f}")
        rows.append({"gage_id": g, **b})

    rows.sort(key=lambda r: r["score"], reverse=True)
    print("\n=== Ranked by composite score ===")
    for r in rows:
        s = r["start"].strftime("%Y%m%d%H%M")
        e = r["end"].strftime("%Y%m%d%H%M")
        print(f"  {r['gage_id']}  score={r['score']:6.2f}  ratio={r['peak_ratio']:5.1f}  "
              f"rise={r['rise_h']:3d}h  rec={r['rec_h']:4d}h  {s}–{e}")

    print("\n=== Top 10 (suggested training set) ===")
    for r in rows[:10]:
        s = r["start"].strftime("%Y%m%d%H%M")
        e = r["end"].strftime("%Y%m%d%H%M")
        print(f"  {r['gage_id']}: time_begin={s!r}  time_end={e!r}")


if __name__ == "__main__":
    main()
