"""Apply audit results: write the chosen 2-month window into each gage YAML."""
from __future__ import annotations
from pathlib import Path
import yaml

# (gage_id, time_begin, time_end) — top 10 from audit_flood_windows.py
WINDOWS = {
    "02294781": ("201804290000", "201806272300"),
    "02312000": ("201811150000", "201901132300"),
    "14207500": ("201804090000", "201806072300"),
    "11152000": ("201805290000", "201807272300"),
    "14301000": ("201809110000", "201811092300"),
    "11376000": ("201809210000", "201811192300"),
    "11043000": ("201903150000", "201905132300"),
    "11179000": ("201806030000", "201808012300"),
    # Replacements for 06279500 / 07144100 (too slow under K=8 rollout):
    "07195430": ("201801040000", "201803042300"),
    "11383500": ("201805190000", "201807172300"),
}

CFG_DIR = Path("/Users/allen/Documents/Python/hydroGPT/configs/gages")


def main() -> None:
    for gid, (tb, te) in WINDOWS.items():
        path = CFG_DIR / f"{gid}.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg["time_begin"] = tb
        cfg["time_end"] = te
        # Standardize obs_dir
        cfg["obs_dir"] = "/app/data/gauge"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        print(f"updated {path}: {tb}{te}")


if __name__ == "__main__":
    main()
