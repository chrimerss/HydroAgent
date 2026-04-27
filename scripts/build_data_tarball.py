"""Assemble the data.tar.gz uploaded to chrimerss/hydro_cali_agent_example.

Layout produced:
    data/
      basic_data/              (CONUS-wide; copied from existing tarball)
      pet/                     (CONUS-wide; copied from existing tarball)
      docs/control.txt         (placeholder template)
      gauge/                   (USGS_<gid>_1h_UTC.csv for 11 gages)
      data_mrms_clip/<gid>/    (only the 2-month window per gage)

Per-gage MRMS subdirs map to `MRMS_LOC=/app/data/data_mrms_clip/<gage_id>/`,
substituted into the control template at runtime (see environment.py).
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

REPO = Path("/Users/allen/Documents/Python/hydroGPT")
SRC_GAUGE = REPO / "data" / "gauge"
SRC_MRMS = REPO / "data" / "MRMS_forcing"
EXTRACTED = Path("/tmp/hydro_data_build/extracted")  # has pet/, basic_data/
BUILD = Path("/tmp/hydro_data_build/data")
TARBALL = Path("/tmp/hydro_data_build/data.tar.gz")
CONTROL_SRC = REPO / "control.txt"

TRAIN_GAGES = [
    "02294781", "02312000", "07195430", "11043000", "11152000",
    "11179000", "11376000", "11383500", "14207500", "14301000",
]
TEST_GAGES = ["02338660"]
ALL_GAGES = TRAIN_GAGES + TEST_GAGES


def parse_yyyymmddhhmm(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M")


def gage_window(gage_id: str) -> tuple[datetime, datetime]:
    cfg_path = REPO / "configs" / "gages" / f"{gage_id}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return parse_yyyymmddhhmm(str(cfg["time_begin"])), parse_yyyymmddhhmm(str(cfg["time_end"]))


def copy_mrms_window(gage_id: str, t0: datetime, t1: datetime) -> int:
    """Copy MRMS hourly tifs for [t0, t1] inclusive into BUILD/data_mrms_clip/<gid>/."""
    src = SRC_MRMS / gage_id
    dst = BUILD / "data_mrms_clip" / gage_id
    dst.mkdir(parents=True, exist_ok=True)

    n = 0
    cur = t0
    while cur <= t1:
        fname = f"GaugeCorr_QPE_01H_00.00_{cur.strftime('%Y%m%d-%H')}0000.grib2.tif"
        sp = src / fname
        if sp.exists():
            shutil.copy2(sp, dst / fname)
            n += 1
        else:
            print(f"  [missing] {gage_id} {fname}", file=sys.stderr)
        cur += timedelta(hours=1)
    return n


def copy_gauge(gage_id: str) -> bool:
    sp = SRC_GAUGE / f"USGS_{gage_id}_1h_UTC.csv"
    if not sp.exists():
        print(f"  [missing obs] {gage_id}", file=sys.stderr)
        return False
    dst = BUILD / "gauge"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sp, dst / sp.name)
    return True


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    # 1) Copy CONUS-wide assets from existing tarball.
    print("Copying basic_data/ and pet/ from existing tarball ...")
    shutil.copytree(EXTRACTED / "basic_data", BUILD / "basic_data")
    shutil.copytree(EXTRACTED / "pet", BUILD / "pet")

    # 2) Control template
    docs = BUILD / "docs"
    docs.mkdir()
    shutil.copy2(CONTROL_SRC, docs / "control.txt")

    # 3) Per-gage assets.
    for gid in ALL_GAGES:
        t0, t1 = gage_window(gid)
        print(f"=== {gid}  window {t0:%Y-%m-%d %H:%M}{t1:%Y-%m-%d %H:%M} ===")
        ok = copy_gauge(gid)
        if not ok:
            print(f"  [skip] missing gauge obs")
            continue
        n = copy_mrms_window(gid, t0, t1)
        print(f"  copied {n} MRMS files")

    # 4) Sanity sizes
    print("\n=== Build sizes ===")
    import subprocess
    subprocess.run(["du", "-sh", "--", str(BUILD)], check=False)
    subprocess.run(["du", "-sh", "--", *(str(p) for p in sorted(BUILD.iterdir()))], check=False)
    subprocess.run(["du", "-sh", "--", *(str(p) for p in sorted((BUILD / 'data_mrms_clip').iterdir()))], check=False)

    # 5) Tarball
    if TARBALL.exists():
        TARBALL.unlink()
    print(f"\nCreating {TARBALL} ...")
    # Note: BUILD's *contents* are placed at the tarball root (no `data/`
    # prefix) so Modal's `tar -xzf data.tar.gz -C /app/data` extracts to
    # /app/data/pet, /app/data/basic_data, etc. COPYFILE_DISABLE=1 stops
    # macOS from bundling AppleDouble metadata files (`._*`); without it
    # EF5 tries to parse those as TIFs and hangs.
    env = {**os.environ, "COPYFILE_DISABLE": "1"}
    subprocess.run(
        [
            "tar",
            "--exclude=._*",
            "--exclude=.DS_Store",
            "-czf", str(TARBALL),
            "-C", str(BUILD),
            ".",
        ],
        check=True,
        env=env,
    )
    subprocess.run(["du", "-sh", "--", str(TARBALL)], check=False)


if __name__ == "__main__":
    main()
