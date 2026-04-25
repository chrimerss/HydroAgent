"""Thread-safe EF5/CREST hydrologic simulation environment.

Each rollout gets an isolated sandbox directory to prevent concurrent
simulations from interfering with each other. The environment wraps
the EF5 binary, handles parameter injection into INI-style control files,
parses simulation output, and computes NSE scores.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from hydrollm.config import (
    DEFAULT_PARAMETERS,
    GageConfig,
    PARAMETER_RANGES,
)

logger = logging.getLogger(__name__)

EF5_EXECUTABLE = os.environ.get("EF5_EXECUTABLE", "/EF5/bin/ef5")


class HydroEnvironment:
    """Isolated EF5/CREST simulation sandbox for a single rollout.

    Each instance operates in its own temporary directory so that
    multiple rollouts can run concurrently without file conflicts.
    """

    def __init__(self, gage_config: GageConfig):
        self.gage = gage_config
        self.sandbox_id = uuid.uuid4().hex[:8]
        self.sandbox_dir = Path(f"/tmp/hydrollm_rollout_{self.sandbox_id}")
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

        self.control_path = self.sandbox_dir / "control.txt"
        self.output_dir = self.sandbox_dir / "results"
        self.output_dir.mkdir(exist_ok=True)

        # State tracking
        self.current_params: dict[str, float] = dict(DEFAULT_PARAMETERS)
        self.nse_history: list[float] = []
        self.run_count: int = 0

        # Discover the observation file
        self._obs_filename = self._find_obs_file()

        # Write initial control file
        self._write_control_file()

    # ------------------------------------------------------------------
    # Public API (called by tool executor)
    # ------------------------------------------------------------------

    def set_parameters(self, params: dict[str, float]) -> dict:
        """Validate, clamp, and write parameter values to the control file.

        Returns a dict with status and the validated parameter values.
        """
        validated = self._validate_and_clamp(params)
        self.current_params.update(validated)
        self._write_control_file()
        return {
            "status": "ok",
            "validated_params": {k: round(v, 6) for k, v in self.current_params.items()},
        }

    def run_simulation(self) -> dict:
        """Execute EF5 with current parameters.

        Runs EF5 binary and verifies output CSV was created.
        Does NOT calculate NSE - call evaluate() to compute metrics.

        Returns:
            dict with execution status and run_number
        """
        self.run_count += 1
        try:
            result = subprocess.run(
                [EF5_EXECUTABLE, str(self.control_path)],
                capture_output=True,
                text=True,
                timeout=self.gage.ef5_timeout,
                cwd=str(self.sandbox_dir),
            )
            if result.returncode != 0:
                logger.warning(
                    "EF5 exited with code %d: %s", result.returncode, result.stderr[:500]
                )
                return {
                    "status": "error",
                    "message": f"EF5 exited with code {result.returncode}",
                    "stderr": result.stderr[:500],
                }

            # Log EF5 output for diagnostics
            if result.stdout:
                logger.info("EF5 stdout (last 500 chars): %s", result.stdout[-500:])
            if result.stderr:
                logger.info("EF5 stderr (last 500 chars): %s", result.stderr[-500:])
        except subprocess.TimeoutExpired:
            logger.warning("EF5 timed out after %d seconds", self.gage.ef5_timeout)
            return {
                "status": "error",
                "message": f"EF5 timed out after {self.gage.ef5_timeout}s",
            }

        # Verify output CSV was created
        output_csv = self._find_output_csv()
        if output_csv is None:
            return {
                "status": "error",
                "message": "Simulation completed but no output CSV found",
            }

        return {
            "status": "ok",
            "message": f"Simulation completed successfully (run #{self.run_count})",
            "output_file": str(output_csv.name),
            "run_number": self.run_count,
        }
    def cleanup(self):
        """Remove the sandbox directory."""
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Convenience: run set_parameters + run_simulation in one call
    # ------------------------------------------------------------------

    def run_with_params(self, params: dict[str, float]) -> float:
        """Set parameters and run simulation, returning the NSE score."""
        self.set_parameters(params)
        result = self.run_simulation()
        return result.get("nse", -999)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_obs_file(self) -> str:
        """Discover the observation file in the gage data directory."""
        obs_dir = Path(self.gage.obs_dir)
        if obs_dir.exists():
            csv_files = list(obs_dir.glob("USGS*.csv"))
            if csv_files:
                logger.info("Found obs file: %s", csv_files[0])
                return csv_files[0].name
            logger.warning("obs_dir %s exists but contains no CSV files", obs_dir)
        else:
            logger.warning("obs_dir %s does not exist", obs_dir)
        fallback = f"{self.gage.gage_id}_obs.csv"
        logger.warning("Using fallback obs filename: %s", fallback)
        return fallback

    def _find_output_csv(self) -> Optional[Path]:
        """Find the most recent output CSV file from EF5 simulation.

        Returns:
            Path to the output CSV file, or None if not found
        """
        # Look for standard pattern: ts.{gage_id}.crest.csv
        standard_pattern = f"ts.{self.gage.gage_id}.crest.csv"
        output_file = self.output_dir / standard_pattern

        if output_file.exists():
            return output_file

        # Try alternative patterns
        candidates = list(self.output_dir.glob(f"ts.{self.gage.gage_id}*.csv"))
        if candidates:
            # Return the most recently modified file
            return max(candidates, key=lambda p: p.stat().st_mtime)

        logger.warning("No output file found matching patterns for gage %s", self.gage.gage_id)
        return None

    def evaluate(self) -> dict:
        """Compute a full metric suite from the most recent simulation.

        Returns NSE, correlation coefficient (CC), Kling-Gupta Efficiency
        (KGE), peak magnitudes and ratio, and the peak-timing lag (hours,
        signed as sim - obs). The `run_simulation` tool intentionally does
        NOT return any of these metrics — only this tool does, so the
        agent must explicitly call evaluate to learn how well it is doing.
        Updates nse_history with the computed NSE.
        """
        csv_path = self._find_output_csv()
        if csv_path is None:
            logger.error("No output CSV found for metric calculation")
            return {
                "status": "error",
                "message": "No output CSV found",
                "nse": -1.0,
            }

        try:
            metrics = compute_metrics(csv_path)
        except Exception as e:
            logger.error("Unexpected error computing metrics: %s", e)
            self.nse_history.append(-999.0)
            return {
                "status": "error",
                "message": f"Unexpected error computing metrics: {e}",
                "nse": -999.0,
            }

        if metrics is None:
            self.nse_history.append(-999.0)
            return {
                "status": "error",
                "message": "Failed to parse CSV or no valid data",
                "nse": -999.0,
            }

        nse_value = metrics["NSE"]
        # Keep the history numeric so max() calls downstream stay valid.
        self.nse_history.append(
            -999.0 if (nse_value is None or np.isnan(nse_value)) else float(nse_value)
        )

        # Round floats for a compact tool response; leave num_points as int.
        rounded = {
            k: (int(v) if k == "num_points" else _round_metric(v))
            for k, v in metrics.items()
        }
        return {
            "status": "ok",
            "message": f"Metrics calculated from {metrics['num_points']} data points",
            **rounded,
        }

    def _validate_and_clamp(self, params: dict[str, float]) -> dict[str, float]:
        """Validate parameter names and clamp values to valid ranges."""
        validated = {}
        for key, value in params.items():
            if key not in PARAMETER_RANGES:
                logger.warning("Unknown parameter '%s', skipping", key)
                continue
            lo, hi = PARAMETER_RANGES[key]
            clamped = max(lo, min(hi, float(value)))
            if clamped != value:
                logger.debug("Clamped %s: %.4f -> %.4f", key, value, clamped)
            validated[key] = clamped
        return validated

    def _write_control_file(self):
        """Write the control file with current parameters."""
        # Read the template
        template_path = Path(self.gage.control_template)
        if not template_path.exists():
            # Fallback: use embedded template
            content = self._generate_control_content()
        else:
            content = template_path.read_text()
            content = self._inject_parameters(content)

        # Fix paths for sandbox
        content = self._fix_output_path(content)
        content = self._fix_obs_filename(content)

        # Log the OBS line for diagnosis
        for line in content.splitlines():
            if line.strip().upper().startswith("OBS="):
                logger.info("Control file OBS path: %s", line.strip())
                break

        self.control_path.write_text(content)

    def _inject_parameters(self, content: str) -> str:
        """Inject current parameter values into control file content."""
        # CREST parameters (in [CrestParamSet ...] section)
        crest_params = ["wm", "b", "im", "ke", "fc"]
        for param in crest_params:
            pattern = rf"^({param}\s*=\s*)[\d.eE+-]+(\s*)$"
            replacement = rf"\g<1>{self.current_params[param]}\2"
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.IGNORECASE)

        # Initial state (also in CREST section)
        pattern = rf"^(iwu\s*=\s*)[\d.eE+-]+(\s*)$"
        content = re.sub(
            pattern,
            rf"\g<1>{self.current_params['iwu']}\2",
            content,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # KW routing parameters (in [kwparamset ...] section)
        kw_params = ["under", "leaki", "alpha", "beta", "alpha0", "th", "isu"]
        for param in kw_params:
            pattern = rf"^({param}\s*=\s*)[\d.eE+-]+(\s*)$"
            replacement = rf"\g<1>{self.current_params[param]}\2"
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.IGNORECASE)

        return content

    def _fix_output_path(self, content: str) -> str:
        """Replace the output directory with sandbox output dir."""
        return re.sub(
            r"^(OUTPUT\s*=\s*).*$",
            rf"\g<1>{self.output_dir}/",
            content,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    def _fix_obs_filename(self, content: str) -> str:
        """Replace the GAUGE_OBS_FILENAME placeholder with actual filename."""
        return content.replace("GAUGE_OBS_FILENAME", self._obs_filename)

    def _generate_control_content(self) -> str:
        """Generate a control file from scratch (fallback if template missing)."""
        p = self.current_params
        g = self.gage
        return f"""[Basic]
DEM=/app/data/basic_data/basic/dem_usa.tif
DDM=/app/data/basic_data/basic/fdir_usa.tif
FAM=/app/data/basic_data/basic/facc_usa.tif
PROJ=geographic
ESRIDDM=true
SelfFAM=false

[PrecipForcing MRMS]
TYPE=TIF
UNIT=mm/h
FREQ=1h
LOC=/app/data/data_mrms_clip/
NAME=GaugeCorr_QPE_01H_00.00_YYYYMMDD-HH0000.grib2.tif

[PETForcing PET]
TYPE=TIF
UNIT=mm/100d
FREQ=d
LOC=/app/data/pet/
NAME=etYYYYMMDD.tif

[Gauge {g.gage_id}]
LON={g.lon}
LAT={g.lat}
OBS=/app/data/gauge_18_19/{self._obs_filename}
OUTPUTTS=TRUE
WANTCO=TRUE
BASINAREA={g.basin_area}

[Basin 0]
GAUGE={g.gage_id}

[CrestParamSet CrestParam]
gauge={g.gage_id}
WM_GRID=/app/data/basic_data/default_param/crest_params/wm_usa.tif
IM_GRID=/app/data/basic_data/default_param/crest_params/im_usa.tif
FC_GRID=/app/data/basic_data/default_param/crest_params/ksat_usa.tif
B_GRID=/app/data/basic_data/default_param/crest_params/b_usa.tif
wm={p['wm']}
b={p['b']}
im={p['im']}
ke={p['ke']}
fc={p['fc']}
iwu={p['iwu']}

[kwparamset KWParam]
gauge={g.gage_id}
leaki_grid=/app/data/basic_data/default_param/kw_params/leaki_usa.tif
alpha_grid=/app/data/basic_data/default_param/kw_params/alpha_usa.tif
beta_grid=/app/data/basic_data/default_param/kw_params/beta_usa.tif
alpha0_grid=/app/data/basic_data/default_param/kw_params/alpha0_usa.tif
under={p['under']}
leaki={p['leaki']}
th={p['th']}
isu={p['isu']}
alpha={p['alpha']}
beta={p['beta']}
alpha0={p['alpha0']}

[Task Simu]
STYLE=SIMU
MODEL=CREST
ROUTING=KW
BASIN=0
PRECIP=MRMS
PET=PET
OUTPUT={self.output_dir}/
PARAM_SET=CrestParam
ROUTING_PARAM_Set=KWParam
TIMESTEP=1h
TIME_BEGIN={g.time_begin}
TIME_END={g.time_end}

[Execute]
TASK=Simu
"""

    @staticmethod
    def _compute_nse(obs: np.ndarray, sim: np.ndarray) -> float:
        """Compute Nash-Sutcliffe Efficiency (kept for unit tests)."""
        valid = ~(np.isnan(obs) | np.isnan(sim))
        obs_c, sim_c = obs[valid], sim[valid]
        if len(obs_c) == 0:
            return -1.0
        denom = np.sum((obs_c - np.mean(obs_c)) ** 2)
        if denom == 0:
            return 0.0
        return float(1.0 - np.sum((obs_c - sim_c) ** 2) / denom)

    @staticmethod
    def _peak_timing_error(obs: np.ndarray, sim: np.ndarray) -> int:
        """Compute |obs_peak_idx - sim_peak_idx| (kept for unit tests)."""
        obs_clean = np.where(np.isnan(obs), -np.inf, obs)
        sim_clean = np.where(np.isnan(sim), -np.inf, sim)
        return abs(int(np.argmax(obs_clean)) - int(np.argmax(sim_clean)))


# ---------------------------------------------------------------------------
# Module-level metric helpers
# ---------------------------------------------------------------------------

# EF5 uses -999 as the observation missing-value sentinel.
_NODATA_THRESHOLD = -998.0


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns NaN rather than raising on zero variance."""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency, KGE = 1 - sqrt((r-1)^2 + (a-1)^2 + (b-1)^2)."""
    if len(obs) < 2:
        return float("nan")
    obs_mean = float(np.mean(obs))
    obs_std = float(np.std(obs))
    sim_std = float(np.std(sim))
    if obs_std == 0 or obs_mean == 0:
        return float("nan")
    r = _safe_corrcoef(sim, obs)
    if np.isnan(r):
        return float("nan")
    alpha = sim_std / obs_std
    beta = float(np.mean(sim)) / obs_mean
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _round_metric(v: float) -> float:
    """Round a metric to 4 dp, passing NaN/Inf through unchanged."""
    if v is None:
        return v
    try:
        if np.isnan(v) or np.isinf(v):
            return float(v)
    except TypeError:
        return v
    return round(float(v), 4)


def _identify_columns(columns: list[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Pick (time, sim, obs) columns from an EF5 output CSV header."""
    time_col = next(
        (c for c in columns if c.lower() in {"time", "date", "datetime", "timestamp"}),
        None,
    )
    if time_col is None:
        time_col = next((c for c in columns if "time" in c.lower() or "date" in c.lower()), None)

    obs_col = next((c for c in columns if "observed" in c.lower()), None)
    if obs_col is None:
        obs_col = next((c for c in columns if "obs" in c.lower()), None)

    sim_col = next(
        (c for c in columns
         if ("discharge" in c.lower() or "sim" in c.lower())
         and "observed" not in c.lower() and "obs" not in c.lower()),
        None,
    )
    return time_col, sim_col, obs_col


def compute_metrics(csv_path: Path) -> Optional[dict]:
    """Compute NSE, CC, KGE, peak magnitudes/ratio, and peak-timing lag.

    Reads the EF5 output CSV, identifies observation/simulation columns,
    filters NaN and the -999 missing-value sentinel, and returns a dict
    with every metric plus the number of valid points. Returns None if
    the CSV cannot be parsed or contains no valid rows.
    """
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error("Failed to read CSV %s: %s", csv_path, e)
        return None

    columns = list(df.columns)
    time_col, sim_col, obs_col = _identify_columns(columns)
    if sim_col is None or obs_col is None:
        logger.error("Cannot identify obs/sim columns. Available: %s", columns)
        return None
    logger.info("Using columns: time='%s', sim='%s', obs='%s'", time_col, sim_col, obs_col)

    s = pd.to_numeric(df[sim_col], errors="coerce")
    o = pd.to_numeric(df[obs_col], errors="coerce")
    valid = ~(s.isna() | o.isna()) & (s > _NODATA_THRESHOLD) & (o > _NODATA_THRESHOLD)

    s = s[valid].to_numpy(dtype=float)
    o = o[valid].to_numpy(dtype=float)
    if s.size == 0:
        logger.warning("No valid obs/sim rows in %s", csv_path)
        return None

    # Time series for peak-timing lag; falls back to integer index if the
    # time column is missing or not parseable.
    if time_col is not None:
        t = pd.to_datetime(df[time_col], errors="coerce")[valid]
        if t.isna().any():
            t = None
    else:
        t = None

    obs_mean = float(np.mean(o))
    denom = float(np.sum((o - obs_mean) ** 2))
    nse = float(1.0 - np.sum((o - s) ** 2) / denom) if denom > 0 else float("nan")

    cc = _safe_corrcoef(s, o)
    kge_val = _kge(s, o)

    s_peak = float(np.max(s))
    o_peak = float(np.max(o))
    peak_ratio = float(s_peak / o_peak) if o_peak > 0 else float("inf")

    s_idx = int(np.argmax(s))
    o_idx = int(np.argmax(o))
    if t is not None:
        lag_hours = float((t.iloc[s_idx] - t.iloc[o_idx]).total_seconds() / 3600.0)
    else:
        # Assume 1h timestep (EF5 default) when no timestamp column is present.
        lag_hours = float(s_idx - o_idx)

    return {
        "NSE": nse,
        "CC": cc,
        "KGE": kge_val,
        "sim_peak": s_peak,
        "obs_peak": o_peak,
        "peak_ratio": peak_ratio,
        "lag_hours_sim_minus_obs": lag_hours,
        "num_points": int(s.size),
    }
