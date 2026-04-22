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
        """Execute EF5 with current parameters and return a hydrograph summary.

        Returns a dict with NSE, peak flows, volume ratio, timing error,
        or an error description if the simulation fails.
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
                self.nse_history.append(-1.0)
                return {
                    "status": "error",
                    "message": f"EF5 exited with code {result.returncode}",
                    "stderr": result.stderr[:500],
                    "nse": -1.0,
                }

            # Log EF5 output for diagnostics
            if result.stdout:
                logger.info("EF5 stdout (last 500 chars): %s", result.stdout[-500:])
            if result.stderr:
                logger.info("EF5 stderr (last 500 chars): %s", result.stderr[-500:])
        except subprocess.TimeoutExpired:
            logger.warning("EF5 timed out after %d seconds", self.gage.ef5_timeout)
            self.nse_history.append(-1.0)
            return {
                "status": "error",
                "message": f"EF5 timed out after {self.gage.ef5_timeout}s",
                "nse": -1.0,
            }

        # Parse output
        sim_data = self._parse_output()
        if sim_data is None:
            self.nse_history.append(-1.0)
            return {
                "status": "error",
                "message": "Could not parse simulation output",
                "nse": -1.0,
            }

        obs = sim_data["obs"]
        sim = sim_data["sim"]

        # logger.info(
        #     "Parsed %d timesteps (obs range: [%.2f, %.2f], sim range: [%.2f, %.2f])",
        #     len(obs), float(np.min(obs)), float(np.max(obs)),
        #     float(np.min(sim)), float(np.max(sim)),
        # )

        nse = self._compute_nse(obs, sim)
        self.nse_history.append(nse)

        return {
            "status": "ok",
            "nse": round(nse, 4),
            "peak_sim_m3s": round(float(np.max(sim)), 2),
            "peak_obs_m3s": round(float(np.max(obs)), 2),
            "volume_ratio": round(float(np.sum(sim) / max(np.sum(obs), 1e-6)), 3),
            "timing_error_hours": self._peak_timing_error(obs, sim),
            "run_number": self.run_count,
        }

    def evaluate(self) -> dict:
        """Return detailed evaluation metrics from all simulation runs."""
        best_nse = max(self.nse_history) if self.nse_history else None
        return {
            "current_nse": self.nse_history[-1] if self.nse_history else None,
            "best_nse": round(best_nse, 4) if best_nse is not None else None,
            "nse_history": [round(n, 4) for n in self.nse_history],
            "num_runs": self.run_count,
            "target_nse": self.gage.target_nse,
            "target_met": (best_nse is not None and best_nse > self.gage.target_nse),
            "current_params": {k: round(v, 6) for k, v in self.current_params.items()},
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
        return result.get("nse", -1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_obs_file(self) -> str:
        """Discover the observation file in the gage data directory."""
        obs_dir = Path(self.gage.obs_dir)
        if obs_dir.exists():
            csv_files = list(obs_dir.glob("*.csv"))
            if csv_files:
                return csv_files[0].name
        # Fallback: check for common name patterns
        return f"{self.gage.gage_id}_obs.csv"

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

    # Missing-value sentinel used by EF5 for no-data observations
    _NODATA_THRESHOLD = -998.0

    def _parse_output(self) -> Optional[dict[str, np.ndarray]]:
        """Parse the EF5 output CSV to extract observed and simulated timeseries.

        Filters out rows where either the observation or simulation value
        is NaN or a missing-data sentinel (≤ -999).
        """
        # Look for output file: ts.{gage_id}.crest.csv
        output_pattern = f"ts.{self.gage.gage_id}.crest.csv"
        output_file = self.output_dir / output_pattern

        if not output_file.exists():
            # Try alternative patterns
            candidates = list(self.output_dir.glob(f"ts.{self.gage.gage_id}*.csv"))
            if not candidates:
                logger.warning("No output file found matching %s", output_pattern)
                return None
            output_file = candidates[0]

        try:
            obs_values = []
            sim_values = []
            skipped = 0
            skip_reasons = {"obs_nan": 0, "sim_nan": 0, "obs_nodata": 0, "sim_nodata": 0}
            first_rows_logged = False
            with open(output_file) as f:
                reader = csv.DictReader(f)
                # Log column headers
                logger.info("CSV columns: %s", reader.fieldnames)
                for row_idx, row in enumerate(reader):
                    # Column names from EF5 output
                    obs_col = None
                    sim_col = None
                    for col in row:
                        if "observed" in col.lower():
                            obs_col = col
                        if "discharge" in col.lower() and "observed" not in col.lower():
                            sim_col = col

                    # Log first 3 rows for diagnosis
                    if row_idx < 3:
                        logger.info(
                            "Row %d: obs_col=%s val=%s, sim_col=%s val=%s",
                            row_idx,
                            obs_col, row.get(obs_col, "N/A") if obs_col else "NO_COL",
                            sim_col, row.get(sim_col, "N/A") if sim_col else "NO_COL",
                        )

                    if obs_col and sim_col:
                        try:
                            obs_val = float(row[obs_col])
                            sim_val = float(row[sim_col])
                        except (ValueError, TypeError):
                            skipped += 1
                            continue

                        # Filter out missing values: EF5 uses -999 as nodata,
                        # and NaN can also appear in model output
                        if np.isnan(obs_val):
                            skip_reasons["obs_nan"] += 1
                            skipped += 1
                            continue
                        if np.isnan(sim_val):
                            skip_reasons["sim_nan"] += 1
                            skipped += 1
                            continue
                        if obs_val <= self._NODATA_THRESHOLD:
                            skip_reasons["obs_nodata"] += 1
                            skipped += 1
                            continue
                        if sim_val <= self._NODATA_THRESHOLD:
                            skip_reasons["sim_nodata"] += 1
                            skipped += 1
                            continue

                        obs_values.append(obs_val)
                        sim_values.append(sim_val)

            if skipped > 0:
                logger.info(
                    "Skipped %d rows in %s — reasons: %s",
                    skipped, output_file.name, skip_reasons,
                )

            if not obs_values:
                logger.warning("No valid data rows in %s", output_file)
                return None

            return {
                "obs": np.array(obs_values),
                "sim": np.array(sim_values),
            }
        except Exception as e:
            logger.error("Error parsing %s: %s", output_file, e)
            return None

    @staticmethod
    def _compute_nse(obs: np.ndarray, sim: np.ndarray) -> float:
        """Compute Nash-Sutcliffe Efficiency.

        NSE = 1 - sum((obs - sim)²) / sum((obs - mean(obs))²)

        Handles residual NaN values by masking them out.
        """
        # Mask out any remaining NaN values
        valid = ~(np.isnan(obs) | np.isnan(sim))
        obs = obs[valid]
        sim = sim[valid]

        if len(obs) == 0:
            logger.warning("No valid obs/sim pairs for NSE computation")
            return -1.0

        obs_mean = np.mean(obs)
        denominator = np.sum((obs - obs_mean) ** 2)
        if denominator == 0:
            return 0.0
        numerator = np.sum((obs - sim) ** 2)
        return float(1.0 - numerator / denominator)

    @staticmethod
    def _peak_timing_error(obs: np.ndarray, sim: np.ndarray) -> int:
        """Compute peak timing error in hours (integer)."""
        # Mask NaN for argmax
        obs_clean = np.where(np.isnan(obs), -np.inf, obs)
        sim_clean = np.where(np.isnan(sim), -np.inf, sim)
        obs_peak_idx = int(np.argmax(obs_clean))
        sim_peak_idx = int(np.argmax(sim_clean))
        return abs(obs_peak_idx - sim_peak_idx)
