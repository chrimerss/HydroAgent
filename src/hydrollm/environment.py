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
            csv_files = list(obs_dir.glob("*.csv"))
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
        """Calculate NSE from the most recent simulation output CSV.

        Parses the EF5 output CSV to extract observed/simulated discharge,
        then computes Nash-Sutcliffe Efficiency. Updates nse_history.

        Returns:
            dict with NSE value and status
        """
        csv_path = self._find_output_csv()
        if csv_path is None:
            logger.error("No output CSV found for NSE calculation")
            return {"status": "error", "message": "No output CSV found", "nse": -1.0}

        try:
            obs, sim = self._parse_output_csv(csv_path)
            if obs is None or sim is None:
                self.nse_history.append(-999.0)
                return {"status": "error", "message": "Failed to parse CSV or no valid data", "nse": -999.0}

            nse = self._compute_nse(obs, sim)
            self.nse_history.append(nse)
            return {
                "status": "ok",
                "nse": round(nse, 4),
                "message": f"NSE calculated from {len(obs)} data points",
                "num_points": len(obs),
            }
        except Exception as e:
            logger.error("Unexpected error in evaluate: %s", e)
            self.nse_history.append(-999.0)
            return {"status": "error", "message": f"Unexpected error: {e}", "nse": -999.0}

    @staticmethod
    def _compute_nse(obs: np.ndarray, sim: np.ndarray) -> float:
        """Compute Nash-Sutcliffe Efficiency between observed and simulated."""
        valid = ~(np.isnan(obs) | np.isnan(sim))
        obs_c, sim_c = obs[valid], sim[valid]
        if len(obs_c) == 0:
            return -1.0
        denom = np.sum((obs_c - np.mean(obs_c)) ** 2)
        if denom == 0:
            return 0.0
        return float(1.0 - np.sum((obs_c - sim_c) ** 2) / denom)

    @staticmethod
    def _parse_output_csv(csv_path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse EF5 output CSV and extract observed/simulated discharge arrays."""
        obs_values: list[float] = []
        sim_values: list[float] = []
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                logger.info("CSV columns: %s", fieldnames)

                obs_col = next(
                    (c for c in fieldnames if "observed" in c.lower() or "obs" in c.lower()),
                    None,
                )
                sim_col = next(
                    (c for c in fieldnames
                     if ("discharge" in c.lower() or "sim" in c.lower())
                     and "observed" not in c.lower()),
                    None,
                )

                if obs_col is None or sim_col is None:
                    logger.error("Cannot identify obs/sim columns. Available: %s", fieldnames)
                    return None, None

                logger.info("Using columns: obs='%s', sim='%s'", obs_col, sim_col)
                total_rows = 0
                skipped_nodata = 0
                for row in reader:
                    total_rows += 1
                    try:
                        ov = float(row[obs_col]) if row[obs_col] else float("nan")
                        sv = float(row[sim_col]) if row[sim_col] else float("nan")
                        if np.isnan(ov) or np.isnan(sv) or ov <= -999 or sv <= -999:
                            skipped_nodata += 1
                            if total_rows <= 3:
                                logger.info("Row %d skipped: obs=%s sim=%s", total_rows, ov, sv)
                            continue
                        obs_values.append(ov)
                        sim_values.append(sv)
                    except (ValueError, KeyError):
                        skipped_nodata += 1
                        continue

            if not obs_values:
                logger.warning(
                    "No valid data in %s (%d rows, %d skipped as nodata/invalid)",
                    csv_path, total_rows, skipped_nodata,
                )
                return None, None
            logger.info("Parsed %d valid rows from %d total", len(obs_values), total_rows)
            return np.array(obs_values), np.array(sim_values)
        except Exception as e:
            logger.error("Error reading CSV %s: %s", csv_path, e)
            return None, None

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
