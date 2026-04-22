"""Unit tests for the HydroEnvironment wrapper.

These tests validate parameter handling, control file generation,
and NSE computation without requiring EF5 to be installed.
"""

import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from hydrollm.config import GageConfig, DEFAULT_PARAMETERS, PARAMETER_RANGES
from hydrollm.environment import HydroEnvironment


@pytest.fixture
def gage_config():
    """Create a test gage configuration."""
    return GageConfig(
        gage_id="02338660",
        lon=-84.9876,
        lat=33.2357,
        basin_area=328.93,
        obs_dir="/tmp/test_obs",
        control_template="/tmp/test_control.txt",
        time_begin="201807010000",
        time_end="201808312300",
    )


@pytest.fixture
def mock_env(gage_config, tmp_path):
    """Create a HydroEnvironment with mocked file paths."""
    # Create a dummy control template
    template = tmp_path / "control.txt"
    template.write_text("""[Basic]
DEM=/app/data/basic_data/basic/dem_usa.tif
DDM=/app/data/basic_data/basic/fdir_usa.tif
FAM=/app/data/basic_data/basic/facc_usa.tif
PROJ=geographic
ESRIDDM=true
SelfFAM=false

[CrestParamSet CrestParam]
gauge=02338660
wm=1.0
b=1.0
im=1.0
ke=1.0
fc=1.0
iwu=25.0

[kwparamset KWParam]
gauge=02338660
under=1.0
leaki=1.0
th=10.0
isu=0.0
alpha=1.0
beta=1.0
alpha0=0.0

[Task Simu]
OUTPUT=/app/results/
TIME_BEGIN=201807010000
TIME_END=201808312300

[Execute]
TASK=Simu
""")

    # Create dummy obs directory
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    (obs_dir / "02338660_obs.csv").write_text("time,discharge\n")

    gage_config.control_template = str(template)
    gage_config.obs_dir = str(obs_dir)

    env = HydroEnvironment(gage_config)
    yield env
    env.cleanup()


class TestParameterValidation:
    """Test parameter validation and clamping."""

    def test_valid_parameters(self, mock_env):
        result = mock_env.set_parameters({
            "wm": 2.5, "b": 0.8, "im": 0.05, "ke": 1.0,
            "fc": 1.0, "under": 1.0, "leaki": 1.0,
            "alpha": 1.0, "beta": 1.0, "alpha0": 0.0, "iwu": 25.0,
        })
        assert result["status"] == "ok"
        assert result["validated_params"]["wm"] == 2.5

    def test_clamp_high(self, mock_env):
        result = mock_env.set_parameters({"wm": 999.0})
        assert result["validated_params"]["wm"] == 10.0

    def test_clamp_low(self, mock_env):
        result = mock_env.set_parameters({"im": -5.0})
        assert result["validated_params"]["im"] == 0.0

    def test_unknown_parameter_ignored(self, mock_env):
        result = mock_env.set_parameters({"wm": 2.0, "unknown_param": 99.0})
        assert "unknown_param" not in result["validated_params"]
        assert result["validated_params"]["wm"] == 2.0

    def test_all_ranges(self, mock_env):
        """Test that all parameter ranges are respected."""
        for param, (lo, hi) in PARAMETER_RANGES.items():
            # Test lower bound
            result = mock_env.set_parameters({param: lo - 1.0})
            assert result["validated_params"][param] == lo

            # Test upper bound
            result = mock_env.set_parameters({param: hi + 1.0})
            assert result["validated_params"][param] == hi


class TestNSEComputation:
    """Test Nash-Sutcliffe Efficiency calculation."""

    def test_perfect_nse(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nse = HydroEnvironment._compute_nse(obs, sim)
        assert nse == pytest.approx(1.0)

    def test_mean_prediction_nse(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.full(5, np.mean(obs))
        nse = HydroEnvironment._compute_nse(obs, sim)
        assert nse == pytest.approx(0.0)

    def test_negative_nse(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed
        nse = HydroEnvironment._compute_nse(obs, sim)
        assert nse < 0.0

    def test_constant_obs(self):
        obs = np.array([3.0, 3.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0])
        nse = HydroEnvironment._compute_nse(obs, sim)
        assert nse == 0.0  # Denominator is 0

    def test_known_nse(self):
        obs = np.array([10.0, 20.0, 30.0, 40.0])
        sim = np.array([12.0, 18.0, 32.0, 38.0])
        # Manual: mean(obs) = 25
        # SS_res = (10-12)^2 + (20-18)^2 + (30-32)^2 + (40-38)^2 = 4+4+4+4 = 16
        # SS_tot = (10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2 = 225+25+25+225 = 500
        # NSE = 1 - 16/500 = 0.968
        nse = HydroEnvironment._compute_nse(obs, sim)
        assert nse == pytest.approx(0.968)


class TestPeakTiming:
    """Test peak timing error computation."""

    def test_no_error(self):
        obs = np.array([1, 2, 5, 3, 1])
        sim = np.array([1, 3, 6, 2, 1])
        error = HydroEnvironment._peak_timing_error(obs, sim)
        assert error == 0

    def test_one_hour_lag(self):
        obs = np.array([1, 2, 5, 3, 1])
        sim = np.array([1, 2, 3, 6, 1])
        error = HydroEnvironment._peak_timing_error(obs, sim)
        assert error == 1


class TestControlFile:
    """Test control file generation and parameter injection."""

    def test_control_file_created(self, mock_env):
        assert mock_env.control_path.exists()

    def test_params_injected(self, mock_env):
        mock_env.set_parameters({"wm": 5.5, "alpha": 2.3})
        content = mock_env.control_path.read_text()
        assert "wm=5.5" in content
        assert "alpha=2.3" in content

    def test_output_dir_set(self, mock_env):
        content = mock_env.control_path.read_text()
        assert str(mock_env.output_dir) in content


class TestEnvironmentLifecycle:
    """Test environment state tracking."""

    def test_initial_state(self, mock_env):
        assert mock_env.run_count == 0
        assert len(mock_env.nse_history) == 0

    def test_evaluate_empty(self, mock_env):
        result = mock_env.evaluate()
        assert result["current_nse"] is None
        assert result["best_nse"] is None
        assert result["num_runs"] == 0

    def test_cleanup(self, mock_env):
        sandbox = mock_env.sandbox_dir
        assert sandbox.exists()
        mock_env.cleanup()
        assert not sandbox.exists()
