#!/usr/bin/env python3
"""Test the EF5 simulation environment locally (inside Docker).

Usage:
    docker build -t hydrollm-test -f Dockerfile .
    docker run hydrollm-test python3 /app/scripts/test_env_local.py

Or on Modal:
    modal run scripts/test_env_local.py
"""

import json
import sys
import os

# Add src to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hydrollm.config import DEFAULT_PARAMETERS, load_gage_config
from hydrollm.environment import HydroEnvironment


def main():
    print("=" * 60)
    print("HydroLLM Environment Sanity Check")
    print("=" * 60)

    # Load gage config
    gage_config_path = os.environ.get(
        "GAGE_CONFIG", "/app/configs/gages/02338660.yaml"
    )

    # Try to load from configs, fall back to inline config
    try:
        gage_cfg = load_gage_config(gage_config_path)
        print(f"✓ Loaded gage config from {gage_config_path}")
    except FileNotFoundError:
        from hydrollm.config import GageConfig
        gage_cfg = GageConfig(
            gage_id="02338660",
            lon=-84.9876,
            lat=33.2357,
            basin_area=328.93,
            obs_dir="/app/data/gauge_18_19",
            control_template="/app/data/docs/control.txt",
            time_begin="201807010000",
            time_end="201808312300",
        )
        print(f"✓ Using inline gage config for {gage_cfg.gage_id}")

    # Test 1: Create environment
    print("\n--- Test 1: Create HydroEnvironment ---")
    env = HydroEnvironment(gage_cfg)
    print(f"✓ Sandbox created at {env.sandbox_dir}")
    print(f"✓ Control file: {env.control_path}")
    assert env.control_path.exists(), "Control file was not created"

    # Test 2: Set parameters
    print("\n--- Test 2: Set Parameters ---")
    result = env.set_parameters({
        "wm": 2.0,
        "b": 0.5,
        "im": 0.05,
        "ke": 1.0,
        "fc": 1.0,
        "under": 1.0,
        "leaki": 1.0,
        "alpha": 1.0,
        "beta": 1.0,
        "alpha0": 0.0,
        "iwu": 25.0,
    })
    print(f"✓ Parameters set: {result['status']}")

    # Test 3: Parameter clamping
    print("\n--- Test 3: Parameter Clamping ---")
    result = env.set_parameters({"wm": 999.0, "im": -5.0})
    validated = result["validated_params"]
    assert validated["wm"] == 10.0, f"Expected wm=10.0, got {validated['wm']}"
    assert validated["im"] == 0.0, f"Expected im=0.0, got {validated['im']}"
    print(f"✓ Clamping works: wm=999→{validated['wm']}, im=-5→{validated['im']}")

    # Test 4: Run simulation
    print("\n--- Test 4: Run EF5 Simulation ---")
    # Reset to default params first
    env.set_parameters(DEFAULT_PARAMETERS)
    sim_result = env.run_simulation()
    print(f"  Status: {sim_result['status']}")
    if sim_result["status"] == "ok":
        print(f"  ✓ NSE: {sim_result['nse']}")
        print(f"  ✓ Peak sim: {sim_result['peak_sim_m3s']} m³/s")
        print(f"  ✓ Peak obs: {sim_result['peak_obs_m3s']} m³/s")
        print(f"  ✓ Volume ratio: {sim_result['volume_ratio']}")
        print(f"  ✓ Timing error: {sim_result['timing_error_hours']} hours")
    else:
        print(f"  ✗ Error: {sim_result.get('message', 'unknown')}")
        print(f"  Stderr: {sim_result.get('stderr', '')[:300]}")

    # Test 5: Evaluate
    print("\n--- Test 5: Evaluate ---")
    eval_result = env.evaluate()
    print(f"  ✓ Best NSE: {eval_result['best_nse']}")
    print(f"  ✓ NSE history: {eval_result['nse_history']}")
    print(f"  ✓ Target met: {eval_result['target_met']}")

    # Cleanup
    env.cleanup()
    print(f"\n✓ Sandbox cleaned up")

    print("\n" + "=" * 60)
    if sim_result["status"] == "ok":
        print("ALL TESTS PASSED ✓")
        print(f"Baseline NSE with default params: {sim_result['nse']}")
    else:
        print("SIMULATION FAILED — check EF5 installation and data paths")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
