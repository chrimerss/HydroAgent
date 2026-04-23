#!/usr/bin/env python3
"""Calculate NSE from EF5 output CSV file.

Usage: python calc_nse.py <csv_file_path>
"""

import sys
import csv
import numpy as np
import json


def compute_nse(obs, sim):
    """Compute Nash-Sutcliffe Efficiency."""
    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_clean = obs[valid_mask]
    sim_clean = sim[valid_mask]

    if len(obs_clean) == 0:
        return -1.0

    obs_mean = np.mean(obs_clean)
    denominator = np.sum((obs_clean - obs_mean) ** 2)

    if denominator == 0:
        return 0.0

    numerator = np.sum((obs_clean - sim_clean) ** 2)
    return float(1.0 - numerator / denominator)


def parse_csv(csv_path):
    """Parse EF5 CSV and extract observed and simulated values."""
    obs_values = []
    sim_values = []

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            # Try to find the correct column names
            fieldnames = reader.fieldnames or []
            obs_col = None
            sim_col = None

            for col in fieldnames:
                if 'observed' in col.lower() or 'obs' in col.lower():
                    obs_col = col
                if ('discharge' in col.lower() or 'sim' in col.lower()) and 'observed' not in col.lower():
                    sim_col = col

            # Fall back to common patterns
            if obs_col is None:
                for col in fieldnames:
                    if col.strip().lower() in ['observed', 'obs', 'observed_discharge']:
                        obs_col = col
                        break

            if sim_col is None:
                for col in fieldnames:
                    if col.strip().lower() in ['discharge', 'sim', 'simulated']:
                        sim_col = col
                        break

            if obs_col is None or sim_col is None:
                print(f"Warning: Could not identify columns. Available: {fieldnames}", file=sys.stderr)
                return None, None

            # Parse rows
            skipped = 0
            for row in reader:
                try:
                    obs_val = float(row[obs_col]) if row[obs_col] else float('nan')
                    sim_val = float(row[sim_col]) if row[sim_col] else float('nan')

                    # Filter out missing values (EF5 uses -999 as nodata)
                    if np.isnan(obs_val) or np.isnan(sim_val) or obs_val <= -999 or sim_val <= -999:
                        skipped += 1
                        continue

                    obs_values.append(obs_val)
                    sim_values.append(sim_val)

                except (ValueError, KeyError):
                    skipped += 1
                    continue

            if skipped > 0:
                print(f"Skipped {skipped} rows with invalid/missing data", file=sys.stderr)

            if not obs_values:
                print("Warning: No valid data found in CSV", file=sys.stderr)
                return None, None

            return np.array(obs_values), np.array(sim_values)

    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}", file=sys.stderr)
        return None, None


def main():
    if len(sys.argv) < 2:
        print("Usage: python calc_nse.py <csv_file_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    obs, sim = parse_csv(csv_path)

    if obs is None or sim is None:
        result = {
            "status": "error",
            "message": "Failed to parse CSV or no valid data",
            "nse": -1.0
        }
    else:
        nse = compute_nse(obs, sim)
        result = {
            "status": "ok",
            "message": f"NSE calculated from {len(obs)} data points",
            "nse": round(nse, 4),
            "num_points": len(obs)
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
