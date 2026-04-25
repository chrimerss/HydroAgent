"""Build a parquet dataset in verl's expected schema.

verl trainer expects each row to have:
    - prompt:        list[{role, content}]   initial chat messages
    - data_source:   str                      dataset tag for reward routing
    - ability:       str                      task category (informational)
    - reward_model:  {style, ground_truth}    style="custom" → use custom_reward_function
    - extra_info:    dict                     forwarded to tools (tools_kwargs)

For each gage we emit `n_repeat` identical rows so GRPO has enough samples
per epoch (within-group variance comes from K rollouts, not from the data).

Usage:
    python scripts/build_verl_dataset.py \
        --train-config configs/train_config.yaml \
        --out data/verl/train.parquet \
        --n-repeat 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hydrollm.config import load_gage_config, load_train_config  # noqa: E402
from hydrollm.prompts import build_messages  # noqa: E402


TOOL_NAMES = ("set_parameters", "run_simulation", "evaluate")


def _row_for_gage(gage_path: str) -> dict:
    gage = load_gage_config(gage_path)
    messages = build_messages(gage)
    tools_kwargs = {
        name: {
            "create_kwargs": {"gage_config_path": gage_path},
        }
        for name in TOOL_NAMES
    }
    return {
        "prompt": messages,
        "data_source": "hydrollm",
        "ability": "hydrologic_calibration",
        "reward_model": {
            "style": "custom",
            "ground_truth": {"target_nse": gage.target_nse, "gage_id": gage.gage_id},
        },
        "extra_info": {
            "gage_id": gage.gage_id,
            "tools_kwargs": tools_kwargs,
            "need_tools_kwargs": True,
        },
    }


def build(train_config: str, out_path: str, n_repeat: int) -> None:
    cfg = load_train_config(train_config)
    if not cfg.gage_configs:
        raise ValueError("train_config has no gage_configs")

    rows = [_row_for_gage(p) for p in cfg.gage_configs]
    rows = rows * n_repeat
    df = pd.DataFrame(rows)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", default="configs/train_config.yaml")
    ap.add_argument("--out", default="data/verl/train.parquet")
    ap.add_argument("--val-out", default="data/verl/val.parquet")
    ap.add_argument("--n-repeat", type=int, default=256)
    ap.add_argument("--n-val-repeat", type=int, default=8)
    args = ap.parse_args()

    build(args.train_config, args.out, args.n_repeat)
    build(args.train_config, args.val_out, args.n_val_repeat)


if __name__ == "__main__":
    main()
