"""Modal entrypoint for verl-based GRPO training.

Runs verl's `main_ppo` (Hydra) on a single 4×H100 container with Ray in
local mode. EF5 is baked into the training image so multi-turn tool
calls execute in-process.

Usage:
    # Default: Qwen3-4B-Instruct, single gage, 4×H100
    modal run modal_app/train.py

    # Override the GRPO config name
    modal run modal_app/train.py --config-name qwen3_4b_grpo

    # Override individual Hydra keys (semicolon-separated)
    modal run modal_app/train.py \
        --extra-overrides "trainer.total_epochs=10;data.train_batch_size=16"
"""

from __future__ import annotations

import modal

from modal_app.images import train_image

app = modal.App("hydrollm-verl")

checkpoint_vol = modal.Volume.from_name("hydrollm-checkpoints", create_if_missing=True)
data_vol = modal.Volume.from_name("hydrollm-data", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="H100:4",
    timeout=86400,  # 24h (Modal max). Each GRPO step is ~30min with K=8 ×
                    # max_turns=50; without bumping this, the function gets
                    # killed before save_freq=10 emits the first checkpoint.
    volumes={"/checkpoints": checkpoint_vol, "/data_vol": data_vol},
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface"),
    ],
    memory=131072,  # 128 GiB
    # 64 CPUs: K=8 rollouts × train_batch=8 = 64 simultaneous EF5 processes;
    # each EF5 sim is CPU-bound, so undersubscribing CPUs causes 300s+ timeouts.
    cpu=64.0,
)
def train(
    config_name: str = "qwen3_4b_grpo",
    config_path: str = "/workspace/configs/verl",
    extra_overrides: str = "",
    train_config: str = "configs/train_config.yaml",
    n_repeat: int = 256,
    n_val_repeat: int = 8,
):
    """Build the parquet dataset, then launch verl GRPO training.

    Args:
        config_name: name of the Hydra config in configs/verl/.
        config_path: directory of the Hydra config.
        extra_overrides: semicolon-separated Hydra overrides
            (e.g. "trainer.total_epochs=10;data.train_batch_size=16").
        train_config: path (relative to /workspace) to the gage list.
        n_repeat: how many times to repeat each gage prompt in train set.
        n_val_repeat: same, for the val set.
    """
    import logging
    import os
    import subprocess
    import sys

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hydrollm.verl_train")

    os.chdir("/workspace")
    sys.path.insert(0, "/workspace/src")

    # 1) Build parquet dataset (verl reads parquet from disk).
    logger.info("Building verl parquet dataset...")
    subprocess.run(
        [
            "python",
            "/workspace/scripts/build_verl_dataset.py",
            "--train-config",
            f"/workspace/{train_config}",
            "--out",
            "/workspace/data/verl/train.parquet",
            "--val-out",
            "/workspace/data/verl/val.parquet",
            "--n-repeat",
            str(n_repeat),
            "--n-val-repeat",
            str(n_val_repeat),
        ],
        check=True,
    )

    # 2) Compose Hydra overrides.
    overrides = []
    if extra_overrides.strip():
        overrides.extend(o.strip() for o in extra_overrides.split(";") if o.strip())

    # 3) Launch verl trainer. Ray initializes in local mode automatically
    #    when no RAY_ADDRESS is set; verl handles GPU placement via
    #    trainer.n_gpus_per_node=4.
    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config_path}",
        f"--config-name={config_name}",
        *overrides,
    ]
    logger.info("Launching verl: %s", " ".join(cmd))

    env = os.environ.copy()
    env.setdefault("WANDB_PROJECT", "hydrollm")
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, check=True, env=env)

    checkpoint_vol.commit()
    logger.info("verl training complete.")


@app.local_entrypoint()
def main(
    config_name: str = "qwen3_4b_grpo",
    config_path: str = "/workspace/configs/verl",
    extra_overrides: str = "",
    train_config: str = "configs/train_config.yaml",
    n_repeat: int = 256,
    n_val_repeat: int = 8,
):
    train.remote(
        config_name=config_name,
        config_path=config_path,
        extra_overrides=extra_overrides,
        train_config=train_config,
        n_repeat=n_repeat,
        n_val_repeat=n_val_repeat,
    )
