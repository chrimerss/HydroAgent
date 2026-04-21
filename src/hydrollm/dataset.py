"""Dataset builder for GRPO training.

Constructs a HuggingFace Dataset where each example is a prompt
asking the model to calibrate a specific gage. For single-gage
training (Phase 1), the same prompt is repeated to allow GRPO
to learn from variance across K rollouts.
"""

from __future__ import annotations

from datasets import Dataset

from hydrollm.config import TrainConfig, load_gage_config
from hydrollm.prompts import build_messages


def build_dataset(train_config: TrainConfig) -> Dataset:
    """Build a training dataset from gage configurations.

    Each example contains the full initial message list (system + user)
    formatted as a single prompt string for the GRPO trainer.

    Args:
        train_config: Training configuration with gage config paths.

    Returns:
        HuggingFace Dataset with 'prompt' column.
    """
    prompts = []

    for gage_path in train_config.gage_configs:
        gage_cfg = load_gage_config(gage_path)
        messages = build_messages(gage_cfg)
        prompts.append({
            "prompt": messages,
            "gage_id": gage_cfg.gage_id,
        })

    if not prompts:
        raise ValueError(
            "No gage configs found. Check train_config.gage_configs paths."
        )

    # For single-gage training, repeat the prompt to create a dataset
    # of sufficient size for GRPO. The variance comes from the K rollouts,
    # not from different prompts.
    min_dataset_size = max(train_config.num_train_epochs * 10, 100)
    if len(prompts) < min_dataset_size:
        multiplier = (min_dataset_size // len(prompts)) + 1
        prompts = prompts * multiplier

    return Dataset.from_list(prompts)


def build_eval_dataset(gage_paths: list[str]) -> Dataset:
    """Build an evaluation dataset (one example per gage, no repetition).

    Args:
        gage_paths: List of paths to gage YAML configs.

    Returns:
        HuggingFace Dataset with 'prompt' and 'gage_id' columns.
    """
    prompts = []
    for gage_path in gage_paths:
        gage_cfg = load_gage_config(gage_path)
        messages = build_messages(gage_cfg)
        prompts.append({
            "prompt": messages,
            "gage_id": gage_cfg.gage_id,
        })
    return Dataset.from_list(prompts)
