"""Configuration dataclasses and YAML loaders for HydroLLM."""

from __future__ import annotations

import os

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter range definitions
# ---------------------------------------------------------------------------

PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "wm": (0.1, 10.0),
    "b": (0.000001, 3.0),
    "im": (0.0, 1.0),
    "ke": (0.8, 1.2),
    "fc": (0.1, 2.0),
    "under": (0.1, 10.0),
    "leaki": (0.1, 10.0),
    "alpha": (0.1, 3.0),
    "beta": (0.1, 3.0),
    "alpha0": (0.0, 3.0),
    "iwu": (0.1, 100.0),
    "th": (10.0, 10.0),
    "isu": (0.0, 0.0),
}

# Parameters that are actually tunable (not fixed)
TUNABLE_PARAMETERS = [k for k, (lo, hi) in PARAMETER_RANGES.items() if lo != hi]

# Default parameter values (all multipliers = 1.0, states at typical values)
DEFAULT_PARAMETERS: dict[str, float] = {
    "wm": 1.0,
    "b": 1.0,
    "im": 1.0,
    "ke": 1.0,
    "fc": 1.0,
    "under": 1.0,
    "leaki": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "alpha0": 0.0,
    "iwu": 25.0,
    "th": 10.0,
    "isu": 0.0,
}


# ---------------------------------------------------------------------------
# Gage configuration
# ---------------------------------------------------------------------------

@dataclass
class GageConfig:
    """Configuration for a single USGS gage / watershed."""

    gage_id: str
    lon: float
    lat: float
    basin_area: float  # km²
    obs_dir: str
    control_template: str
    time_begin: str  # YYYYMMDDHHMM
    time_end: str
    target_nse: float = 0.8075
    ef5_timeout: int = 120  # seconds
    parameter_ranges: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        # Merge any overrides with defaults
        if not self.parameter_ranges:
            self.parameter_ranges = {
                k: list(v) for k, v in PARAMETER_RANGES.items()
            }


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a specific model to train."""

    name: str
    model_id: str  # HuggingFace model ID
    hf_repo_id: str = ""  # HuggingFace repo for pushing trained model
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    learning_rate: float = 5e-6
    grad_accum_steps: int = 4
    gpu_mode: str = "colocate"  # "colocate" or "server"


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Shared GRPO training hyperparameters."""

    # GRPO
    num_generations: int = 8
    max_completion_length: int = 2048
    kl_coef: float = 0.05
    num_train_epochs: int = 30

    # Multi-turn
    max_turns: int = 10

    # Environment
    ef5_timeout: int = 120

    # Gages
    gage_configs: list[str] = field(default_factory=list)

    # Logging
    report_to: str = "wandb"
    logging_steps: int = 1
    save_steps: int = 50


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------

def load_gage_config(path: str | Path) -> GageConfig:
    """Load a gage configuration from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    for key in ("obs_dir", "control_template"):
        if key in data:
            data[key] = resolve_data_path(data[key])
    return GageConfig(**data)


# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------

# Gage YAMLs, the control template, and the EF5 control writer all use the
# container path /app/data. Outside a container (e.g. on an HPC cluster) set
# HYDROLLM_DATA_ROOT to the directory holding the extracted data.tar.gz and
# every /app/data path is rewritten to it. Unset, behaviour is unchanged.
CONTAINER_DATA_ROOT = "/app/data"
DATA_ROOT = os.environ.get("HYDROLLM_DATA_ROOT", CONTAINER_DATA_ROOT).rstrip("/")


def resolve_data_path(path: str) -> str:
    """Map a /app/data/... path onto DATA_ROOT."""
    if path == CONTAINER_DATA_ROOT or path.startswith(CONTAINER_DATA_ROOT + "/"):
        return DATA_ROOT + path[len(CONTAINER_DATA_ROOT):]
    return path


def remap_data_paths(text: str) -> str:
    """Rewrite every /app/data/ occurrence in a control-file body."""
    if DATA_ROOT == CONTAINER_DATA_ROOT:
        return text
    return text.replace(CONTAINER_DATA_ROOT + "/", DATA_ROOT + "/")


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a model configuration from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)


def load_train_config(path: str | Path) -> TrainConfig:
    """Load shared training configuration from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


def load_all_gage_configs(train_config: TrainConfig) -> list[GageConfig]:
    """Load all gage configs referenced in a training config."""
    return [load_gage_config(p) for p in train_config.gage_configs]
