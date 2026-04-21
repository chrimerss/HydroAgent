# HydroLLM

**Reinforcement learning fine-tuning of LLMs with hydrologic simulation feedback.**

HydroLLM trains open-source tool-calling language models to calibrate the EF5/CREST distributed hydrologic model using GRPO (Group Relative Policy Optimization). The reward signal comes directly from Nash-Sutcliffe Efficiency (NSE) scores computed after running real hydrologic simulations — no human feedback required.

## Motivation

Most LLM agents cannot reliably calibrate hydrologic models. They lack the domain reasoning to propose physically plausible parameters and iteratively converge on a good solution. HydroLLM addresses this by training model weights directly with simulation feedback, so the model _internalizes_ hydrologic calibration reasoning rather than relying on prompt engineering alone.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                   │
│                                                         │
│  1. Model receives gage info + parameter ranges         │
│  2. Model reasons about watershed → calls set_params()  │
│  3. EF5/CREST runs simulation → returns hydrograph      │
│  4. Model analyzes errors → adjusts parameters          │
│  5. Repeat for N turns                                  │
│  6. Best NSE across turns → trajectory reward           │
│  7. GRPO updates weights using group-relative advantage │
└─────────────────────────────────────────────────────────┘
```

Each training step generates **K=8 rollouts** (multi-turn calibration trajectories). GRPO compares them by relative reward and updates the policy — no critic model needed.

## Models

All models use the Qwen2.5-Instruct family with LoRA adapters:

| Model | GPU Config | Use Case |
|-------|-----------|----------|
| Qwen2.5-7B-Instruct | 1–2 GPU (colocate) | Fast iteration, debugging |
| Qwen2.5-32B-Instruct | 4 GPU (server mode) | Strong mid-scale |
| Qwen2.5-72B-Instruct | 8 GPU (server + FSDP) | Maximum reasoning |

Training uses **Unsloth** for ~90% VRAM reduction over standard TRL, making 72B feasible on 8×H100.

## Project Structure

```
HydroLLM/
├── Dockerfile                  # EF5/CREST simulation environment
├── Dockerfile.train            # Combined training + simulation image
├── control.txt                 # EF5 config template
├── instruction.md              # Calibration task specification
├── pyproject.toml              # Project configuration
│
├── src/hydrollm/               # Core library
│   ├── config.py               # Configuration dataclasses + YAML loaders
│   ├── environment.py          # Thread-safe EF5 simulation sandbox
│   ├── tools.py                # Tool definitions + executor + parser
│   ├── reward.py               # Trajectory-level NSE reward function
│   ├── dataset.py              # Multi-gage HF dataset builder
│   ├── prompts.py              # System/user prompt templates
│   └── baseline.py             # Base model inference evaluator
│
├── modal_app/                  # Modal serverless deployment
│   ├── images.py               # Modal image definitions
│   ├── train.py                # GRPO training entrypoint
│   ├── baseline.py             # Baseline evaluation entrypoint
│   └── eval.py                 # Post-training evaluation
│
├── configs/
│   ├── models/                 # Per-model training configs
│   │   ├── qwen2.5_7b.yaml
│   │   ├── qwen2.5_32b.yaml
│   │   └── qwen2.5_72b.yaml
│   ├── gages/                  # Per-gage watershed configs
│   │   └── 02338660.yaml
│   └── train_config.yaml       # Shared GRPO hyperparameters
│
├── scripts/                    # Utility scripts
│   ├── test_env_local.py       # Environment sanity check
│   ├── run_baseline.sh         # Run all baselines
│   └── push_model.py           # Push model to HuggingFace
│
└── tests/                      # Unit tests (46 tests)
    ├── test_environment.py
    ├── test_tools.py
    └── test_reward.py
```

## Quick Start

### Prerequisites

- [Modal](https://modal.com) account with API token
- [Weights & Biases](https://wandb.ai) account (for training monitoring)
- [HuggingFace](https://huggingface.co) token (for model download/upload)

### 1. Install Modal

```bash
pip install modal
modal token new
```

### 2. Configure Secrets

```bash
modal secret create wandb WANDB_API_KEY=your_wandb_key
modal secret create huggingface HF_TOKEN=your_hf_token
```

### 3. Run Baseline Evaluation (Phase 0)

Evaluate the base Qwen2.5 models _before_ any training to establish the performance floor:

```bash
# Single model
modal run modal_app/baseline.py \
    --model-config configs/models/qwen2.5_7b.yaml

# All 3 models
modal run modal_app/baseline.py --all
```

Results are saved to the `hydrollm-results` Modal Volume.

### 4. Train with GRPO (Phase 1)

Start with the 7B model for fast iteration:

```bash
modal run modal_app/train.py \
    --model-config configs/models/qwen2.5_7b.yaml
```

Scale up after validating the pipeline:

```bash
# 32B model
modal run modal_app/train.py \
    --model-config configs/models/qwen2.5_32b.yaml

# 72B model (requires 8×H100)
modal run modal_app/train.py \
    --model-config configs/models/qwen2.5_72b.yaml
```

Monitor training on [W&B](https://wandb.ai) — look for increasing mean reward (NSE) and bounded KL divergence.

### 5. Evaluate Trained Model

```bash
modal run modal_app/eval.py \
    --model-path /checkpoints/qwen2.5-7b/final
```

### 6. Push to HuggingFace

```bash
python scripts/push_model.py \
    --model-path /checkpoints/qwen2.5-7b/final \
    --repo-id your-username/hydrollm-qwen2.5-7b
```

## Local Development

### Run Unit Tests

Tests validate parameter logic, NSE math, tool parsing, and reward computation — no GPU or EF5 needed:

```bash
pip install pyyaml numpy pytest
PYTHONPATH=src pytest tests/ -v
```

### Test EF5 Environment (Docker)

```bash
docker build -t hydrollm-test -f Dockerfile .
docker run hydrollm-test python3 scripts/test_env_local.py
```

## Configuration

### Training Hyperparameters (`configs/train_config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | 8 | Rollouts per prompt (K in GRPO) |
| `max_completion_length` | 2048 | Max tokens per generation |
| `kl_coef` | 0.05 | KL divergence penalty |
| `num_train_epochs` | 30 | Training epochs |
| `max_turns` | 10 | Max calibration rounds per rollout |
| `ef5_timeout` | 120 | Seconds per EF5 simulation |

### Model Configs (`configs/models/*.yaml`)

| Parameter | 7B | 32B | 72B |
|-----------|----|-----|-----|
| `lora_r` | 16 | 16 | 8 |
| `lora_alpha` | 32 | 32 | 16 |
| `learning_rate` | 5e-6 | 2e-6 | 1e-6 |
| `grad_accum_steps` | 4 | 8 | 16 |
| `gpu_mode` | colocate | server | server |

### Adding New Gages (Phase 2)

1. Create a YAML file in `configs/gages/`:
   ```yaml
   gage_id: "01632000"
   lon: -78.1234
   lat: 38.5678
   basin_area: 500.0
   obs_dir: /app/data/gauge_observations
   control_template: /app/data/docs/control.txt
   time_begin: "201807010000"
   time_end: "201808312300"
   target_nse: 0.8
   ```

2. Add data (precipitation, PET, observations) to the Docker image or Modal Volume

3. Add the config path to `train_config.yaml`:
   ```yaml
   gage_configs:
     - configs/gages/02338660.yaml
     - configs/gages/01632000.yaml
   ```

## Reward Function

The trajectory-level reward encourages calibration quality and efficiency:

| Component | Value | Purpose |
|-----------|-------|---------|
| Best NSE | `[-1, 1]` | Primary signal (clipped) |
| Target bonus | `+0.5` | Reward for NSE > 0.8075 |
| Improvement bonus | `+0.2` | Reward for improving across turns |
| Error penalty | `-0.5` | Per invalid tool call |
| Efficiency penalty | `-0.02` | Per simulation run |

## Tools

The model has access to three tools during calibration:

| Tool | Description |
|------|-------------|
| `set_parameters` | Set 11 tunable CREST parameter multipliers (wm, b, im, ke, fc, under, leaki, alpha, beta, alpha0, iwu) |
| `run_simulation` | Execute EF5 and return NSE, peak flows, volume ratio, timing error |
| `evaluate` | Get calibration progress: NSE history, best NSE, target status |

## Roadmap

- [x] **Phase 0**: Baseline evaluation infrastructure
- [x] **Phase 1**: Multi-turn GRPO on single gage (02338660)
- [ ] **Phase 2**: Scale to 20 CONUS gages
- [ ] **Phase 3**: Paper & model release on HuggingFace

## Citation

If you use HydroLLM in your research, please cite:

```bibtex
@software{hydrollm2026,
  title={HydroLLM: Reinforcement Learning Fine-Tuning of LLMs with Hydrologic Simulation Feedback},
  year={2026},
  url={https://github.com/chrimerss/HydroLLM}
}
```

## License

MIT
