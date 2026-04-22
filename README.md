# HydroLLM

**SFT + RL fine-tuning of LLMs for hydrologic model calibration.**

HydroLLM trains open-source tool-calling language models to calibrate the EF5/CREST distributed hydrologic model via a two-stage pipeline:

1. **SFT (Supervised Fine-Tuning)**: Distill GPT-4o calibration expertise into Qwen3-8B using 2,500+ multi-turn trajectories across 29 USGS gages → [`Qwen-3-8B-hydro-distill`](https://huggingface.co/chrimerss/Qwen-3-8B-hydro-distill)
2. **RL (GRPO)**: Reinforce with online EF5 simulation feedback on target gages → [`Qwen-3-8B-hydroLLM`](https://huggingface.co/chrimerss/Qwen-3-8B-hydroLLM)

## Motivation

Most LLM agents cannot reliably calibrate hydrologic models. They lack the domain reasoning to propose physically plausible parameters and iteratively converge on a good solution. HydroLLM addresses this by training model weights directly with simulation feedback, so the model _internalizes_ hydrologic calibration reasoning rather than relying on prompt engineering alone.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│             Three-Phase Training Pipeline                   │
│                                                             │
│  Phase 0: Base LLM (Qwen3-8B) — baseline performance       │
│                      ↓                                      │
│  Phase 1: SFT — distill GPT-4o calibration trajectories     │
│           2,576 examples × 29 gages × quality-weighted      │
│           → Qwen-3-8B-hydro-distill                         │
│                      ↓                                      │
│  Phase 2: RL (GRPO) — online EF5 simulation feedback        │
│           K=8 rollouts per prompt, NSE reward signal         │
│           → Qwen-3-8B-hydroLLM                              │
└─────────────────────────────────────────────────────────────┘
```

The SFT stage teaches the model calibration reasoning and tool usage from 73 expert trajectories across diverse US watersheds. The RL stage then refines this with real EF5 simulation feedback.

## Model

Training uses **Qwen3-8B** with LoRA adapters in BF16 precision on a single H100 GPU:

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen3-8B` |
| Precision | BF16 (no quantization) |
| LoRA rank | 16 (α=32) |
| GPU | 1×H100 (80 GB) |
| Attention | PyTorch SDPA |
| Training stack | PEFT + TRL GRPOTrainer |

**Why Qwen3?** Qwen3's dual-mode reasoning (thinking + non-thinking) and improved tool-calling are well-suited for iterative hydrologic calibration, where the model must reason about parameter adjustments based on hydrograph errors.

**Why not Unsloth/vLLM?** Both have compatibility issues with the current PyTorch/CUDA stack:
- Unsloth's compiled GRPO trainer has a chunked log-softmax bug (shape mismatch in `torch.gather`)
- vLLM's torch.compile has a SymInt bug with bitsandbytes quantization

The standard PEFT + TRL stack is more stable and 8B in BF16 (~16 GB) fits comfortably on a single H100 with room for LoRA gradients and optimizer states.

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
│   ├── dataset.py              # Multi-gage HF dataset builder (GRPO)
│   ├── sft_dataset.py          # SFT dataset loader
│   ├── prompts.py              # System/user prompt templates
│   └── baseline.py             # Base model inference evaluator
│
├── modal_app/                  # Modal serverless deployment
│   ├── images.py               # Modal image definitions
│   ├── sft.py                  # SFT training entrypoint
│   ├── train.py                # GRPO training entrypoint
│   └── eval.py                 # Unified inference evaluation
│
├── configs/
│   ├── sft_config.yaml         # SFT hyperparameters
│   ├── train_config.yaml       # Shared GRPO hyperparameters
│   ├── models/                 # Per-model training configs
│   │   ├── qwen3_8b.yaml       # Base model (Qwen3-8B)
│   │   ├── qwen3_8b_sft.yaml   # SFT distillation config
│   │   └── qwen3_8b_rl.yaml    # RL config (starts from SFT)
│   └── gages/                  # Per-gage watershed configs
│       └── 02338660.yaml
│
├── data/                       # Generated training data
│   └── sft_train.jsonl         # SFT conversations (2,576 examples)
│
├── sets_for_SFT_RL/            # Raw GPT-4o calibration histories
│                               # (73 experiments × 29 gages)
│
├── scripts/                    # Utility scripts
│   ├── prepare_sft_data.py     # Convert calibration histories → JSONL
│   ├── test_env_local.py       # Environment sanity check
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

Evaluate the base Qwen3 model _before_ any training to establish the performance floor:

```bash
modal run modal_app/eval.py --model-id Qwen/Qwen3-8B
```

Results are saved to the `hydrollm-results` Modal Volume.

### 4. Prepare SFT Data

Convert GPT-4o calibration histories to multi-turn chat format:

```bash
python scripts/prepare_sft_data.py
```

This processes 73 calibration experiments across 29 gages and outputs `data/sft_train.jsonl` (2,576 quality-weighted examples).

### 5. SFT Training (Phase 1)

Distill GPT-4o calibration expertise into Qwen3-8B:

```bash
modal run modal_app/sft.py
```

The SFT model is automatically pushed to [`chrimerss/Qwen-3-8B-hydro-distill`](https://huggingface.co/chrimerss/Qwen-3-8B-hydro-distill).

### 6. GRPO RL Training (Phase 2)

Reinforce the SFT model with online EF5 simulation feedback:

```bash
modal run modal_app/train.py \
    --model-config configs/models/qwen3_8b_rl.yaml
```

The RL model is automatically pushed to [`chrimerss/Qwen-3-8B-hydroLLM`](https://huggingface.co/chrimerss/Qwen-3-8B-hydroLLM).

Monitor training on [W&B](https://wandb.ai) — look for increasing mean reward (NSE) and bounded KL divergence.

### 7. Evaluate All Models

```bash
# Baseline (raw Qwen3-8B)
modal run modal_app/eval.py --model-id Qwen/Qwen3-8B

# Experiment: SFT model
modal run modal_app/eval.py --model-id chrimerss/Qwen-3-8B-hydro-distill

# Experiment: RL model
modal run modal_app/eval.py --model-id chrimerss/Qwen-3-8B-hydroLLM
```

Results are tagged as `baseline` or `experiment` and saved to the `hydrollm-results` Modal Volume.

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

### Model Config (`configs/models/qwen3_8b.yaml`)

| Parameter | Value |
|-----------|-------|
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `learning_rate` | 5e-6 |
| `gpu_mode` | colocate |

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

## Technical Notes

### Known Compatibility Issues

- **Unsloth GRPO bug**: Unsloth's compiled `chunked_hidden_states_selective_log_softmax` has a dimension mismatch (index 32 tokens longer than logits). Affects all models. Workaround: use standard TRL GRPOTrainer.
- **vLLM + bitsandbytes**: vLLM's torch.compile produces `SymInt` errors when the model uses bitsandbytes 4-bit quantization. Workaround: use BF16 precision, disable vLLM for rollouts.
- **vLLM V1 engine**: `VLLM_USE_V1=0` env var is not reliably respected. Both V0 and V1 trigger the SymInt crash.
- **flash-attn**: Requires CUDA SDK (`CUDA_HOME`) installed in the container. Modal's base pytorch images don't include it. Workaround: use PyTorch's built-in SDPA (`attn_implementation="sdpa"`).

### Memory Budget (1×H100, 80 GB)

| Component | Estimate |
|-----------|----------|
| Qwen3-8B BF16 weights | ~16 GB |
| LoRA adapters | ~0.1 GB |
| Optimizer states (AdamW) | ~0.4 GB |
| Activations (gradient checkpointing) | ~8 GB |
| KV cache (8 rollouts × 2048 tokens) | ~12 GB |
| **Headroom** | **~43 GB** |

## Roadmap

- [x] **Phase 0**: Baseline evaluation infrastructure
- [x] **Phase 1**: SFT distillation from GPT-4o calibration trajectories (29 gages)
- [x] **Phase 2**: GRPO RL on single gage (02338660)
- [ ] **Phase 3**: Scale RL to 20+ CONUS gages
- [ ] **Phase 4**: Paper & model release on HuggingFace

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
