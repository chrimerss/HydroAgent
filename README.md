# HydroLLM

**RL fine-tuning of LLMs for hydrologic model calibration.**

HydroLLM trains open-source tool-calling language models to calibrate the EF5/CREST distributed hydrologic model. The active pipeline is **multi-turn GRPO via verl + SGLang** on `Qwen3-4B-Instruct-2507`, with an in-process EF5 tool. A legacy SFT path (Qwen3-8B distilled from GPT-4o trajectories) is preserved but not the recommended starting point — bootstrapping the tool-call format via prompt + format reward (set in `verl_tools.py`) is sufficient.

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

Training uses **Qwen3-4B-Instruct-2507** with **full fine-tuning** in BF16 + FSDP across 4 H100 GPUs, served by SGLang during rollouts:

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Precision | BF16 |
| Tuning | Full FT (FSDP, no LoRA) |
| GPU | 4×H100 (80 GB each) |
| Attention | flash-attn 2 |
| Rollout backend | SGLang (multi-turn tool dispatch) |
| Training stack | verl 0.5 GRPO trainer |

**Why Qwen3-4B?** Qwen3-Instruct ships with native tool-calling (Hermes-style `<tool_call>` JSON) and the 4B variant is the largest model that runs comfortably with full FT + FSDP on 4×H100, leaving room for K=8 multi-turn rollouts in the rollout engine.

**Why verl + SGLang and not TRL/vLLM?** TRL's GRPOTrainer doesn't natively dispatch registered Python tools mid-rollout — multi-turn tool use requires manual interleaving of generation and tool execution. verl handles this via its multi-turn rollout abstraction. Within verl, **SGLang** is the only rollout backend (in v0.5) that actually invokes registered tool classes; vLLM in v0.5 only parses `<tool_call>` blocks for telemetry without dispatching to Python functions. Full FT is used because verl 0.5 + SGLang's weight-transfer path doesn't unwrap PEFT's `base_model.model.*` prefix, which makes LoRA infeasible without per-step merge-and-unload (not built in).

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
│   ├── tools.py                # Legacy tool defs + parser (TRL path)
│   ├── verl_tools.py           # verl-compatible BaseTool subclasses (RL)
│   ├── verl_reward.py          # verl custom_reward_function (RL)
│   ├── reward.py               # Legacy trajectory reward (TRL path)
│   ├── dataset.py              # HF dataset builder (TRL path)
│   ├── sft_dataset.py          # SFT dataset loader
│   ├── prompts.py              # System/user prompt templates
│   └── baseline.py             # Base model inference evaluator
│
├── modal_app/                  # Modal serverless deployment
│   ├── images.py               # train_image (verl/SGLang), sft_image, eval_image
│   ├── sft.py                  # SFT training entrypoint (legacy TRL)
│   ├── train.py                # verl GRPO training entrypoint
│   └── eval.py                 # Unified inference evaluation
│
├── configs/
│   ├── sft_config.yaml         # SFT hyperparameters (legacy)
│   ├── train_config.yaml       # Gage list + shared training settings
│   ├── verl/                   # verl GRPO configs (active)
│   │   ├── tool_config.yaml    # Tool schemas + class registration
│   │   └── qwen3_4b_grpo.yaml  # Hydra overlay on verl ppo_trainer
│   ├── models/                 # Legacy TRL-path model configs
│   │   ├── qwen3_8b.yaml
│   │   ├── qwen3_8b_sft.yaml
│   │   └── qwen3_8b_rl.yaml
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
│   ├── prepare_sft_data.py     # Convert calibration histories → JSONL (SFT)
│   ├── build_verl_dataset.py   # Convert gage configs → parquet (RL)
│   ├── test_env_local.py       # Environment sanity check
│   └── push_model.py           # Push model to HuggingFace
│
└── tests/                      # Unit tests (46 tests)
    ├── test_environment.py
    ├── test_tools.py
    └── test_reward.py
```

## Quick Start

### TL;DR — train the RL model

```bash
pip install modal
modal token new
modal secret create wandb HF_TOKEN=...      # one-time
modal secret create huggingface HF_TOKEN=...

modal run modal_app/train.py
```

That's it. The default path runs verl GRPO on `Qwen3-4B-Instruct-2507` for the gage at `configs/gages/02338660.yaml` on 4×H100. First image build is ~15 min (flash-attn compile); subsequent runs reuse cached layers.

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

### 3. Run Baseline Evaluation

Evaluate the base model _before_ training to establish the performance floor:

```bash
modal run modal_app/eval.py --model-id Qwen/Qwen3-4B-Instruct-2507
```

Results are saved to the `hydrollm-results` Modal Volume.

### 4. RL Training (verl GRPO + SGLang on 4×H100)

The default config trains `Qwen3-4B-Instruct-2507` directly with multi-turn rollouts and an in-process EF5 tool:

```bash
modal run modal_app/train.py
```

Override defaults via Hydra-style overrides (semicolon-separated):

```bash
# Shorter run
modal run modal_app/train.py --extra-overrides "trainer.total_epochs=10"

# Larger train batch
modal run modal_app/train.py --extra-overrides "data.train_batch_size=16;trainer.total_epochs=20"

# Different gage list
modal run modal_app/train.py --train-config configs/train_config.yaml
```

What happens under the hood, in order:
1. Modal builds the training image: `verlai/verl:app-verl0.5-sglang0.4.8-mcore0.13.0-te2.2` + EF5 binary + flash-attn (built from source on first run).
2. `scripts/build_verl_dataset.py` writes `train.parquet` / `val.parquet` to the Modal volume — each row carries the chat prompt plus `extra_info.tools_kwargs` so verl knows how to instantiate `HydroEnvironment` per rollout.
3. `python -m verl.trainer.main_ppo` is launched with the Hydra config at `configs/verl/qwen3_4b_grpo.yaml` (which composes on top of verl's `ppo_trainer` defaults via the `defaults` block + `hydra.searchpath`).
4. SGLang serves rollouts with K=8 generations per prompt; for each rollout it dispatches `set_parameters` → `run_simulation` → `evaluate` to the registered `BaseTool` subclasses in `src/hydrollm/verl_tools.py`. Each tool runs in-process and returns a per-turn reward (`+0.02` for valid `set_parameters`, `+0.05` for `run_simulation`, ΔNSE for `evaluate`, `−0.5` for invalid).
5. After the trajectory completes, `verl_reward.compute_score` adds the terminal NSE bonus (best-NSE clipped + `+0.5` if NSE > target − `0.02·n_runs`).
6. Checkpoints land in `/checkpoints/qwen3-4b-grpo` on the persistent Modal Volume; W&B logs `pg_loss`, `grad_norm`, `critic/score/mean`, and `actor/entropy` per step.

Monitor on [W&B](https://wandb.ai) — look for `critic/score/mean` rising above the validation step:0 baseline (~−0.6 for the raw model on gage 02338660) and `pg_loss` non-zero (zero `pg_loss` means no GRPO variance → broken setup, not just slow learning).

### 5. SFT Training (legacy)

Only useful if you want to compare against a distilled-from-GPT-4o baseline:

```bash
python scripts/prepare_sft_data.py     # builds data/sft_train.jsonl (2,576 examples)
modal run modal_app/sft.py             # uses sft_image (TRL stack), pushes to chrimerss/Qwen-3-8B-hydro-distill
```

The SFT path predates the verl RL path and uses the original Qwen3-8B + LoRA + TRL stack.

### 6. Evaluate

```bash
# Baseline
modal run modal_app/eval.py --model-id Qwen/Qwen3-4B-Instruct-2507

# RL-trained checkpoint
modal run modal_app/eval.py --model-id <your_hf_repo_or_local_volume_path>
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

### RL training (`configs/verl/qwen3_4b_grpo.yaml`)

Hydra overlay on top of verl's `ppo_trainer` defaults. Key knobs:

| Hydra key | Default | Description |
|-----------|---------|-------------|
| `algorithm.adv_estimator` | `grpo` | GRPO (group-relative advantages) |
| `data.train_batch_size` | 8 | Prompts per step (×K=8 rollouts) |
| `actor_rollout_ref.rollout.n` | 8 | K rollouts per prompt |
| `actor_rollout_ref.rollout.temperature` | 1.1 | Sampling temperature (>1 for exploration) |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | 10 | Max calibration rounds per rollout |
| `actor_rollout_ref.actor.optim.lr` | 5e-6 | Actor learning rate |
| `actor_rollout_ref.actor.kl_loss_coef` | 0.001 | KL penalty (low — leans on multi-turn variance) |
| `data.max_response_length` | 4096 | Max tokens per assistant turn |
| `trainer.total_epochs` | 30 | Training epochs |
| `trainer.save_freq` | 50 | Checkpoint cadence (steps) |
| `trainer.n_gpus_per_node` | 4 | GPUs (single Modal container) |

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

Two layers of signal: **per-turn** (returned by tools) and **terminal** (returned by `verl_reward.compute_score`):

**Per-turn (in `src/hydrollm/verl_tools.py`):**

| Tool call | Reward | Purpose |
|-----------|--------|---------|
| `set_parameters` (valid) | `+0.02` | Densify per-turn signal; reward protocol step |
| `run_simulation` (valid) | `+0.05` | Slightly higher: produces the artifact to evaluate |
| `evaluate` (valid) | `ΔNSE` | Real progress signal (this turn's NSE − previous) |
| Any tool (invalid) | `−0.5` | Format penalty |

Per-trajectory format-bonus is implicitly capped via `max_assistant_turns=10` (≈ +0.5 max).

**Terminal (in `src/hydrollm/verl_reward.py`):**

| Component | Value | Purpose |
|-----------|-------|---------|
| Best NSE | `[-1, 1]` | Primary signal (clipped) |
| Target bonus | `+0.5` | NSE exceeds gage's `target_nse` |
| Efficiency penalty | `-0.02` | Per evaluate call |
| Empty-trajectory penalty | `-1.0` | No NSE produced at all |

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

### Memory Budget (4×H100, 80 GB each, FSDP-sharded)

| Component | Per GPU | Notes |
|-----------|---------|-------|
| Qwen3-4B BF16 weights | ~2 GB | 8 GB total / 4 shards |
| Gradients (BF16) | ~2 GB | |
| AdamW states (FP32, m+v) | ~8 GB | 32 GB total / 4 shards |
| Activations (gradient checkpointing) | ~5 GB | depends on seq length |
| SGLang KV cache (rollout) | ~12 GB | `gpu_memory_utilization: 0.55` |
| **Total** | **~29 GB** | |
| **Headroom** | **~50 GB** | |

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
