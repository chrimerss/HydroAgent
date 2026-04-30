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

Training uses **Qwen3-4B-Instruct-2507** with **full fine-tuning** in BF16 + FSDP across 8 H100 GPUs, served by SGLang during rollouts:

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Precision | BF16 |
| Tuning | Full FT (FSDP, no LoRA) |
| GPU | 4×H100 (80 GB each) — GPU is *not* the bottleneck (MFU ~0.20-0.25); 4 H100 is the cost-efficient choice over 8 |
| Attention | flash-attn 2 (built from source on first image build) |
| Rollout backend | SGLang (multi-turn tool dispatch) |
| Training stack | verl 0.5 GRPO trainer |
| EF5 concurrency | 32 simultaneous (per-worker semaphore × num workers) |

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
│   ├── images.py               # train_image (verl/SGLang/EF5), sft_image
│   ├── sft.py                  # SFT training entrypoint (legacy TRL)
│   ├── train.py                # verl GRPO training entrypoint
│   └── eval.py                 # SGLang multi-turn eval (HF model OR verl ckpt)
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
modal secret create wandb WANDB_API_KEY=...    # one-time
modal secret create huggingface HF_TOKEN=...

# Detached so the run survives your laptop closing / wifi drop:
modal run --detach modal_app/train.py --n-repeat 32 --n-val-repeat 1
```

The default path runs verl GRPO on `Qwen3-4B-Instruct-2507` over **10 CONUS gages** (small/medium basins, ≤2401 km²) on 4×H100. First image build is ~15 min (flash-attn compile); subsequent runs reuse cached layers.

**Use `--detach`** — each training step is ~15-30 minutes wall time (K=8 rollouts × up to 50 multi-turn EF5 calls). With `save_freq=10` and 24h Modal function timeout, the run auto-resumes (`resume_mode: auto`) across timeout cycles, so the only thing you need is internet briefly to launch.

Watch live on the Modal dashboard URL printed at launch, or on W&B (project `hydrollm`). Look for: `pg_loss` non-zero (gradient flowing), `actor/entropy ≈ 0.30` (not collapsed), and `critic/score/mean` rising above the validation-step:0 baseline of `~-0.4`.

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

Evaluate the base model _before_ training to establish the performance floor.
This runs SGLang inference + a multi-turn calibration loop on the held-out
test gage (`02338660`) — same protocol used to evaluate trained checkpoints,
so results are directly comparable.

```bash
modal run modal_app/eval.py --model-id Qwen/Qwen3-4B-Instruct-2507
```

Result is saved to `hydrollm-results/sglang_baseline_Qwen3-4B-Instruct-2507_02338660.json`
on the `hydrollm-results` Modal Volume.

### 4. RL Training (verl GRPO + SGLang on 4×H100)

The default config trains `Qwen3-4B-Instruct-2507` directly with multi-turn rollouts and an in-process EF5 tool over 10 CONUS gages:

```bash
# Recommended — detached so the launch terminal can disconnect:
modal run --detach modal_app/train.py --n-repeat 32 --n-val-repeat 1
```

Override defaults via Hydra-style overrides (semicolon-separated):

```bash
# Shorter run (10 epochs ≈ 800 steps)
modal run --detach modal_app/train.py --extra-overrides "trainer.total_epochs=10"

# Different gage list
modal run --detach modal_app/train.py --train-config configs/train_config.yaml
```

What happens under the hood, in order:
1. Modal builds the training image: `verlai/verl:app-verl0.5-sglang0.4.8-mcore0.13.0-te2.2` + EF5 binary + flash-attn (built from source on first run, cached after).
2. The data layer pulls a *commit-pinned* `data.tar.gz` from `chrimerss/hydro_cali_agent_example` (`pet/`, `basic_data/`, `gauge/USGS_*_1h_UTC.csv`, per-gage `data_mrms_clip/<gage_id>/`, `docs/control.txt`) and strips macOS AppleDouble metadata. Bump the SHA in `modal_app/images.py` whenever you re-upload `data.tar.gz`.
3. `scripts/build_verl_dataset.py` writes `train.parquet` / `val.parquet` to the Modal volume — each row carries the chat prompt plus `extra_info.tools_kwargs` so verl knows how to instantiate `HydroEnvironment` per rollout.
4. `python -m verl.trainer.main_ppo` is launched with the Hydra config at `configs/verl/qwen3_4b_grpo.yaml` (composes on top of verl's `ppo_trainer` defaults via `defaults: [ppo_trainer, _self_]` + `hydra.searchpath`).
5. SGLang serves rollouts with K=6 generations per prompt; for each rollout it dispatches `set_parameters` → `run_simulation` → `evaluate` to the registered `BaseTool` subclasses in `src/hydrollm/verl_tools.py`. EF5 invocations are gated by a per-worker asyncio semaphore (`_EF5_CONCURRENCY=8`, × 4 workers = 32 system-wide) to avoid CPU/IO contention.
6. Per-turn rewards: `+0.02` for valid `set_parameters`, `+0.05` for `run_simulation`, ΔNSE for `evaluate`, `−0.5` for invalid. Terminal: `verl_reward.compute_score` adds best-NSE (clipped) + `+0.5` if NSE > target − `0.02·n_runs`.
7. Checkpoints land in `/checkpoints/qwen3-4b-grpo` on the persistent Modal Volume every 10 steps. `resume_mode: auto` recovers from the 24h Modal function-timeout cycle automatically.

Monitor on [W&B](https://wandb.ai) — look for `critic/score/mean` rising above the validation step:0 baseline (`-0.41`) and `pg_loss` non-zero (zero `pg_loss` means no GRPO variance → mode collapse, not just slow learning).

#### Updating the data tarball

If you change gages or time windows:

```bash
# 1. (optional) Re-run the audit to pick clean flood-event windows:
python scripts/audit_flood_windows.py
python scripts/apply_audit.py

# 2. Edit configs/train_config.yaml gage_configs list.
# 3. Edit scripts/build_data_tarball.py TRAIN_GAGES list.
# 4. Rebuild + upload:
python scripts/build_data_tarball.py
python -c "from huggingface_hub import HfApi, login; import os; login(token=os.environ['HF_TOKEN']); HfApi().upload_file(path_or_fileobj='/tmp/hydro_data_build/data.tar.gz', path_in_repo='data.tar.gz', repo_id='chrimerss/hydro_cali_agent_example', repo_type='dataset')"

# 5. Note the new commit SHA from the upload, then bump it in
#    modal_app/images.py (search for `hf_dataset_commit=`) so Modal
#    rebuilds only the data layer.
```

### 5. SFT Training (legacy)

Only useful if you want to compare against a distilled-from-GPT-4o baseline:

```bash
python scripts/prepare_sft_data.py     # builds data/sft_train.jsonl (2,576 examples)
modal run modal_app/sft.py             # uses sft_image (TRL stack), pushes to chrimerss/Qwen-3-8B-hydro-distill
```

The SFT path predates the verl RL path and uses the original Qwen3-8B + LoRA + TRL stack.

### 6. Evaluate

`modal_app/eval.py` runs an SGLang-based multi-turn calibration loop. **Default behavior: it loops over every gage in `configs/gages/` that is NOT in `configs/train_config.yaml`** — i.e., the held-out test set. It accepts either a HuggingFace model id (baseline) or a verl checkpoint path (experiment):

```bash
# Baseline — stock HuggingFace model on all held-out gages
modal run modal_app/eval.py --model-id Qwen/Qwen3-4B-Instruct-2507

# Experiment — latest verl GRPO checkpoint
modal run modal_app/eval.py

# Experiment — pin to a specific step
modal run modal_app/eval.py --checkpoint-path /checkpoints/qwen3-4b-grpo-anchored/global_step_50

# Single-gage override
modal run modal_app/eval.py --gage-config configs/gages/02338660.yaml
```

Eval defaults to **greedy decoding** (`temperature=0.0`) for reproducibility. On `<tool_call>` parse failure the loop retries up to 3 times with a "re-emit valid JSON" hint before giving up.

Results are saved to the `hydrollm-results` Modal Volume in a per-step directory layout:

```
/results/
├── baseline/
│   └── Qwen3-4B-Instruct-2507/
│       ├── 02338660.json
│       ├── 01403060.json   # if shipped in data.tar.gz
│       └── _summary.json
└── experiment/
    └── global_step_50/
        ├── 02338660.json
        └── _summary.json
```

Pull a result locally with:

```bash
modal volume ls hydrollm-results experiment/global_step_50
modal volume get hydrollm-results experiment/global_step_50/02338660.json ./
modal volume get hydrollm-results experiment/global_step_50/_summary.json ./
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

### RL training (`configs/verl/qwen3_4b_grpo.yaml`)

Hydra overlay on top of verl's `ppo_trainer` defaults. Key knobs:

| Hydra key | Default | Description |
|-----------|---------|-------------|
| `algorithm.adv_estimator` | `grpo` | GRPO (group-relative advantages) |
| `data.train_batch_size` | 4 | Prompts per step (×K=8 rollouts → 32 trajectories per step) |
| `actor_rollout_ref.rollout.n` | 6 | K rollouts per prompt (GRPO group size) |
| `actor_rollout_ref.rollout.temperature` | 1.0 | Sampling temperature |
| `actor_rollout_ref.rollout.top_p` | 0.95 | Nucleus sampling top-p |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | 50 | Max calibration rounds per rollout |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | Actor learning rate (5e-6 caused overshoot/collapse) |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | 0.05 | LR warmup over first 5% of steps |
| `actor_rollout_ref.actor.entropy_coeff` | 0.01 | Light entropy bonus — base distribution is already multi-modal |
| `actor_rollout_ref.actor.kl_loss_coef` | 0.2 | Strong anchor to base policy — prevents catastrophic forgetting / token-level degeneration |
| `data.max_response_length` | 4096 | Max tokens per assistant turn |
| `trainer.total_epochs` | 30 | Training epochs |
| `trainer.save_freq` | 10 | Checkpoint cadence (steps) — first save lands ~5h in |
| `trainer.test_freq` | 25 | Validation cadence (steps) |
| `trainer.n_gpus_per_node` | 4 | GPUs (single Modal container) |

### Gages — training vs testing

Selected by `scripts/audit_flood_windows.py`, which slides a 60-day window over each gage's observation series and scores by `log10(peak/median + 1) × sqrt(rise_h × rec_h)`. Each window is a clear flood event (rising + receding limbs, edge-buffered). Two of the originally selected gages (`06279500` at 40792 km², `07144100` at 3209 km²) were swapped out after EF5 timeouts under K=8 multi-turn rollouts; replaced with smaller-basin candidates from the audit drop-pool.

**Training set (10 gages — `configs/train_config.yaml`)**:

| Gage ID | Basin (km²) | Lat | Lon | Window (UTC) |
|---|---:|---:|---:|---|
| 11383500 |  539 | 40.0140 | -121.9483 | 2018-05-19 → 2018-07-17 |
| 11043000 |  575 | 33.4798 | -117.1439 | 2019-03-15 → 2019-05-13 |
| 11152000 |  632 | 36.2805 | -121.3227 | 2018-05-29 → 2018-07-27 |
| 02294781 | 1064 | 27.8245 |  -81.8017 | 2018-04-29 → 2018-06-27 |
| 02312000 | 1476 | 28.4800 |  -82.1776 | 2018-11-15 → 2019-01-13 |
| 07195430 | 1489 | 36.1086 |  -94.5333 | 2018-01-04 → 2018-03-04 |
| 11179000 | 1639 | 37.5871 | -121.9608 | 2018-06-03 → 2018-08-01 |
| 14301000 | 1727 | 45.7040 | -123.7554 | 2018-09-11 → 2018-11-09 |
| 14207500 | 1828 | 45.3507 | -122.6762 | 2018-04-09 → 2018-06-07 |
| 11376000 | 2401 | 40.3871 | -122.2386 | 2018-09-21 → 2018-11-19 |

**Testing set** (held out — used only by `modal_app/eval.py`):

| Gage ID | Basin (km²) | Lat | Lon | Window (UTC) |
|---|---:|---:|---:|---|
| 02338660 |   329 | 33.2357 |  -84.9876 | 2018-07-01 → 2018-08-31 |
| 01403060 |  2033 | 40.5511 |  -74.5483 | 2018-11-11 → 2019-01-09 |
| 06279500 | 40792 | 44.7585 | -108.1816 | 2018-06-13 → 2018-08-11 |
| 07144100 |  3209 | 37.8831 |  -97.4245 | 2019-03-30 → 2019-05-28 |

`modal_app/eval.py` defaults to evaluating every gage in `configs/gages/` that is **not** in `train_config.gage_configs`. Currently only `02338660` has data shipped in the image — the others fail at EF5 (no MRMS clip) until you regenerate `data.tar.gz` to include them.

### Adding New Gages

1. Create a YAML file in `configs/gages/`:
   ```yaml
   gage_id: "01632000"
   lon: -78.1234
   lat: 38.5678
   basin_area: 500.0
   obs_dir: /app/data/gauge
   control_template: /app/data/docs/control.txt
   time_begin: "201807010000"
   time_end: "201808312300"
   target_nse: 0.7
   ef5_timeout: 300
   ```

2. Add the gage_id to `scripts/build_data_tarball.py:TRAIN_GAGES` and re-run that script + the upload + bump the SHA in `modal_app/images.py` (see "Updating the data tarball" above).

3. Add the config path to `configs/train_config.yaml`:
   ```yaml
   gage_configs:
     - configs/gages/<gage_id>.yaml
   ```

## Reward Function

Two layers of signal: **per-turn** (returned by tools) and **terminal** (returned by `verl_reward.compute_score`):

**Per-turn (in `src/hydrollm/verl_tools.py`):**

| Tool call | Reward | Purpose |
|-----------|--------|---------|
| `set_parameters` (valid) | `+0.02` | Format/protocol bonus |
| `run_simulation` (valid) | `+0.05` | Slightly higher: produces the artifact to evaluate |
| `evaluate` (valid) | `ΔNSE` | Real progress signal (this turn's NSE − previous) |
| Any tool (invalid) | `−0.5` | Format penalty |

**Terminal (in `src/hydrollm/verl_reward.py`):**

| Component | Value | Purpose |
|-----------|-------|---------|
| Best NSE | `[−1, 1]` | Primary signal, clipped |
| Target bonus | `+0.5` | If best NSE > gage's `target_nse` |
| Per-evaluate **bonus** | `+0.02 × n_evaluates` | Encourages sustained iteration (replaces the earlier `-0.02` efficiency penalty, which had taught the agent to exit after one round) |
| Improvement bonus | `+0.10 × max(0, n_improvements − 1)` | Each evaluate that beat the running best — explicitly trains "iterate until you can't improve" |
| Empty-trajectory penalty | `−1.0` | Agent never produced a parseable evaluate result |

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
- [x] **Phase 3**: Scale RL to 10 CONUS gages (current — small/medium basins on 4×H100)
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

## Acknowledgement

We appreciate **Modal** for sponsoring computing credits for this research.

## License

MIT
