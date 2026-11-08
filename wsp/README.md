# Weight-space programming on HydroAgent

A replication of Experiment 8 of "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab; MIT CSAIL, November 2026; [post](https://rikiyat.github.io/blog/weight-space-programming/), [logs](https://github.com/RikiyaT/lm_understand-logs)) on HydroAgent's calibration agent, `Qwen/Qwen3-4B-Instruct-2507`. In the paper a frontier LLM reads a smaller model's weights and edits a few of them, with no gradient descent, to repair a failure; on Qwen3.5-0.8B, removing a repetition loop raised MATH-500 from 7.9% to 48.1%.

**Result: the method worked as a repair but did not make the model a better calibrator.** Five independent editors found the same failure and fixed it with small edits that beat matched random edits, but none beat the unedited model on the held-out gages, and on the fixed evaluation harness all five were worse than no edit.

## Protocol

Adapted from the paper's Experiment 8 harness (`TASK.md` is the editor's task message, adapted from theirs):

- **Editor:** one fresh Claude agent per trial, with no context shared between trials (the paper used GPT-6 Astra).
- **Practice data:** the 10 training gages only, plus the unedited model's full sessions on them. The 4 held-out gages are never shown to the editor.
- **Tools:** a CPU Python kernel over the weights (`kernel.py`: `load`, `load_original`, `save_edit`; edits to the embeddings/output layer, the final norm and whole-tensor scaling are refused), and a forward-pass tool (`fwd.py`) with 25 passes: generation (calibration sessions on training gages, original/edited/both paired) or teacher-forced reads (log-probs, top-k, residual states, MLP activations).
- **Rules:** no gradients or fitting loops, no running the model outside the tool, no internet, an explanation logged before every step.
- **Held-out evaluation:** each submitted edit is frozen and cast to bfloat16 (`finalize.py`), along with a matched random control: the same tensors and changed entries, with Gaussian noise of the same Frobenius norm in place of the edit.
- **Primary metric (fixed before any held-out run):** mean over the 4 held-out gages of best NSE clipped to [-1, 1], as in `src/hydrollm/verl_reward.py`.

## Results

### What the editors changed

All five editors independently identified the same failure: once the 8k context dropped the user's task message (around turn 25-30), the model declared "calibration is complete", stopped calling tools, and the retry hint locked it into resending its last call.

| Trial | Edit | Tensors | Changed weights |
|---|---|---|---:|
| 1 | gated steering from "concluding" to "planning" sentences, plus a nudge toward a tool call at turn ends | `down_proj` of layers 23, 32, 34 (22 columns) | 25,582 |
| 2 | three repurposed neurons that cancel a "task accomplished" direction above a threshold | `gate/up/down_proj` of layers 21, 23, 25 | 23,028 |
| 3 | gated steering away from the stop direction, plus recovery after clamped values | `down_proj` of layers 20, 24 (7 columns) | 17,920 |
| 4 | two rank-1 additions: anti-stop and anti-"complaint" feedback | `down_proj` of layers 18, 24 | ~42 M (dense) |
| 5 | a bias-like push away from "end the turn without a tool call" | `down_proj` of layer 24 (4 columns) | 10,239 |

### Round 1: original harness, greedy decoding (`analyze.py`)

| Model | Clipped mean | vs unedited | Random control vs unedited | Edit vs control |
|---|---:|---:|---:|---:|
| Unedited | 0.163 | | | |
| Trial 1 | 0.196 | +0.033 | -0.068 | +0.100 |
| Trial 2 | 0.073 | -0.090 | -0.137 | +0.047 |
| Trial 3 | -0.019 | -0.182 | -0.268 | +0.086 |
| Trial 4 | 0.236 | +0.073 | -0.095 | +0.168 |
| Trial 5 | 0.256 | +0.093 | -0.199 | +0.292 |
| **Across trials** | | **-0.015** (95% CI -0.16 to +0.13, p = 0.80) | **-0.153** (p = 0.01) | **+0.139** (5/5, p = 0.03) |

The targeted repair carried over to the held-out gages in trials 1-4 (turns without a tool call fell from 21.8 to 0-12 per gage), but more simulations did not turn into better fits. Trial 5's held-out gain came from a shifted first parameter guess, not from the repair.

### Round 2: fixed harness (v2), sampled decoding, 5 seeds (`analyze_v2.py`)

The fixed harness (`src/hydrollm/session.py`) removes the stopping failure itself, so this round asks whether the edits add anything on top of it.

| Model | Clipped mean ± SE | vs unedited | p |
|---|---:|---:|---:|
| Unedited | 0.225 ± 0.036 | | |
| Trial 1 | 0.168 ± 0.025 | -0.056 | 0.24 |
| Trial 2 | 0.199 ± 0.049 | -0.026 | 0.68 |
| Trial 3 | 0.093 ± 0.037 | -0.131 | 0.03 |
| Trial 4 | 0.031 ± 0.065 | -0.194 | 0.04 |
| Trial 5 | 0.146 ± 0.043 | -0.079 | 0.20 |
| **All edits pooled** | | **-0.097** | 0.05 |

With the failure handled by the harness, only the edits' side effects remain (shifted starting guesses, reworded diagnoses, a perturbed parameter search). The conclusion holds without gage 06279500, where some edited runs had 0-3 EF5 timeouts and the unedited runs had none.

### Why it didn't help

- **Wrong bottleneck.** In the paper, the loop was the reason answers failed. Here, stopping early only costs late turns, while the model's parameter search is close to random (its moves go the right way 44-51% of the time), so extra iterations rarely set a new best. This matches the paper's Gemma follow-up: editing removes obstacles but does not add missing ability.
- **The failure was largely caused by the harness**, which is cheaper and safer to fix directly.
- **Noise.** Single greedy sessions are chaotic; editors selecting their best-measured candidate on 10 training gages partly selected luck (training-gage gains of +0.075 to +0.21 did not transfer).

## Caveats

- Claude editors with a 25-pass budget, not GPT-6 Astra with up to 50; the paper's large gain appeared only at 50.
- Trial 4 briefly saw a fragment of trial 3's log through a shared temporary folder (disclosed; it says its approach was already fixed).
- The random controls are a rough structural match for trial 3 (very large edit norm) and trial 4 (dense noise rather than a random rank-1 edit).
- Only 4 held-out gages; every model scores at or below -1 on 06279500, so the clipped metric rests mostly on 3.

## Running it

Commands run from the repo root on CURC Alpine (see `curc/` for setup); work happens in a Slurm allocation per trial, results go to `$HYDRO_ROOT/wsp_runs/` (scratch) and `results/` (not in git).

```bash
# Shared starting data: the unedited model on the 10 training gages
wsp/wsp alloc shared
#   write {"type": "generate", "model": "original", "max_turns": 100} to
#   $HYDRO_ROOT/wsp_runs/shared/baseline_train/request.json, then inside the allocation:
#   python wsp/fwd.py --trial shared --k baseline_train --no-budget

# One trial
wsp/wsp alloc trial_1          # a GPU slice for 24 h (--account/--partition or WSP_ACCOUNT/WSP_PARTITION)
wsp/wsp task trial_1           # renders TASK.md into $HYDRO_ROOT/wsp_runs/trial_1/message.txt
#   give message.txt to a fresh editor agent; it works through
#   wsp/wsp py / fwd / wait / status trial_1 ...

# Round 1 evaluation: freeze, build the random control, run both on the held-out gages
srun --jobid=<alloc> --overlap --ntasks=1 --cpus-per-task=16 --gres=gpu:h200_3g.71gb:1 \
    bash wsp/eval_trial.sh trial_1 1
python wsp/analyze.py

# Round 2 evaluation: fixed harness, sampled, 5 seeds per edit
MODEL=$HYDRO_ROOT/wsp_runs/trial_1/final_ckpt RUN_KIND=wsp_v2 RUN_TAG=trial_1_s1 \
    EXTRA_ARGS="--harness v2 --temperature 0.7 --top-p 0.8 --top-k 20 --seed 1" \
    sbatch --export=ALL --gres=gpu:h200_3g.71gb:1 curc/eval.sbatch
python wsp/analyze_v2.py       # the unedited baseline is results/sampled/v2_s<seed>
```

The editors' forward-pass sessions used the original harness; `fwd.py` keeps that (`WSP_HARNESS=v1`) unless `WSP_HARNESS=v2` is set.
