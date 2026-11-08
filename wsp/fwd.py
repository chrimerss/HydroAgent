"""Run one forward pass for a WSP trial (inside the trial's GPU allocation).

    python wsp/fwd.py --trial trial_1 --k 3          # request in <trial>/fwd_3/request.json

Two pass types, as in the paper's Experiment 8 harness:

GENERATION — calibration rollouts on training gages with the evaluation's
protocol (curc/eval.py:calibrate_gage; greedy, 2,048 new tokens per turn,
8,192-token context, up to 100 turns):
    {"type": "generate", "model": "original" | "edited" | "both",
     "gages": [<training gage ids>] (default: all 10), "n": samples per gage
     (n > 1 needs temperature > 0), "temperature": 0.0, "max_turns": <= 100}
  At most 30 rollouts per model. Rollouts run concurrently on one SGLang
  server per model; "both" runs original then edited on the same items.

TEACHER-FORCED — the model reads up to 1,024 items (no generation):
    {"type": "teacher_forced", "model": "original" | "edited",
     "items": [ "<raw text>" | {"prompt": str, "output": str}
                | {"ref": {"source": "baseline" | "fwd_<k>",
                           "model": "original" | "edited", "gage": id,
                           "sample": 0, "turn": t}} , ...],
     "return": {"positions": "output" | "all" | "last_prompt" | "output_mean",
                "token_logprobs": true, "topk": k <= 20,
                "hidden_states": [0..36], "mlp_act": [0..35]}}
  Position p's row describes the model's state at token p and its
  prediction of token p+1. "output" = positions predicting each output token
  (and the closing <|im_end|>); "last_prompt" = the position that predicts
  the first output token. Arrays are saved to <trial>/fwd_<k>/.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from wsp.common import BASE_MODEL, TRAIN_GAGES, build_ckpt, list_edits, locked_state, trial_dir  # noqa: E402

_spec = importlib.util.spec_from_file_location("hydro_eval", REPO / "curc" / "eval.py")
hydro_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hydro_eval)

from hydrollm.config import load_gage_config  # noqa: E402

logger = logging.getLogger("wsp.fwd")

MAX_ROLLOUTS_PER_MODEL = 30
MAX_TF_ITEMS = 1024
MAX_TF_TOKENS_PER_ITEM = 8192
MAX_TF_TOKENS = 524_288
MAX_TF_BYTES = 6 * 1024**3
IM_END = 151645


class RequestError(ValueError):
    """Bad request: reported back to the editor, not charged to the budget."""


def _model_path(trial: str, which: str) -> Path:
    if which == "original":
        return BASE_MODEL
    if which == "edited":
        if not list_edits(trial):
            raise RequestError("model 'edited' requested but you have no edits saved")
        return build_ckpt(trial)
    raise RequestError(f"model must be original/edited/both, got {which!r}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def sglang_server(model_path: Path, log_path: Path, extra_args: list[str] | None = None):
    port = _free_port()
    cmd = [sys.executable, "-m", "sglang.launch_server",
           "--model-path", str(model_path), "--tokenizer-path", str(model_path),
           "--host", "127.0.0.1", "--port", str(port),
           "--mem-fraction-static", "0.80",
           "--trust-remote-code", "--log-level", "warning", "--cuda-graph-max-bs", "32"]
    cmd += extra_args or []
    log = open(log_path, "ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"SGLang server exited ({proc.returncode}); see {log_path}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t0 > 900:
                raise RuntimeError("SGLang server did not become healthy in 900 s")
            time.sleep(3)
        logger.info("SGLang up on %s (%s) in %.0fs", url, model_path, time.time() - t0)
        yield url
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        log.close()


def _http_generate(url: str, sp: dict):
    def generate(prompt: str) -> str:
        body = json.dumps({"text": prompt, "sampling_params": sp}).encode()
        err = None
        for _ in range(3):
            try:
                req = urllib.request.Request(url + "/generate", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=900) as r:
                    return json.loads(r.read())["text"]
            except Exception as e:  # transient server hiccup
                err = e
                time.sleep(5)
        raise err
    return generate


def _rollout_summary(gage: str, sample: int, res: dict, transcript: list) -> dict:
    hist = res["nse_history"]
    return {
        "gage": gage, "sample": sample,
        "best_nse": res["best_nse"],
        "final_nse": None if res["final_nse"] is None else round(res["final_nse"], 4),
        "n_evaluates": len(hist),
        "n_failed_evaluates": sum(1 for x in hist if x <= -998),
        "n_sims": res["num_simulation_runs"],
        "parse_failure_turns": sum(1 for t in transcript if not t["parsed"]),
        "invalid_tool_calls": res["invalid_tool_calls"],
        "turns": res["num_turns"],
        "elapsed_s": res["elapsed_seconds"],
    }


def _run_rollouts(model_path: Path, items: list[tuple[str, int]], temperature: float,
                  max_turns: int, out: Path) -> list[dict]:
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True,
                                              extra_special_tokens={})
    # The editors' sessions in the original experiment ran on the original
    # harness (hydrollm.session "v1"); set WSP_HARNESS=v2 for the fixed one.
    ns = types.SimpleNamespace(max_turns=max_turns, temperature=temperature, max_tokens=2048,
                               context_len=8192, parse_retry_limit=3, run_kind="wsp",
                               run_tag=out.name, model=str(model_path),
                               harness=os.environ.get("WSP_HARNESS", "v1"))
    with sglang_server(model_path, out / "sglang.log") as url:
        generate = _http_generate(url, hydro_eval.sampling_params(ns))

        def one(item):
            gage, s = item
            gcfg = load_gage_config(REPO / "configs" / "gages" / f"{gage}.yaml")
            transcript: list = []
            res = hydro_eval.calibrate_gage(generate, tokenizer, gcfg, ns, transcript)
            stem = f"{gage}_s{s}"
            (out / f"{stem}.json").write_text(json.dumps(res, indent=2, default=str))
            with gzip.open(out / f"{stem}.transcript.jsonl.gz", "wt") as f:
                for t in transcript:
                    f.write(json.dumps(t) + "\n")
            return _rollout_summary(gage, s, res, transcript)

        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            return list(pool.map(one, items))


def _agg(rows: list[dict]) -> dict:
    best = [r["best_nse"] for r in rows if r["best_nse"] is not None and r["best_nse"] > -998]
    return {
        "rollouts": len(rows),
        "mean_best_nse": round(float(np.mean(best)), 4) if best else None,
        "median_best_nse": round(float(np.median(best)), 4) if best else None,
        "mean_best_nse_clipped_at_-1": round(float(np.mean(np.clip(best, -1, 1))), 4) if best else None,
        "rollouts_without_valid_nse": len(rows) - len(best),
        "mean_sims": round(float(np.mean([r["n_sims"] for r in rows])), 2),
        "mean_parse_failure_turns": round(float(np.mean([r["parse_failure_turns"] for r in rows])), 2),
        "mean_invalid_tool_calls": round(float(np.mean([r["invalid_tool_calls"] for r in rows])), 2),
    }


def run_generate(trial: str, req: dict, out: Path) -> dict:
    which = req.get("model", "both")
    models = ["original", "edited"] if which == "both" else [which]
    gages = [str(g) for g in (req.get("gages") or TRAIN_GAGES)]
    bad = [g for g in gages if g not in TRAIN_GAGES]
    if bad:
        raise RequestError(f"only training gages are available: {bad} not allowed")
    n = int(req.get("n", 1))
    temperature = float(req.get("temperature", 0.0))
    max_turns = int(req.get("max_turns", 100))
    if temperature == 0.0 and n != 1:
        raise RequestError("greedy decoding (temperature 0) is deterministic per gage: use n = 1")
    if not 1 <= max_turns <= 100:
        raise RequestError("max_turns must be in [1, 100]")
    items = [(g, s) for g in gages for s in range(n)]
    if not 1 <= len(items) <= MAX_ROLLOUTS_PER_MODEL:
        raise RequestError(f"gages x n must be 1..{MAX_ROLLOUTS_PER_MODEL} per model")
    paths = {m: _model_path(trial, m) for m in models}  # validate before spending time

    rows = {}
    for m in models:
        logger.info("generation: model=%s items=%d max_turns=%d T=%.2f", m, len(items), max_turns, temperature)
        rows[m] = _run_rollouts(paths[m], items, temperature, max_turns, out / m)
    summary = {"type": "generate", "model": which, "gages": gages, "n": n,
               "temperature": temperature, "max_turns": max_turns,
               "per_model": {m: _agg(r) for m, r in rows.items()},
               "rollouts": rows}
    if which == "both":
        paired = []
        for o, e in zip(rows["original"], rows["edited"]):
            ok = all(x["best_nse"] is not None and x["best_nse"] > -998 for x in (o, e))
            paired.append({"gage": o["gage"], "sample": o["sample"],
                           "original": o["best_nse"], "edited": e["best_nse"],
                           "diff": round(e["best_nse"] - o["best_nse"], 4) if ok else None})
        d = np.array([p["diff"] for p in paired if p["diff"] is not None], dtype=float)
        dc = np.array([np.clip(p["edited"], -1, 1) - np.clip(p["original"], -1, 1)
                       for p in paired if p["diff"] is not None], dtype=float)
        summary["paired_best_nse"] = {
            "n_pairs": int(d.size),
            "mean_diff_clipped_at_-1": round(float(dc.mean()), 4) if dc.size else None,
            "se_clipped": round(float(dc.std(ddof=1) / np.sqrt(dc.size)), 4) if dc.size > 1 else None,
            "mean_diff_unclipped": round(float(d.mean()), 4) if d.size else None,
            "se_unclipped": round(float(d.std(ddof=1) / np.sqrt(d.size)), 4) if d.size > 1 else None,
            "n_edited_better": int((d > 0).sum()), "n_edited_worse": int((d < 0).sum()),
            "pairs": paired,
        }
    return summary


# ---------------------------------------------------------------------------
# Teacher-forced
# ---------------------------------------------------------------------------

def _resolve_ref(trial: str, ref: dict) -> dict:
    src = ref.get("source", "baseline")
    if src == "baseline":
        base = trial_dir("shared") / "baseline_train" / "original"
    elif src.startswith("fwd_") and src[4:].isdigit():
        base = trial_dir(trial) / src / ref.get("model", "original")
    else:
        raise RequestError(f"bad ref source {src!r}")
    path = base / f"{ref['gage']}_s{int(ref.get('sample', 0))}.transcript.jsonl.gz"
    if not path.exists():
        raise RequestError(f"no transcript at {path}")
    with gzip.open(path, "rt") as f:
        turns = [json.loads(line) for line in f]
    t = int(ref["turn"])
    for row in turns:
        if row["turn"] == t:
            return {"prompt": row["prompt"], "output": row["output"]}
    raise RequestError(f"turn {t} not in {path}")


def run_teacher_forced(trial: str, req: dict, out: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    which = req.get("model", "original")
    if which not in ("original", "edited"):
        raise RequestError("teacher_forced model must be original or edited")
    items = req.get("items") or []
    if not 1 <= len(items) <= MAX_TF_ITEMS:
        raise RequestError(f"items must have 1..{MAX_TF_ITEMS} entries")
    ret = req.get("return", {})
    positions = ret.get("positions", "output")
    if positions not in ("output", "all", "last_prompt", "output_mean"):
        raise RequestError(f"bad positions {positions!r}")
    topk = int(ret.get("topk", 0))
    if not 0 <= topk <= 20:
        raise RequestError("topk must be 0..20")
    want_lp = bool(ret.get("token_logprobs", True))
    hs_layers = sorted({int(x) for x in ret.get("hidden_states", [])})
    mlp_layers = sorted({int(x) for x in ret.get("mlp_act", [])})
    if any(not 0 <= x <= 36 for x in hs_layers) or any(not 0 <= x <= 35 for x in mlp_layers):
        raise RequestError("hidden_states layers must be 0..36, mlp_act layers 0..35")

    path = _model_path(trial, which)
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, extra_special_tokens={})

    # Tokenize: prompt and output separately (as the model saw them), then <|im_end|>.
    seqs = []
    for it in items:
        if isinstance(it, str):
            ids = tokenizer.encode(it, add_special_tokens=False)
            seqs.append({"ids": ids, "prompt_len": 0})
            continue
        if "ref" in it:
            it = _resolve_ref(trial, it["ref"])
        p = tokenizer.encode(it["prompt"], add_special_tokens=False)
        o = tokenizer.encode(it["output"], add_special_tokens=False)
        seqs.append({"ids": p + o + [IM_END], "prompt_len": len(p)})
    lens = [len(s["ids"]) for s in seqs]
    if max(lens) > MAX_TF_TOKENS_PER_ITEM or sum(lens) > MAX_TF_TOKENS:
        raise RequestError(f"token limits: <= {MAX_TF_TOKENS_PER_ITEM} per item and <= {MAX_TF_TOKENS} total "
                           f"(got max {max(lens)}, total {sum(lens)})")

    def select(s):
        L, pl = len(s["ids"]), s["prompt_len"]
        if positions == "all":
            return list(range(L))
        if positions == "last_prompt":
            return [max(pl - 1, 0)]
        return list(range(max(pl - 1, 0), L - 1))  # output / output_mean

    sel = [select(s) for s in seqs]
    rows = sum(1 if positions in ("last_prompt", "output_mean") else len(x) for x in sel)
    est = rows * 2 * (2560 * len(hs_layers) + 9728 * len(mlp_layers)) + rows * (8 + 6 * topk)
    if est > MAX_TF_BYTES:
        raise RequestError(f"estimated output {est / 1e9:.1f} GB > 6 GB: fewer items, layers or positions")

    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    captured: dict = {}
    hooks = []
    layers = model.model.layers
    for l in hs_layers:
        mod = model.model.embed_tokens if l == 0 else layers[l - 1]
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, l=l: captured.__setitem__(("hs", l), (o[0] if isinstance(o, tuple) else o)[0])))
    for l in mlp_layers:
        hooks.append(layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda m, i, l=l: captured.__setitem__(("mlp", l), i[0][0])))

    acc = {k: [] for k in ["targets", "logprobs", "topk_ids", "topk_logprobs"]}
    acc.update({("hs", l): [] for l in hs_layers})
    acc.update({("mlp", l): [] for l in mlp_layers})
    index = []
    offset = 0
    t0 = time.time()
    with torch.no_grad():
        for s, pos in zip(seqs, sel):
            captured.clear()
            ids = torch.tensor([s["ids"]], device="cuda")
            h = model.model(input_ids=ids).last_hidden_state[0]
            pos_t = torch.tensor(pos, device="cuda")
            L = len(s["ids"])
            tgt = [s["ids"][p + 1] if p + 1 < L else -1 for p in pos]
            if want_lp or topk:
                lp = torch.log_softmax(model.lm_head(h[pos_t]).float(), dim=-1)
                if want_lp:
                    t = torch.tensor([max(x, 0) for x in tgt], device="cuda")
                    v = lp.gather(1, t[:, None])[:, 0]
                    v = torch.where(torch.tensor(tgt, device="cuda") >= 0, v, torch.full_like(v, float("nan")))
                    v = v.cpu().numpy().astype(np.float32)
                    acc["logprobs"].append(np.array([np.nanmean(v)], dtype=np.float32)
                                           if positions == "output_mean" else v)
                if topk:
                    tv, ti = lp.topk(topk, dim=-1)
                    if positions == "output_mean":
                        tv, ti = tv[:1], ti[:1]
                    acc["topk_ids"].append(ti.cpu().numpy().astype(np.int32))
                    acc["topk_logprobs"].append(tv.cpu().numpy().astype(np.float16))
            for key, val in captured.items():
                x = val[pos_t].float()
                if positions == "output_mean":
                    x = x.mean(0, keepdim=True)
                acc[key].append(x.cpu().numpy().astype(np.float16))
            n_rows = 1 if positions == "output_mean" else len(pos)
            acc["targets"].append(np.array(tgt[:n_rows] if positions != "output_mean" else [-1], dtype=np.int32))
            index.append({"offset": offset, "count": n_rows, "length": L, "prompt_len": s["prompt_len"]})
            offset += n_rows
    for hk in hooks:
        hk.remove()

    files = {}
    for key, chunks in acc.items():
        if not chunks:
            continue
        name = key if isinstance(key, str) else f"{key[0]}_l{key[1]}"
        np.save(out / f"{name}.npy", np.concatenate(chunks, axis=0))
        files[name] = f"{name}.npy"
    meta = {"model": which, "positions": positions, "items": len(seqs), "rows": offset,
            "tokens_read": int(sum(lens)), "files": files, "index": index,
            "conventions": "row = state at token p and its prediction of token p+1; "
                           "targets = token p+1 (-1 if none); hs_l0 = embeddings, hs_l{l} = after layer l "
                           "(1-based); mlp_l{l} = input to layers[l].mlp.down_proj (0-based)"}
    (out / "meta.json").write_text(json.dumps(meta))
    lp_all = np.concatenate(acc["logprobs"]) if acc["logprobs"] else np.array([])
    return {"type": "teacher_forced", "model": which, "items": len(seqs), "rows": offset,
            "tokens_read": int(sum(lens)), "seconds": round(time.time() - t0, 1),
            "mean_logprob": round(float(np.nanmean(lp_all)), 4) if lp_all.size else None,
            "files": files, "folder": str(out)}


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--k", required=True, help="pass index (folder fwd_<k>) or a folder name")
    ap.add_argument("--no-budget", action="store_true", help="do not charge the pass (experimenter use)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # EF5 subprocesses inherit this

    out = trial_dir(args.trial) / (f"fwd_{args.k}" if str(args.k).isdigit() else str(args.k))
    req = json.loads((out / "request.json").read_text())
    try:
        kind = req.get("type")
        if kind == "generate":
            summary = run_generate(args.trial, req, out)
        elif kind == "teacher_forced":
            summary = run_teacher_forced(args.trial, req, out)
        else:
            raise RequestError("type must be 'generate' or 'teacher_forced'")
    except RequestError as e:
        (out / "failed.json").write_text(json.dumps({"error": str(e), "charged": False}))
        logger.error("request refused: %s", e)
        return 2
    except Exception as e:
        (out / "failed.json").write_text(json.dumps(
            {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:],
             "charged": False}))
        logger.exception("pass failed")
        return 1
    if not args.no_budget:
        with locked_state(args.trial) as st:
            st["passes_used"] += 1
            summary["passes_used"] = st["passes_used"]
            summary["passes_left"] = st["budget"]["passes"] - st["passes_used"]
    (out / "done.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("pass done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
