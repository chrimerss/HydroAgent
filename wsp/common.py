"""Shared paths, rules and checkpoint building for the weight-space-programming
(WSP) experiment, adapted from "Sparks of Weight-Space Programming" (Takehi,
Teoh, Zhang, Khattab; MIT CSAIL, Nov 2026), Experiment 8.

Layout on scratch (override the root with WSP_ROOT):

    $WSP_ROOT/shared/baseline_train/   original model on the 10 training gages
    $WSP_ROOT/<trial>/edited/          the editor's current edit, one
                                       <tensor name>.safetensors (float32) per tensor
    $WSP_ROOT/<trial>/ckpt/<hash>/     full HF checkpoint built from that edit
    $WSP_ROOT/<trial>/fwd_<k>/         results of forward pass k
    $WSP_ROOT/<trial>/state.json       budget accounting
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_USER = os.environ.get("USER") or os.popen("id -un").read().strip()
HYDRO_ROOT = Path(os.environ.get("HYDRO_ROOT", f"/scratch/alpine/{_USER}/hydroagent"))
BASE_MODEL = Path(os.environ.get("WSP_BASE_MODEL", HYDRO_ROOT / "models" / "Qwen3-4B-Instruct-2507"))
WSP_ROOT = Path(os.environ.get("WSP_ROOT", HYDRO_ROOT / "wsp_runs"))

# Training gages (README, "Training set"). The held-out test gages are never
# accepted by the forward-pass tool.
TRAIN_GAGES = [
    "11383500", "11043000", "11152000", "02294781", "02312000",
    "07195430", "11179000", "14301000", "14207500", "11376000",
]

# Out of scope for edits (as in the paper): embeddings (tied with the output
# layer in Qwen3-4B) and the final norm.
PROTECTED = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}

DEFAULT_BUDGET = {"passes": 25, "code_runs": 300}


def base_weight_map() -> dict[str, str]:
    return json.loads((BASE_MODEL / "model.safetensors.index.json").read_text())["weight_map"]


def trial_dir(trial: str) -> Path:
    if not trial or "/" in trial or trial.startswith("."):
        raise ValueError(f"bad trial name {trial!r}")
    return WSP_ROOT / trial


def edits_dir(trial: str) -> Path:
    return trial_dir(trial) / "edited"


def list_edits(trial: str) -> dict[str, Path]:
    d = edits_dir(trial)
    if not d.is_dir():
        return {}
    return {p.name[: -len(".safetensors")]: p for p in sorted(d.glob("*.safetensors"))}


def edit_hash(trial: str) -> str:
    h = hashlib.sha256()
    for name, path in list_edits(trial).items():
        h.update(name.encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# State / budget
# ---------------------------------------------------------------------------

@contextmanager
def locked_state(trial: str):
    """Read-modify-write trial/state.json under an exclusive file lock."""
    td = trial_dir(trial)
    td.mkdir(parents=True, exist_ok=True)
    lock = open(td / ".state.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        path = td / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {
            "trial": trial, "budget": dict(DEFAULT_BUDGET),
            "passes_used": 0, "code_runs_used": 0, "alloc_job": None,
        }
        yield state
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(path)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def read_state(trial: str) -> dict:
    path = trial_dir(trial) / "state.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------------
# Checkpoint building
# ---------------------------------------------------------------------------

def build_ckpt(trial: str, edits: dict[str, Path] | None = None, out: Path | None = None) -> Path:
    """Materialize base weights + edits as a loadable HF checkpoint directory.

    Edited tensors are cast to bfloat16 (the model's dtype, as in the paper's
    evaluation); shards without edits and all non-weight files are symlinked.
    Cached by edit hash, so rebuilding an unchanged edit is free.
    """
    import torch
    from safetensors.torch import load_file, save_file

    edits = list_edits(trial) if edits is None else edits
    if not edits:
        raise ValueError(f"trial {trial} has no edits")
    if out is None:
        out = trial_dir(trial) / "ckpt" / edit_hash(trial)
    if (out / ".complete").exists():
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    wmap = base_weight_map()
    by_shard: dict[str, list[str]] = {}
    for name in edits:
        by_shard.setdefault(wmap[name], []).append(name)

    manifest = {"base_model": str(BASE_MODEL), "tensors": {}}
    for f in BASE_MODEL.iterdir():
        if f.name.endswith(".safetensors") and f.name in by_shard:
            tensors = load_file(str(f))
            for name in by_shard[f.name]:
                new = load_file(str(edits[name]))["tensor"].to(torch.bfloat16)
                orig = tensors[name]
                assert new.shape == orig.shape, (name, new.shape, orig.shape)
                manifest["tensors"][name] = {
                    "shape": list(orig.shape),
                    "changed_entries_bf16": int((new != orig).sum()),
                    "rel_frobenius_change": float(
                        (new.float() - orig.float()).norm() / orig.float().norm()),
                }
                tensors[name] = new.contiguous()
            save_file(tensors, str(out / f.name), metadata={"format": "pt"})
        else:
            (out / f.name).symlink_to(f.resolve())
    (out / "edit_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / ".complete").touch()
    return out
