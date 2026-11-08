"""The editor's weight API (run via wsp/py.sh, which does `from wsp.kernel import *`).

Mirrors the paper's kernel: NAMES, load(name), load_original(name),
save_edit(name, tensor), drop_edit(name), edited(), tok, plus CONFIG and
TRIAL / WORK (this trial's read-write folder). Gradients are disabled and no
GPU is visible.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from wsp.common import BASE_MODEL, PROTECTED, base_weight_map, edits_dir, list_edits, trial_dir

torch.set_grad_enabled(False)

TRIAL = os.environ["WSP_TRIAL"]
WORK = trial_dir(TRIAL) / "work"
WORK.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((BASE_MODEL / "config.json").read_text())
_WMAP = base_weight_map()
NAMES = sorted(_WMAP)
tok = Tokenizer.from_file(str(BASE_MODEL / "tokenizer.json"))


def load_original(name: str) -> torch.Tensor:
    """The unedited tensor, as float32."""
    with safe_open(str(BASE_MODEL / _WMAP[name]), framework="pt") as f:
        return f.get_tensor(name).float()


def load(name: str) -> torch.Tensor:
    """Your current edit of `name` if there is one, else the original (float32)."""
    path = list_edits(TRIAL).get(name)
    if path is not None:
        return load_file(str(path))["tensor"].float()
    return load_original(name)


def edited() -> list[str]:
    return list(list_edits(TRIAL))


def drop_edit(name: str) -> None:
    path = list_edits(TRIAL).get(name)
    if path is not None:
        path.unlink()
        print(f"dropped edit of {name}")


def save_edit(name: str, tensor) -> dict:
    """Replace `name` with `tensor` in your edited model (stored float32,
    evaluated in bfloat16)."""
    if name not in _WMAP:
        raise KeyError(f"unknown tensor {name!r}")
    if name in PROTECTED:
        raise PermissionError(f"{name} is out of scope (embeddings / output layer / final norm)")
    t = torch.as_tensor(tensor).detach().float().cpu().contiguous()
    orig = load_original(name)
    if tuple(t.shape) != tuple(orig.shape):
        raise ValueError(f"shape {tuple(t.shape)} != original {tuple(orig.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("tensor has NaN or Inf")
    # Refuse edits that only multiply the whole tensor by a constant.
    denom = float((orig * orig).sum())
    if denom > 0:
        c = float((t * orig).sum()) / denom
        resid = float((t - c * orig).norm())
        if abs(c - 1.0) > 1e-6 and resid <= 1e-5 * float(t.norm()):
            raise PermissionError(
                "edits that only multiply a whole tensor by a constant are out of scope")
    n_changed = int((t.to(torch.bfloat16) != orig.to(torch.bfloat16)).sum())
    if n_changed == 0:
        drop_edit(name)
        print(f"{name}: identical to the original after bfloat16 rounding; no edit kept")
        return {"name": name, "changed_entries_bf16": 0}
    d = edits_dir(TRIAL)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{name}.tmp"
    save_file({"tensor": t}, str(tmp))
    tmp.replace(d / f"{name}.safetensors")
    info = {
        "name": name,
        "changed_entries_bf16": n_changed,
        "rel_frobenius_change": float((t - orig).norm() / orig.norm()),
    }
    print(f"saved edit: {info}")
    return info


__all__ = [
    "np", "torch", "json", "Path", "TRIAL", "WORK", "CONFIG", "NAMES", "tok",
    "load", "load_original", "save_edit", "drop_edit", "edited", "BASE_MODEL",
]
