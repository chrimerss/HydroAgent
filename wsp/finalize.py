"""Freeze a finished trial's submission and build its matched random control.

    python wsp/finalize.py --trial trial_1 --seed 1

1. Copies <trial>/edited/ to <trial>/final_edited/ and builds the evaluated
   checkpoint at <trial>/final_ckpt/ (edited tensors in bfloat16).
2. Builds <trial>_control/final_ckpt/: the same tensors and the same changed
   entries, with the change replaced by Gaussian noise of the same Frobenius
   norm per tensor (a norm- and support-matched random edit).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.common import BASE_MODEL, base_weight_map, build_ckpt, list_edits, trial_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()

    td = trial_dir(a.trial)
    final = td / "final_edited"
    if final.exists():
        print(f"{final} already exists; refusing to overwrite a frozen submission")
        return 1
    edits = list_edits(a.trial)
    if not edits:
        print(f"{a.trial} submitted no edit")
        return 1
    shutil.copytree(td / "edited", final)
    frozen = {p.name[: -len(".safetensors")]: p for p in sorted(final.glob("*.safetensors"))}
    ckpt = build_ckpt(a.trial, edits=frozen, out=td / "final_ckpt")
    print(f"final checkpoint: {ckpt}")

    ctrl = trial_dir(f"{a.trial}_control")
    (ctrl / "final_edited").mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed)
    wmap = base_weight_map()
    report = {}
    for name, path in frozen.items():
        with safe_open(str(BASE_MODEL / wmap[name]), framework="pt") as f:
            orig = f.get_tensor(name)  # bfloat16
        new = load_file(str(path))["tensor"].to(torch.bfloat16)
        delta = new.float() - orig.float()
        mask = delta != 0
        noise = torch.randn(int(mask.sum()), generator=gen)
        noise *= delta.norm() / noise.norm()
        rand = orig.float().clone()
        rand[mask] += noise
        save_file({"tensor": rand.contiguous()}, str(ctrl / "final_edited" / f"{name}.safetensors"))
        report[name] = {"changed_entries": int(mask.sum()), "delta_norm": float(delta.norm())}
    cfrozen = {p.name[: -len(".safetensors")]: p for p in sorted((ctrl / "final_edited").glob("*.safetensors"))}
    cckpt = build_ckpt(f"{a.trial}_control", edits=cfrozen, out=ctrl / "final_ckpt")
    (ctrl / "control.json").write_text(json.dumps({"of": a.trial, "seed": a.seed, "tensors": report}, indent=2))
    print(f"control checkpoint: {cckpt}")
    for n in report:
        e = json.loads((ckpt / "edit_manifest.json").read_text())["tensors"][n]
        c = json.loads((cckpt / "edit_manifest.json").read_text())["tensors"][n]
        print(f"  {n}: edit changed={e['changed_entries_bf16']} rel={e['rel_frobenius_change']:.4g} | "
              f"control changed={c['changed_entries_bf16']} rel={c['rel_frobenius_change']:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
