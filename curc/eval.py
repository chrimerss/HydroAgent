"""Multi-turn SGLang calibration eval on a local GPU (no Modal).

Runs the session harness in src/hydrollm/session.py (the same one
modal_app/eval.py and wsp/fwd.py use) on each gage and writes one JSON per
gage, its full transcript (.transcript.jsonl.gz) and `_summary.json`. Used by
`curc/eval.sbatch`; can also be run directly on any GPU node:

    python curc/eval.py --model /path/to/Qwen3-4B-Instruct-2507 \
        --gages 02338660,01403060 --out-dir results

`--harness v2` (default) is the fixed harness, `--harness v1` the original
one (see hydrollm.session). Decoding is greedy by default; pass
`--temperature 0.7 --top-p 0.8 --top-k 20 --seed N` for a sampled session.

Gages default to the held-out test set listed in the README. Gages whose
MRMS forcing is missing from $HYDROLLM_DATA_ROOT are skipped with a warning,
since EF5 cannot run them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hydrollm.config import DATA_ROOT, load_gage_config  # noqa: E402
from hydrollm.session import HARNESSES, run_session  # noqa: E402

logger = logging.getLogger("hydrollm.eval")

# Held-out test gages (README, "Testing set").
DEFAULT_TEST_GAGES = ["02338660", "01403060", "06279500", "07144100"]


def _gage_yaml(spec: str) -> Path:
    """Accept a gage id ('02338660') or a YAML path."""
    p = Path(spec)
    if p.suffix in (".yaml", ".yml"):
        return p if p.is_absolute() else REPO / p
    return REPO / "configs" / "gages" / f"{spec}.yaml"


def sampling_params(args) -> dict:
    sp = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_tokens,
        "stop": ["<|im_end|>"],
    }
    if getattr(args, "top_p", None) is not None:
        sp["top_p"] = args.top_p
    if getattr(args, "top_k", None) is not None:
        sp["top_k"] = args.top_k
    return sp


def calibrate_gage(generate, tokenizer, gcfg, args, transcript: list | None = None) -> dict:
    """Run one calibration session (hydrollm.session.run_session) and label it.

    `generate(prompt) -> completion text` wraps the inference backend (the
    in-process SGLang engine here, an SGLang server in wsp/fwd.py).
    """
    record = run_session(
        generate, tokenizer, gcfg,
        harness=getattr(args, "harness", "v2"),
        max_turns=args.max_turns, max_tokens=args.max_tokens,
        context_len=args.context_len, parse_retry_limit=args.parse_retry_limit,
        transcript=transcript,
    )
    return {
        "run_kind": args.run_kind,
        "run_tag": args.run_tag,
        "model_path": args.model,
        "temperature": args.temperature,
        "top_p": getattr(args, "top_p", None),
        "top_k": getattr(args, "top_k", None),
        "seed": getattr(args, "seed", None),
        **record,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="Local HF model dir (or HF repo id) for SGLang to load.")
    ap.add_argument("--gages", default=",".join(DEFAULT_TEST_GAGES),
                    help="Comma-separated gage ids or YAML paths.")
    ap.add_argument("--out-dir", default=str(REPO / "results"))
    ap.add_argument("--run-kind", default="baseline")
    ap.add_argument("--run-tag", default=None, help="Defaults to the model dir name.")
    ap.add_argument("--harness", default="v2", choices=HARNESSES)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="SGLang random seed (sampled runs).")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--context-len", type=int, default=8192,
                    help="Prompt+completion budget; older turns are dropped to fit.")
    ap.add_argument("--parse-retry-limit", type=int, default=3)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    args = ap.parse_args()
    args.run_tag = args.run_tag or Path(args.model.rstrip("/")).name

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    gages = []
    for spec in [s.strip() for s in args.gages.split(",") if s.strip()]:
        gcfg = load_gage_config(_gage_yaml(spec))
        mrms = Path(DATA_ROOT) / "data_mrms_clip" / gcfg.gage_id
        if not mrms.is_dir():
            logger.warning("Skipping %s: no MRMS forcing at %s", gcfg.gage_id, mrms)
            continue
        gages.append(gcfg)
    if not gages:
        logger.error("No runnable gages (data root %s).", DATA_ROOT)
        return 1
    logger.info("Evaluating %d gage(s): %s  harness=%s", len(gages),
                [g.gage_id for g in gages], args.harness)

    out_dir = Path(args.out_dir) / args.run_kind / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, extra_special_tokens={},
    )
    logger.info("Starting SGLang engine on %s ...", args.model)
    engine_kwargs = {}
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    engine = sgl.Engine(
        model_path=args.model,
        tokenizer_path=args.model,
        trust_remote_code=True,
        tool_call_parser="hermes",
        mem_fraction_static=args.mem_fraction,
        log_level="warning",
        **engine_kwargs,
    )
    # sgl.Engine re-runs logging.basicConfig(level=WARNING, force=True), which
    # would silence the per-turn progress lines below.
    logger.setLevel(logging.INFO)
    sp = sampling_params(args)

    def generate(prompt: str) -> str:
        return engine.generate([prompt], sp)[0]["text"]

    summary = []
    try:
        for gcfg in gages:
            logger.info("=" * 60)
            logger.info("HydroAgent eval — gage %s  run=%s/%s  harness=%s  max_turns=%d",
                        gcfg.gage_id, args.run_kind, args.run_tag, args.harness, args.max_turns)
            logger.info("=" * 60)
            transcript: list = []
            result = calibrate_gage(generate, tokenizer, gcfg, args, transcript)
            out_path = out_dir / f"{gcfg.gage_id}.json"
            out_path.write_text(json.dumps(result, indent=2, default=str))
            with gzip.open(out_dir / f"{gcfg.gage_id}.transcript.jsonl.gz", "wt") as f:
                for t in transcript:
                    f.write(json.dumps(t) + "\n")
            summary.append({k: result[k] for k in (
                "gage_id", "best_nse", "final_nse", "num_simulation_runs",
                "elapsed_seconds", "target_met", "end_reason")} | {"path": str(out_path)})
            logger.info("[%s] best_nse=%s  sim_runs=%d  elapsed=%.1fs  end=%s",
                        gcfg.gage_id, result["best_nse"], result["num_simulation_runs"],
                        result["elapsed_seconds"], result["end_reason"])
            # Rewrite the summary after every gage so a timeout keeps partial results.
            (out_dir / "_summary.json").write_text(json.dumps({
                "run_kind": args.run_kind, "run_tag": args.run_tag, "harness": args.harness,
                "model_path": args.model, "n_gages": len(summary), "results": summary,
            }, indent=2, default=str))
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass

    logger.info("Eval done. %d gage(s) -> %s", len(summary), out_dir)
    for r in summary:
        logger.info("  %s  best_nse=%s  sim_runs=%d",
                    r["gage_id"], r["best_nse"], r["num_simulation_runs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
