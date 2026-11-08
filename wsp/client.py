"""Login-node commands for a WSP trial; heavy work runs in the trial's GPU
allocation via `srun --overlap`.

    wsp/wsp alloc  <trial> [--hours 24] [--qos gpu-normal] [--gres gpu:h200_3g.71gb:1]
                   [--account $WSP_ACCOUNT] [--partition $WSP_PARTITION]
    wsp/wsp task   <trial> [--stdout]                      # render TASK.md -> <trial>/message.txt
    wsp/wsp status <trial>
    wsp/wsp py     <trial> <script.py> "<explanation>"     # kernel code, CPU only, <= 900 s
    wsp/wsp fwd    <trial> <request.json> "<explanation>"  # one forward pass (waits <= 9 min)
    wsp/wsp wait   <trial> <k>                             # keep waiting for pass k (<= 9 min)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.common import (  # noqa: E402
    DEFAULT_BUDGET, REPO, TRAIN_GAGES, WSP_ROOT, list_edits, locked_state, read_state, trial_dir,
)

OUT_LIMIT = 10_000
WAIT_SECONDS = 540


def die(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def _job_state(job) -> str:
    if not job:
        return ""
    r = subprocess.run(["squeue", "-h", "-j", str(job), "-o", "%T"], capture_output=True, text=True)
    return r.stdout.strip()


def _require_alloc(trial: str) -> dict:
    st = read_state(trial)
    if not st:
        die(f"trial {trial} does not exist")
    state = _job_state(st.get("alloc_job"))
    if state != "RUNNING":
        die(f"the trial's GPU allocation (job {st.get('alloc_job')}) is {state or 'gone'}; "
            "stop and report this to the experimenter")
    return st


def _srun(st: dict, trial: str, inner: str, cpus: int, minutes: int | None = None) -> list[str]:
    cmd = ["srun", f"--jobid={st['alloc_job']}", "--overlap", "--ntasks=1", "--nodes=1",
           f"--cpus-per-task={cpus}", "--quiet"]
    if st.get("alloc_gres"):
        cmd.append(f"--gres={st['alloc_gres']}")
    if minutes:
        cmd.append(f"--time={minutes}")
    env = f"export PYTHONPATH={shlex.quote(str(REPO))} WSP_TRIAL={shlex.quote(trial)}"
    cmd += ["/bin/bash", "-c",
            f"cd {shlex.quote(str(REPO))} && source curc/env.sh && activate_hydro_env && {env} && {inner}"]
    return cmd


def _log_explanation(trial: str, header: str, explanation: str) -> None:
    with open(trial_dir(trial) / "explanations.md", "a") as f:
        f.write(f"\n## {header} ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n{explanation.strip()}\n")


def cmd_alloc(a) -> None:
    td = trial_dir(a.trial)
    td.mkdir(parents=True, exist_ok=True)
    with locked_state(a.trial) as st:
        if _job_state(st.get("alloc_job")) in ("RUNNING", "PENDING"):
            die(f"allocation {st['alloc_job']} already active")
        r = subprocess.run(
            ["sbatch", "--parsable", f"--job-name=wsp-{a.trial}", f"--partition={a.partition}",
             f"--qos={a.qos}", f"--account={a.account}", f"--gres={a.gres}", "--nodes=1",
             "--ntasks=1", f"--cpus-per-task={a.cpus}", f"--mem={a.mem}", f"--time={a.hours}:00:00",
             f"--output={td}/alloc_%j.out", "--wrap", "sleep infinity"],
            capture_output=True, text=True)
        if r.returncode != 0:
            die(r.stderr.strip())
        st["alloc_job"] = r.stdout.strip().split(";")[0]
        st["alloc_gres"] = a.gres
        st["alloc_cpus"] = a.cpus
    print(f"allocation job {st['alloc_job']} submitted for {a.trial}")


def render_task(trial: str) -> str:
    """The editor's task message: wsp/TASK.md with this trial's paths filled in."""
    text = (REPO / "wsp" / "TASK.md").read_text()
    values = {
        "{TRIAL_DIR}": str(trial_dir(trial)), "{TRIAL}": trial, "{REPO}": str(REPO),
        "{TRAIN_GAGES}": ", ".join(TRAIN_GAGES), "{SHARED}": str(WSP_ROOT / "shared"),
        "{PASSES}": str(DEFAULT_BUDGET["passes"]), "{WSP_ROOT}": str(WSP_ROOT),
    }
    for key, value in values.items():
        text = text.replace(key, value)
    return text


def cmd_task(a) -> None:
    text = render_task(a.trial)
    if a.stdout:
        print(text)
        return
    td = trial_dir(a.trial)
    td.mkdir(parents=True, exist_ok=True)
    (td / "message.txt").write_text(text)
    print(f"wrote {td / 'message.txt'}")


def cmd_status(a) -> None:
    st = read_state(a.trial)
    if not st:
        die(f"trial {a.trial} does not exist")
    b = st["budget"]
    print(f"trial {a.trial}: allocation {st.get('alloc_job')} {_job_state(st.get('alloc_job')) or 'gone'}")
    print(f"forward passes used {st['passes_used']}/{b['passes']}; code runs used {st['code_runs_used']}/{b['code_runs']}")
    running = trial_dir(a.trial) / ".fwd_running"
    if running.exists():
        print(f"pass running: {running.read_text().strip()}")
    edits = list_edits(a.trial)
    print(f"current edit: {len(edits)} tensor(s)" + "".join(f"\n  {n}" for n in edits))


def cmd_py(a) -> None:
    if not a.explanation.strip():
        die("an explanation is required before every code run")
    script = Path(a.script).resolve()
    if not script.is_file():
        die(f"no such script {script}")
    st = _require_alloc(a.trial)
    with locked_state(a.trial) as s:
        if s["code_runs_used"] >= s["budget"]["code_runs"]:
            die("code-run budget exhausted")
        s["code_runs_used"] += 1
        n = s["code_runs_used"]
    code_dir = trial_dir(a.trial) / "code"
    code_dir.mkdir(exist_ok=True)
    saved = code_dir / f"run_{n:03d}.py"
    saved.write_text(script.read_text())
    _log_explanation(a.trial, f"code run {n}", a.explanation)
    inner = f"CUDA_VISIBLE_DEVICES= timeout 900 python -m wsp.run_kernel {shlex.quote(str(saved))}"
    t0 = time.time()
    r = subprocess.run(_srun(st, a.trial, inner, cpus=8, minutes=16), capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    out = "\n".join(l for l in out.splitlines() if "deactivate" not in l and "usage: conda" not in l)
    if r.returncode == 124:
        out += "\n[killed: 900 s limit]"
    shown = out if len(out) <= OUT_LIMIT else out[:OUT_LIMIT] + f"\n[output cut at {OUT_LIMIT} characters]"
    with open(trial_dir(a.trial) / "tool_calls.jsonl", "a") as f:
        f.write(json.dumps({"kind": "python", "n": n, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "seconds": round(time.time() - t0, 1), "returncode": r.returncode,
                            "explanation": a.explanation, "code": saved.read_text(),
                            "output": shown}) + "\n")
    print(shown)
    print(f"[code run {n}, exit {r.returncode}, {time.time() - t0:.0f}s]")


def _print_summary(trial: str, k: int, out: Path) -> bool:
    if (out / "failed.json").exists():
        f = json.loads((out / "failed.json").read_text())
        print(f"pass {k} FAILED (not charged): {f['error']}")
        (trial_dir(trial) / ".fwd_running").unlink(missing_ok=True)
        return True
    if not (out / "done.json").exists():
        return False
    s = json.loads((out / "done.json").read_text())
    (trial_dir(trial) / ".fwd_running").unlink(missing_ok=True)
    if s["type"] == "generate":
        print(f"pass {k} done: generate model={s['model']} gages={len(s['gages'])} n={s['n']} "
              f"T={s['temperature']} max_turns={s['max_turns']}")
        for m, agg in s["per_model"].items():
            print(f"  {m}: {json.dumps(agg)}")
        if "paired_best_nse" in s:
            p = {k2: v for k2, v in s["paired_best_nse"].items() if k2 != "pairs"}
            print(f"  paired best NSE (edited - original): {json.dumps(p)}")
        print("  gage      s  model     best_nse  final  sims  parse_fail  invalid  evals(failed)")
        for m, rows in s["rollouts"].items():
            for r in rows:
                print(f"  {r['gage']}  {r['sample']}  {m:8s}  {r['best_nse']!s:>8}  {r['final_nse']!s:>6}  "
                      f"{r['n_sims']:4d}  {r['parse_failure_turns']:10d}  {r['invalid_tool_calls']:7d}  "
                      f"{r['n_evaluates']}({r['n_failed_evaluates']})")
        print(f"  files: {out}/<model>/<gage>_s<i>.json and .transcript.jsonl.gz")
    else:
        print(f"pass {k} done: {json.dumps(s)}")
    print(f"  passes used {s.get('passes_used')}, left {s.get('passes_left')}")
    return True


def _wait(trial: str, k: int) -> None:
    out = trial_dir(trial) / f"fwd_{k}"
    t0 = time.time()
    while time.time() - t0 < WAIT_SECONDS:
        if _print_summary(trial, k, out):
            return
        running = trial_dir(trial) / ".fwd_running"
        if running.exists():
            pid = json.loads(running.read_text()).get("pid")
            try:
                os.kill(pid, 0)
            except (OSError, TypeError):
                time.sleep(5)
                if not _print_summary(trial, k, out):
                    tail = (out / "log.txt").read_text()[-2000:] if (out / "log.txt").exists() else ""
                    running.unlink(missing_ok=True)
                    print(f"pass {k} died without a result (not charged). log tail:\n{tail}")
                return
        time.sleep(10)
    print(f"pass {k} still running after {(time.time() - t0) / 60:.0f} more minutes; "
          f"run `wsp/wsp wait {trial} {k}` to keep waiting (log: {out}/log.txt)")


def cmd_fwd(a) -> None:
    if not a.explanation.strip():
        die("an explanation is required before every forward pass")
    try:
        req = json.loads(Path(a.request).read_text())
    except Exception as e:
        die(f"request is not valid JSON: {e}")
    st = _require_alloc(a.trial)
    running = trial_dir(a.trial) / ".fwd_running"
    if running.exists():
        die(f"a pass is already running ({running.read_text().strip()}); use `wsp/wsp wait`")
    if st["passes_used"] >= st["budget"]["passes"]:
        die("forward-pass budget exhausted")
    with locked_state(a.trial) as s:
        k = s.get("next_k", 1)
        s["next_k"] = k + 1
    out = trial_dir(a.trial) / f"fwd_{k}"
    out.mkdir(parents=True)
    (out / "request.json").write_text(json.dumps(req, indent=2))
    _log_explanation(a.trial, f"forward pass {k}", a.explanation)
    with open(trial_dir(a.trial) / "tool_calls.jsonl", "a") as f:
        f.write(json.dumps({"kind": "forward", "k": k, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "explanation": a.explanation, "request": req}) + "\n")
    cpus = st.get("alloc_cpus", 16)
    inner = f"python wsp/fwd.py --trial {shlex.quote(a.trial)} --k {k}"
    log = open(out / "log.txt", "wb")
    proc = subprocess.Popen(_srun(st, a.trial, inner, cpus=cpus), stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    running.write_text(json.dumps({"k": k, "pid": proc.pid}))
    print(f"pass {k} started ({req.get('type')})")
    _wait(a.trial, k)


def cmd_wait(a) -> None:
    _wait(a.trial, int(a.k))


def main() -> None:
    ap = argparse.ArgumentParser(prog="wsp/wsp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("alloc"); p.add_argument("trial")
    p.add_argument("--hours", type=int, default=24); p.add_argument("--qos", default="gpu-normal")
    p.add_argument("--gres", default="gpu:h200_3g.71gb:1"); p.add_argument("--cpus", type=int, default=16)
    p.add_argument("--mem", default="160G")
    p.add_argument("--account", default=os.environ.get("WSP_ACCOUNT", "ucb-general"))
    p.add_argument("--partition", default=os.environ.get("WSP_PARTITION", "ah200"))
    p = sub.add_parser("task"); p.add_argument("trial"); p.add_argument("--stdout", action="store_true")
    p = sub.add_parser("status"); p.add_argument("trial")
    p = sub.add_parser("py"); p.add_argument("trial"); p.add_argument("script"); p.add_argument("explanation")
    p = sub.add_parser("fwd"); p.add_argument("trial"); p.add_argument("request"); p.add_argument("explanation")
    p = sub.add_parser("wait"); p.add_argument("trial"); p.add_argument("k")
    a = ap.parse_args()
    {"alloc": cmd_alloc, "task": cmd_task, "status": cmd_status, "py": cmd_py, "fwd": cmd_fwd,
     "wait": cmd_wait}[a.cmd](a)


if __name__ == "__main__":
    main()
