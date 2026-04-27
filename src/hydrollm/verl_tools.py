"""verl-compatible tools for EF5/CREST calibration.

Each rollout gets an isolated `HydroEnvironment` keyed on the verl
`instance_id`. The three tools share the same env so `set_parameters` →
`run_simulation` → `evaluate` cooperate within a single trajectory.

Per-turn reward shape (returned by `evaluate`):
    reward = ΔNSE - λ * invalid_in_this_turn

Where ΔNSE is the change vs. the previous valid NSE in this trajectory.
The terminal NSE bonus is added separately by `verl_reward.compute_score`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from hydrollm.config import GageConfig, load_gage_config
from hydrollm.environment import HydroEnvironment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_ENV_REGISTRY: dict[str, HydroEnvironment] = {}
_NSE_REGISTRY: dict[str, list[float]] = {}
_INVALID_PENALTY = 0.5
_REWARD_SET_PARAMS = 0.02   # densify per-turn signal: reward valid protocol step
_REWARD_RUN_SIM = 0.05      # slightly higher: this is the costly artifact-producing call

# Limit concurrent EF5 invocations PER PYTHON WORKER PROCESS. verl spawns
# one worker per GPU, so the *system-wide* concurrency is
# `_EF5_CONCURRENCY × num_workers`. With 4 workers × 8 = 32 system-wide,
# each EF5 gets ~2 of the 64 vCPUs — MRMS files page-cache after the first
# read across rollouts, so I/O contention stays low.
_EF5_CONCURRENCY = 8
_EF5_SEMAPHORE: asyncio.Semaphore | None = None


def _get_ef5_semaphore() -> asyncio.Semaphore:
    global _EF5_SEMAPHORE
    if _EF5_SEMAPHORE is None:
        _EF5_SEMAPHORE = asyncio.Semaphore(_EF5_CONCURRENCY)
    return _EF5_SEMAPHORE


def _get_env(instance_id: str) -> HydroEnvironment | None:
    return _ENV_REGISTRY.get(instance_id)


def _release_instance(instance_id: str) -> None:
    env = _ENV_REGISTRY.pop(instance_id, None)
    _NSE_REGISTRY.pop(instance_id, None)
    if env is not None:
        try:
            env.cleanup()
        except Exception:
            logger.exception("Error cleaning up sandbox %s", instance_id)


# ---------------------------------------------------------------------------
# Base class shim
# ---------------------------------------------------------------------------
# We inherit from verl.tools.base_tool.BaseTool when running under verl, but
# keep a stub for local unit tests that don't have verl installed.

try:  # pragma: no cover - import guard
    from verl.tools.base_tool import BaseTool  # type: ignore
    from verl.tools.schemas import OpenAIFunctionToolSchema  # type: ignore
    _VERL_AVAILABLE = True
except Exception:  # pragma: no cover
    _VERL_AVAILABLE = False

    class BaseTool:  # minimal stand-in
        def __init__(self, config: dict, tool_schema: Any):
            self.config = config or {}
            self.tool_schema = tool_schema

        def get_openai_tool_schema(self):
            return self.tool_schema

    OpenAIFunctionToolSchema = dict  # type: ignore


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class _HydroToolBase(BaseTool):
    """Shared helpers for the three EF5 tools."""

    def __init__(self, config: dict, tool_schema: Any):
        super().__init__(config, tool_schema)
        # Default gage config path can come from tool config; per-rollout
        # `create()` may override via tools_kwargs.
        self._default_gage_path: str | None = (config or {}).get("gage_config_path")

    async def create(
        self,
        instance_id: str | None = None,
        gage_config_path: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Initialize a sandbox for this rollout if not already present.

        verl creates one tool *instance* per registered tool, but calls
        `create()` once per rollout per tool. We dedupe so the first
        call wins and the other two tools just attach to the same env.
        """
        instance_id = instance_id or uuid.uuid4().hex
        if instance_id not in _ENV_REGISTRY:
            path = gage_config_path or self._default_gage_path
            if path is None:
                raise ValueError(
                    "verl_tools: gage_config_path must be supplied via "
                    "tool config or tools_kwargs."
                )
            gage = load_gage_config(path)
            _ENV_REGISTRY[instance_id] = HydroEnvironment(gage)
            _NSE_REGISTRY[instance_id] = []
            logger.warning("[HYDRO_TOOL] create instance=%s gage=%s", instance_id, gage.gage_id)
        return instance_id

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        # Per-turn rewards are returned in `execute`; there's nothing extra
        # to add here. Final NSE bonus is delivered by compute_score.
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        _release_instance(instance_id)


class SetParametersTool(_HydroToolBase):
    """Validate and write CREST parameters to the control file."""

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[str, float, dict]:
        env = _get_env(instance_id)
        if env is None:
            return (
                json.dumps({"status": "error", "message": "no sandbox"}),
                -_INVALID_PENALTY,
                {"invalid": True},
            )
        try:
            logger.warning("[HYDRO_TOOL] set_parameters instance=%s params=%s", instance_id, parameters)
            result = await asyncio.to_thread(env.set_parameters, parameters)
            return json.dumps(result), _REWARD_SET_PARAMS, {"invalid": False}
        except Exception as e:
            logger.exception("set_parameters failed")
            return (
                json.dumps({"status": "error", "message": str(e)}),
                -_INVALID_PENALTY,
                {"invalid": True},
            )


class RunSimulationTool(_HydroToolBase):
    """Run EF5 with current parameters."""

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[str, float, dict]:
        env = _get_env(instance_id)
        if env is None:
            return (
                json.dumps({"status": "error", "message": "no sandbox"}),
                -_INVALID_PENALTY,
                {"invalid": True},
            )
        logger.warning("[HYDRO_TOOL] run_simulation instance=%s", instance_id)
        async with _get_ef5_semaphore():
            result = await asyncio.to_thread(env.run_simulation)
        invalid = result.get("status") != "ok"
        if invalid:
            logger.warning(
                "[HYDRO_TOOL] run_simulation FAIL instance=%s gage=%s msg=%s stderr=%s",
                instance_id, env.gage.gage_id,
                result.get("message"), (result.get("stderr") or "")[:500],
            )
            _dump_sandbox_diagnostics(env)
        reward = -_INVALID_PENALTY if invalid else _REWARD_RUN_SIM
        return json.dumps(result), reward, {"invalid": invalid}


class EvaluateTool(_HydroToolBase):
    """Compute NSE/KGE/etc and emit per-turn ΔNSE reward."""

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[str, float, dict]:
        env = _get_env(instance_id)
        if env is None:
            return (
                json.dumps({"status": "error", "message": "no sandbox"}),
                -_INVALID_PENALTY,
                {"invalid": True},
            )
        logger.warning("[HYDRO_TOOL] evaluate instance=%s", instance_id)
        result = await asyncio.to_thread(env.evaluate)
        if result.get("status") != "ok":
            logger.warning(
                "[HYDRO_TOOL] evaluate FAIL instance=%s gage=%s msg=%s",
                instance_id, env.gage.gage_id, result.get("message"),
            )
            _dump_csv_diagnostics(env)
        nse = result.get("NSE")
        history = _NSE_REGISTRY.setdefault(instance_id, [])
        if isinstance(nse, (int, float)) and nse > -998:
            prev = history[-1] if history else 0.0
            delta = float(nse) - prev
            history.append(float(nse))
            reward = delta
            metrics = {"nse": float(nse), "delta_nse": delta, "invalid": False}
        else:
            reward = -_INVALID_PENALTY
            metrics = {"invalid": True}
        return json.dumps(result), reward, metrics


# ---------------------------------------------------------------------------
# Helpers for tests / orchestration
# ---------------------------------------------------------------------------

def _dump_sandbox_diagnostics(env: HydroEnvironment) -> None:
    """Log control-file path, MRMS LOC contents, OBS path existence."""
    from pathlib import Path
    try:
        gage_id = env.gage.gage_id
        cp = env.control_path
        logger.warning("[HYDRO_DIAG] control=%s exists=%s", cp, cp.exists())
        if cp.exists():
            for line in cp.read_text().splitlines():
                up = line.strip().upper()
                if any(up.startswith(k) for k in ("OBS=", "LOC=", "TIME_", "OUTPUT=", "[GAUGE")):
                    logger.warning("[HYDRO_DIAG]   %s", line.strip())
        mrms = Path(f"/app/data/data_mrms_clip/{gage_id}")
        if mrms.exists():
            files = sorted(p.name for p in mrms.iterdir())
            logger.warning(
                "[HYDRO_DIAG] mrms=%s n=%d first=%s last=%s",
                mrms, len(files), files[0] if files else None, files[-1] if files else None,
            )
        else:
            logger.warning("[HYDRO_DIAG] mrms=%s MISSING", mrms)
        obs = Path(env.gage.obs_dir) / env._obs_filename
        logger.warning("[HYDRO_DIAG] obs=%s exists=%s", obs, obs.exists())
        if env.output_dir.exists():
            outs = sorted(p.name for p in env.output_dir.iterdir())
            logger.warning("[HYDRO_DIAG] sandbox_out=%s files=%s", env.output_dir, outs[:10])
    except Exception as e:
        logger.warning("[HYDRO_DIAG] dump failed: %s", e)


def _dump_csv_diagnostics(env: HydroEnvironment) -> None:
    """When evaluate fails, dump the EF5 output CSV header and first rows."""
    try:
        from pathlib import Path
        out = env.output_dir
        if out.exists():
            outs = sorted(out.iterdir())
            logger.warning("[HYDRO_DIAG] eval_fail sandbox_out=%s files=%s", out, [p.name for p in outs[:10]])
            for p in outs:
                if p.suffix == ".csv":
                    head = p.read_text()[:1500]
                    logger.warning("[HYDRO_DIAG] csv=%s head=\n%s", p.name, head)
                    break
        else:
            logger.warning("[HYDRO_DIAG] eval_fail sandbox_out=%s MISSING", out)
        # Also dump control file lines for context
        _dump_sandbox_diagnostics(env)
    except Exception as e:
        logger.warning("[HYDRO_DIAG] csv dump failed: %s", e)


def get_nse_history(instance_id: str) -> list[float]:
    """Read the NSE trajectory accumulated for a given rollout."""
    return list(_NSE_REGISTRY.get(instance_id, []))


def reset_registry() -> None:
    """Test helper: drop all sandboxes."""
    for iid in list(_ENV_REGISTRY.keys()):
        _release_instance(iid)
