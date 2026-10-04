"""Orchestrates `yv ab`: arms, run directory, stage ladder, verdict. Resumable and fail-fast."""

from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from . import stages as st
from .core import (
    RUNS,
    Arm,
    Gpuq,
    InfraError,
    RunDir,
    now,
    resolve_arm,
    sha,
    write_json_atomic,
)
from .execute import Executor
from .suites import parse_suite
from .verdict import build_verdict, render_md


def parse_kv(items: list) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"expected K=V, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def apply_overrides(cfg: dict, a) -> dict:
    cfg = dict(cfg)
    if getattr(a, "ctx", None):
        cfg["ctx"] = [int(x) for x in a.ctx.split(",")]
    if getattr(a, "reps", None):
        cfg["reps"] = a.reps
    if getattr(a, "mmlu_n", None):
        cfg["mmlu_n"] = a.mmlu_n
    if getattr(a, "mem_sizes", None):
        cfg["mem_sizes"] = [int(x) for x in a.mem_sizes.split(",")]
    if getattr(a, "mem_reps", None):
        cfg["mem_reps"] = a.mem_reps
    if getattr(a, "speed_tol", None) is not None:
        cfg["speed_tol_pct"] = a.speed_tol
    if getattr(a, "spec_off", None) is not None:
        cfg["spec_off"] = a.spec_off
    if getattr(a, "no_apc_hit_required", False):
        cfg["apc_require_hit"] = False
    return cfg


def run_dir_for(label: str, cand: Arm, runs: Path | None = None) -> Path:
    return (runs or RUNS) / f"{label}-{cand.key[:21]}"


def run_ab(
    a,
    gq: Gpuq | None = None,
    log: Callable[[str], None] | None = None,
    runs: Path | None = None,
    trees: Path | None = None,
) -> int:
    log = log or (lambda m: print(f"[yv {time.strftime('%H:%M:%S')}] {m}", flush=True))
    started = now()
    gq = gq or Gpuq()
    try:
        suite = apply_overrides(parse_suite(a.suite), a)
        env, cand_env, base_env = (
            parse_kv(a.env),
            parse_kv(a.cand_env),
            parse_kv(getattr(a, "base_env", [])),
        )
        base = resolve_arm("base", a.base, trees)
        cand = resolve_arm("cand", a.cand, trees)
    except (ValueError, InfraError) as e:
        print(f"yv: {e}")
        return 2
    rd = RunDir(run_dir_for(a.label, cand, runs))
    big = "27B" in a.model
    mem_gb = a.mem_gb or (60 if big else 14)
    fp = sha(
        base.key, cand.key, env, cand_env, base_env, suite, a.model, mem_gb, a.engaged
    )
    state = rd.state()
    if state and state.get("fingerprint") != fp:
        print(
            f"yv: {rd.path} holds a run with different parameters (base/cand/env/suite/model); "
            "use another --label or remove it"
        )
        return 2
    state.update(
        fingerprint=fp,
        label=a.label,
        base=base.to_json(),
        cand=cand.to_json(),
        env=env,
        cand_env=cand_env,
        base_env=base_env,
        suite=suite,
        model=a.model,
        pid=os.getpid(),
        status="running",
        stage=None,
        started=state.get("started", started),
        args=vars(a) if hasattr(a, "__dict__") else {},
    )
    state.pop("verdict", None)
    rd.save_state(state)
    (rd.path / "verdict.json").unlink(missing_ok=True)
    exe = Executor(rd, gq, f"infra-{a.label}"[:40], log, priority=a.priority)
    ctx = st.Ctx(
        run=rd,
        exe=exe,
        base=base,
        cand=cand,
        env=env,
        cand_env=cand_env,
        base_env=base_env,
        model=a.model,
        model_name=a.model_name or Path(a.model).name,
        suite=suite,
        mem_gb=mem_gb,
        engaged=list(a.engaged or []),
        log=log,
    )
    results, infra = [], ""
    log(
        f"run dir {rd.path}; base {base.key} cand {cand.key}; suite {suite['name']} {suite['stages']}"
    )
    try:
        for name in suite["stages"]:
            state["stage"] = name
            rd.save_state(state)
            log(f"stage {name}")
            res = st.STAGE_FUNCS[name](ctx)
            results.append(res.to_json())
            state.setdefault("stages", {})[name] = res.to_json()["status"]
            rd.save_state(state)
            log(
                f"stage {name}: {'PASS' if res.passed else 'FAIL'} {'; '.join(map(str, res.reasons[:2]))[:300]}"
            )
            if not res.passed:
                break  # fail-fast
    except InfraError as e:
        infra = str(e)
    except Exception as e:  # noqa: BLE001 - the tool's own bug is an infra error, never a pass
        infra = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
    v = build_verdict(
        label=a.label,
        base=base.to_json(),
        cand=cand.to_json(),
        env=env,
        cand_env=cand_env,
        base_env=base_env,
        suite=suite,
        model=a.model,
        stages=results,
        planned=suite["stages"],
        infra_error=infra,
        started=started,
        run_dir=str(rd.path),
    )
    write_json_atomic(rd.path / "verdict.json", v)
    (rd.path / "verdict.md").write_text(render_md(v))
    state.update(
        status="finished", overall=v["overall"], exit_code=v["exit_code"], stage=None
    )
    rd.save_state(state)
    log(f"verdict {v['overall']} (exit {v['exit_code']}): {rd.path / 'verdict.md'}")
    return v["exit_code"]


def load_run(spec: str, runs: Path | None = None) -> RunDir:
    p = Path(spec)
    if not p.is_dir():
        p = (runs or RUNS) / spec
    if not p.is_dir():
        raise InfraError(f"no such run: {spec}")
    return RunDir(p)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def status_text(rd: RunDir) -> str:
    s = rd.state()
    if not s:
        return f"{rd.path}: no state"
    v = rd.path / "verdict.json"
    head = f"{s.get('label')} base {s['base']['key']} cand {s['cand']['key']} suite {s['suite'].get('name')}"
    if v.exists():
        vv = json.loads(v.read_text())
        return f"{head}\nverdict {vv['overall']} (exit {vv['exit_code']}) failed_stage={vv['failed_stage']}\n{rd.path / 'verdict.md'}"
    run = (
        "running"
        if alive(int(s.get("pid", 0) or 0))
        else "NOT RUNNING (rerun the same command to resume)"
    )
    return f"{head}\n{run}; stage {s.get('stage')}; done {s.get('stages', {})}"


def wait_run(rd: RunDir, sleep: float = 10.0, poll=time.sleep) -> int:
    """Block until the run has a verdict, or its process is gone; exit code = the verdict's."""
    while True:
        v = rd.path / "verdict.json"
        if v.exists():
            return int(json.loads(v.read_text())["exit_code"])
        s = rd.state()
        if s and not alive(int(s.get("pid", 0) or 0)):
            print(
                "yv: the run process is gone without a verdict; rerun the same command to resume"
            )
            return 2
        poll(sleep)
