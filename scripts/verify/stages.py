"""The verification ladder: one function per stage, each returns a StageResult.

A stage writes its final `stage_complete` record only when its analysis finished
(fail closed: missing evidence = fail). Cells are gpuq jobs run by execute.Executor.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import analyze
from .core import (
    REPO,
    VENV_PY,
    Arm,
    InfraError,
    RunDir,
    changed_files,
    read_jsonl,
    related_tests,
)
from .execute import Cell, Executor

TFBENCH = REPO / "scripts/research/tfbench.py"
MEMORY_AB = REPO / "scripts/research/memory_ab.py"
PAIRED = REPO / "scripts/research/accuracy/paired_eval.py"
PROMPTS = Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts")
PORT_LAST = "18996"  # servers of this tool stay inside 18990-18996


@dataclass
class StageResult:
    name: str
    passed: bool
    reasons: list = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    jobs: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "status": "PASS" if self.passed else "FAIL",
            "reasons": self.reasons,
            "numbers": self.numbers,
            "jobs": self.jobs,
        }


@dataclass
class Ctx:
    run: RunDir
    exe: Executor
    base: Arm
    cand: Arm
    env: dict  # both arms
    cand_env: dict  # candidate only
    base_env: dict
    model: str
    model_name: str
    suite: dict
    mem_gb: float
    engaged: list = field(default_factory=list)
    py: str = VENV_PY
    log: object = print

    def arm_env(self, arm: str, extra: dict | None = None) -> dict:
        e = dict(self.env)
        e.update(self.cand_env if arm == "cand" else self.base_env)
        e.update(extra or {})
        return e

    def tree(self, arm: str) -> Arm:
        return self.cand if arm == "cand" else self.base

    @property
    def big(self) -> bool:
        return self.mem_gb >= 40


def _env_flags(env: dict) -> list:
    out = []
    for k, v in sorted(env.items()):
        out += ["--env", f"{k}={v}"]
    return out


def _tfbench_argv(
    ctx: Ctx, arm: str, stage: str, key: str, extra: list, env: dict | None = None
) -> list:
    tfb_out = ctx.run.path / "tfb" / f"{stage}.{key}"
    return [
        "env",
        f"TFB_YUNSHU_SRC={ctx.tree(arm).path / 'python'}",
        f"TFB_OUT={tfb_out}",
        f"TFB_PORT_LAST={PORT_LAST}",
        ctx.py,
        str(TFBENCH),
        "--engine",
        "yunshu",
        "--model",
        ctx.model,
        "--out",
        "{out}",
        f"--tag=-{stage}-{key}",
        *_env_flags(ctx.arm_env(arm) if env is None else env),
        *extra,
    ]


def _tfb_log(ctx: Ctx, stage: str, key: str, rep: int = 0) -> Path:
    return (
        ctx.run.path
        / "tfb"
        / f"{stage}.{key}"
        / "out"
        / f"server-yunshu-decode-{rep}-{stage}-{key}.log"
    )


def _decode_est_min(ctx: Ctx, ctxs: list, kinds: int) -> float:
    start = 2.5 if ctx.big else 0.7
    secs = 0.0
    for c in ctxs:
        per = (3 + c / 1000 * (1.4 if ctx.big else 0.15)) * 3  # cold, warm, follow-up
        secs += per * kinds
    return min(20.0, max(5.0, (start * 60 + secs) * 1.6 / 60))


def _check_prompts(ctxs: list, kinds: list) -> None:
    missing = [
        f"{k}-{c}"
        for c in ctxs
        for k in kinds
        if not (PROMPTS / f"{k}-{c}.txt").exists()
    ]
    if missing:
        raise InfraError(f"tfbench prompt files missing in {PROMPTS}: {missing}")


def _finish(ctx: Ctx, res: StageResult) -> StageResult:
    res.jobs = [j for j in ctx.exe.jobs if j[0] == res.name]
    ctx.run.append(res.name, {"ev": "stage_complete", **res.to_json()})
    return res


def _failed_cells(results: dict) -> list:
    return [f"{k}: {r.reason}" for k, r in results.items() if not r.ok]


# ── a. preflight (CPU only; no gpuq) ─────────────────────────────────────
def stage_preflight(ctx: Ctx) -> StageResult:
    name, reasons, numbers = "preflight", [], {}
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    for arm in (ctx.base, ctx.cand):
        e = dict(env, PYTHONPATH=str(arm.path / "python"))
        r = subprocess.run(
            [
                ctx.py,
                "-c",
                "import yunshu_engine, yunshu_gateway, yunshu_kv; print('ok')",
            ],
            cwd=str(arm.path),
            env=e,
            capture_output=True,
            text=True,
        )
        ok = r.returncode == 0
        numbers[f"import_{arm.name}"] = ok
        ctx.run.append(
            name, {"ev": "import", "arm": arm.name, "ok": ok, "err": r.stderr[-400:]}
        )
        if not ok:
            reasons.append(
                f"{arm.name} tree does not import: {r.stderr.strip().splitlines()[-1:]}"
            )
    changed = changed_files(ctx.base, ctx.cand)
    tests = related_tests(changed, ctx.cand.path)
    numbers.update(changed_files=len(changed), related_tests=len(tests))
    ctx.run.append(name, {"ev": "mapping", "changed": changed, "tests": tests})
    if not reasons and tests:
        cmd = [
            "nice",
            "-n",
            "15",
            ctx.py,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-x",
            *tests,
        ]
        r = subprocess.run(
            cmd,
            cwd=str(ctx.cand.path),
            env=dict(env, PYTHONPATH=str(ctx.cand.path / "python")),
            capture_output=True,
            text=True,
            timeout=30 * 60,
        )
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-8:])
        ctx.run.append(
            name, {"ev": "pytest", "rc": r.returncode, "tail": tail, "tests": tests}
        )
        numbers["pytest_rc"] = r.returncode
        if r.returncode != 0:
            reasons.append(
                f"related unit tests failed (rc {r.returncode}): {tail[-300:]}"
            )
    elif not tests and any(f.endswith(".py") for f in changed):
        numbers["note"] = "no related unit tests found for the changed python files"
    return _finish(ctx, StageResult(name, not reasons, reasons, numbers))


# ── b. smoke ─────────────────────────────────────────────────────────────
def engaged_checks(specs: list, rows: list, log_text: str) -> list:
    """Evaluate --engaged specs on the candidate's evidence; returns failure reasons.
    log:REGEX (server log), field:KEY=REGEX (x_yunshu response field), spec:MODE (engaged spec mode)."""
    bad = []
    sess = analyze.session_info(rows)
    for s in specs:
        kind, _, arg = s.partition(":")
        if kind == "log":
            if not re.search(arg, log_text):
                bad.append(f"candidate path not engaged: log has no match for /{arg}/")
        elif kind == "field":
            k, _, rx = arg.partition("=")
            vals = [
                str((r.get("xy") or {}).get(k))
                for r in rows
                if r.get("part") == "decode"
            ]
            if not any(re.search(rx, v) for v in vals):
                bad.append(
                    f"candidate path not engaged: x_yunshu.{k} never matched /{rx}/ (saw {sorted(set(vals))[:4]})"
                )
        elif kind == "spec":
            if sess.get("engaged_spec_mode") != arg:
                bad.append(
                    f"candidate spec mode {sess.get('engaged_spec_mode')!r} != {arg!r}"
                )
        else:
            bad.append(f"unknown --engaged kind {s!r} (use log:, field:, spec:)")
    return bad


def _smoke_valid(path: Path):
    rows = read_jsonl(path)
    dec = [r for r in rows if r.get("part") == "decode"]
    if not any(r.get("complete") is True for r in rows):
        return False, "no complete record"
    if not dec:
        return False, "no decode records"
    if any(
        not (r.get("ct") or 0) > 0 or not (r.get("text") or "").strip() for r in dec
    ):
        return False, "a smoke request returned no tokens"
    return True, ""


def stage_smoke(ctx: Ctx) -> StageResult:
    cells = []
    for arm in ("base", "cand"):
        cells.append(
            Cell(
                "smoke",
                arm,
                _tfbench_argv(
                    ctx,
                    arm,
                    "smoke",
                    arm,
                    ["--part", "decode", "--smoke", "--rep", "0"],
                ),
                mem_gb=ctx.mem_gb,
                timeout_min=12 if ctx.big else 5,
                stall_min=6 if ctx.big else 2,
                validate=_smoke_valid,
            )
        )
    res = ctx.exe.run_cells(cells)
    reasons = _failed_cells(res)
    numbers = {}
    if not reasons and ctx.engaged:
        rows = read_jsonl(res["cand"].evidence)
        lg = _tfb_log(ctx, "smoke", "cand")
        reasons += engaged_checks(
            ctx.engaged, rows, lg.read_text(errors="replace") if lg.exists() else ""
        )
        numbers["engaged"] = list(ctx.engaged)
        numbers["engaged_ok"] = not reasons
    for k, r in res.items():
        if r.ok and r.evidence:
            numbers[f"engaged_spec_mode_{k}"] = analyze.session_info(
                read_jsonl(r.evidence)
            ).get("engaged_spec_mode")
    return _finish(ctx, StageResult("smoke", not reasons, reasons, numbers))


# ── c. identity ──────────────────────────────────────────────────────────
def _identity_cells(ctx: Ctx) -> list:
    cfg = ctx.suite
    arms = [("base", {}), ("cand", {})]
    off = ctx.arm_env("cand").get("YUNSHU_VLM_DRAFT", "").lower() in ("off", "none")
    if cfg["spec_off"] and not off:
        arms.append(("cand-off", {"YUNSHU_VLM_DRAFT": "off"}))
    cells = []
    for c in cfg["ctx"]:
        for arm, extra in arms:
            base_arm = "cand" if arm.startswith("cand") else "base"
            cells.append(
                Cell(
                    "identity",
                    f"{arm}-{c}",
                    _tfbench_argv(
                        ctx,
                        base_arm,
                        "identity",
                        f"{arm}-{c}",
                        ["--part", "decode", "--rep", "0", "--only-ctx", str(c)],
                        env=ctx.arm_env(base_arm, extra),
                    ),
                    mem_gb=ctx.mem_gb,
                    timeout_min=_decode_est_min(ctx, [c], len(cfg["kinds"])),
                    stall_min=12 if ctx.big else 4,
                )
            )
    return cells


def _rows_for(res: dict, prefix: str) -> list:
    out = []
    for k, r in sorted(res.items()):
        if re.fullmatch(re.escape(prefix) + r"-\d+", k) and r.ok and r.evidence:
            out += read_jsonl(r.evidence)
    return out


def stage_identity(ctx: Ctx) -> StageResult:
    _check_prompts(ctx.suite["ctx"], ctx.suite["kinds"])
    res = ctx.exe.run_cells(_identity_cells(ctx))
    reasons = _failed_cells(res)
    numbers: dict = {}
    if not reasons:
        base, cand = _rows_for(res, "base"), _rows_for(res, "cand")
        cmp_ = analyze.compare_identity(base, cand, "base", "cand")
        numbers["base_vs_cand"] = {
            "compared": cmp_["compared"],
            "mismatches": len(cmp_["mismatches"]),
        }
        if not cmp_["ok"]:
            reasons.append(f"base != cand: {cmp_['mismatches'][:3]}")
        offrows = _rows_for(res, "cand-off")
        if offrows:
            c2 = analyze.compare_identity(offrows, cand, "spec_off", "cand")
            numbers["spec_on_vs_off"] = {
                "compared": c2["compared"],
                "mismatches": len(c2["mismatches"]),
            }
            if not c2["ok"]:
                reasons.append(f"spec on != spec off: {c2['mismatches'][:3]}")
        numbers["engaged_spec_mode"] = {
            "base": analyze.session_info(base).get("engaged_spec_mode"),
            "cand": analyze.session_info(cand).get("engaged_spec_mode"),
        }
    return _finish(ctx, StageResult("identity", not reasons, reasons, numbers))


# ── d. apc (analysis of the candidate's cold/warm pairs from the identity cells) ──
def stage_apc(ctx: Ctx) -> StageResult:
    rows = []
    for c in ctx.suite["ctx"]:
        p = ctx.run.cell_path("identity", f"cand-{c}")
        rows += read_jsonl(p)
    if not rows:
        return _finish(
            ctx,
            StageResult(
                "apc",
                False,
                ["no candidate identity evidence (run the identity stage)"],
            ),
        )
    a = analyze.check_apc(rows, require_hit=ctx.suite.get("apc_require_hit", True))
    reasons = [f"{m['cell']}: {m['why']}" for m in a["mismatches"]]
    return _finish(
        ctx,
        StageResult(
            "apc", a["ok"], reasons, {"compared": a["compared"], "hits": a["hits"]}
        ),
    )


# ── e. quality ───────────────────────────────────────────────────────────
def _quality_arm_file(ctx: Ctx, arm: str) -> Path:
    return ctx.run.path / "quality" / "mmlu_pro" / f"{arm}.jsonl"


def _quality_counts(ctx: Ctx, n: int) -> dict:
    out = {}
    for arm in ("base", "cand"):
        ids = {
            r["id"]
            for r in read_jsonl(_quality_arm_file(ctx, arm))
            if r.get("kind") == "q"
            and not r.get("error")
            and r.get("correct") is not None
        }
        out[arm] = len(ids)
    return out


def stage_quality(ctx: Ctx, max_rounds: int = 6) -> StageResult:
    n = int(ctx.suite["mmlu_n"])
    budget = 14
    reasons: list = []
    for _ in range(max_rounds):
        counts = _quality_counts(ctx, n)
        todo = [a for a in ("base", "cand") if counts[a] < n]
        if not todo:
            break
        cells = []
        for arm in todo:
            cells.append(
                Cell(
                    "quality",
                    _round_key(ctx, arm),
                    [
                        "env",
                        f"PAIRED_TREE={ctx.tree(arm).path}",
                        f"PAIRED_OUT={ctx.run.path / 'quality'}",
                        f"PAIRED_PORT_LAST={PORT_LAST}",
                        os.environ.get(
                            "PAIRED_PY",
                            "/Volumes/P5Plus/yunshu-test-envs/paired-eval/bin/python",
                        ),
                        str(PAIRED),
                        "run",
                        "--bench",
                        "mmlu_pro",
                        "--mmlu-n",
                        str(n),
                        "--arm",
                        arm,
                        "--model",
                        ctx.model,
                        "--model-name",
                        ctx.model_name,
                        "--budget-min",
                        str(budget),
                        "--concurrency",
                        "8",
                        "--port",
                        "18990",
                        *_env_flags(ctx.arm_env(arm)),
                    ],
                    mem_gb=ctx.mem_gb + (8 if ctx.big else 0),
                    timeout_min=20,
                    stall_min=12,
                    needs_out=False,
                )
            )
        res = ctx.exe.run_cells(cells)
        reasons = _failed_cells(res)
        if reasons:
            break
    counts = _quality_counts(ctx, n)
    q = analyze.quality_compare(
        read_jsonl(_quality_arm_file(ctx, "base")),
        read_jsonl(_quality_arm_file(ctx, "cand")),
        n,
        ctx.suite["quality_allowed"],
    )
    numbers = dict(q, counts=counts)
    if not reasons:
        if not q["complete"]:
            reasons.append(
                f"incomplete after {max_rounds} rounds: scored {q['n']}/{n} paired items"
            )
        elif not q["ok"]:
            reasons.append(
                f"net {q['net']:+d} correct (base {q['base_correct']}, cand {q['cand_correct']}) outside +-{q['allowed']}"
            )
    return _finish(ctx, StageResult("quality", not reasons, reasons, numbers))


def _round_key(ctx: Ctx, arm: str) -> str:
    """The in-flight round of `arm` (a resume re-attaches to it), else the next round number."""
    rounds: dict = {}
    for r in ctx.run.rows("quality"):
        c = str(r.get("cell", ""))
        if not c.startswith(arm + "-r"):
            continue
        if r.get("ev") == "cell_submitted":
            rounds[c] = "open"
        elif r.get("ev") in ("cell_done", "cell_cancelled"):
            rounds[c] = "closed"
    for c, st in rounds.items():
        if st == "open":
            return c
    return f"{arm}-r{len(rounds) + 1}"


# ── f. speed ─────────────────────────────────────────────────────────────
def stage_speed(ctx: Ctx) -> StageResult:
    cfg = ctx.suite
    _check_prompts(cfg["ctx"], cfg["kinds"])
    cells = []
    for rep in range(int(cfg["reps"])):
        for arm in ("base", "cand"):  # A B A B ...
            cells.append(
                Cell(
                    "speed",
                    f"{arm}-r{rep}",
                    _tfbench_argv(
                        ctx,
                        arm,
                        "speed",
                        f"{arm}-r{rep}",
                        ["--part", "decode", "--rep", str(rep)]
                        + [x for c in cfg["ctx"] for x in ("--only-ctx", str(c))],
                    ),
                    mem_gb=ctx.mem_gb,
                    timeout_min=_decode_est_min(ctx, cfg["ctx"], len(cfg["kinds"])),
                    stall_min=12 if ctx.big else 4,
                    quiet=True,
                )
            )
    res = ctx.exe.run_cells(cells)
    reasons = _failed_cells(res)
    numbers: dict = {}
    if not reasons:
        reps = int(cfg["reps"])
        base = [read_jsonl(res[f"base-r{i}"].evidence) for i in range(reps)]
        cand = [read_jsonl(res[f"cand-r{i}"].evidence) for i in range(reps)]
        sp = analyze.speed_compare(base, cand, float(cfg["speed_tol_pct"]))
        numbers = {k: sp[k] for k in ("cells", "reps", "tol_pct")}
        for r in sp["regressions"]:
            reasons.append(
                f"{r['kind']}@{r['ctx']} {r['metric']}: {r['base_median']} -> {r['cand_median']} "
                f"({r['delta_pct']:+.1f}%, limit {r['limit_pct']:.1f}%)"
            )
        reasons += [f"missing {m}" for m in sp["missing"]]
        if not sp["cells"]:
            reasons.append("no comparable speed cells")
    return _finish(ctx, StageResult("speed", not reasons, reasons, numbers))


# ── g. memory ────────────────────────────────────────────────────────────
def stage_memory(ctx: Ctx) -> StageResult:
    cfg = ctx.suite
    reps = int(cfg.get("mem_reps", 2))
    sizes = [str(s) for s in cfg["mem_sizes"]]
    cells = []
    for rep in range(reps):
        order = ("base", "cand") if rep % 2 == 0 else ("cand", "base")
        for arm in order:
            e = ctx.arm_env(arm)
            cells.append(
                Cell(
                    "memory",
                    f"{arm}-r{rep}",
                    [
                        ctx.py,
                        str(MEMORY_AB),
                        "--arm",
                        f"{arm}={ctx.tree(arm).path}",
                        *sum(
                            (
                                ["--arm-env", f"{arm}:{k}={v}"]
                                for k, v in sorted(e.items())
                            ),
                            [],
                        ),
                        "--model",
                        ctx.model,
                        "--port",
                        "18995",
                        "--reps",
                        "1",
                        "--rep-offset",
                        str(rep),
                        "--sizes",
                        *sizes,
                        "--out",
                        "{out}",
                    ],
                    mem_gb=ctx.mem_gb + (8 if ctx.big else 0),
                    timeout_min=20 if ctx.big else 8,
                    stall_min=12 if ctx.big else 4,
                    validate=_memory_valid,
                    quiet=False,
                )
            )
    res = ctx.exe.run_cells(cells)
    reasons = _failed_cells(res)
    numbers: dict = {}
    if not reasons:
        rows = []
        for r in res.values():
            rows += read_jsonl(r.evidence)
        m = analyze.memory_compare(rows, "base", "cand", float(cfg["mem_tol_pct"]))
        numbers = {"cells": m["cells"], "reps": m["reps"]}
        for r in m["regressions"]:
            reasons.append(
                f"{r['metric']}: {r['base_gib']} -> {r['cand_gib']} GiB (+{r['delta_gib']}, limit {r['limit_gib']})"
            )
        reasons += [f"missing {x}" for x in m["missing"]]
    return _finish(ctx, StageResult("memory", not reasons, reasons, numbers))


def _memory_valid(path: Path):
    rows = read_jsonl(path)
    if not any(r.get("complete") is True for r in rows):
        return False, "memory_ab did not finish (no complete=true)"
    if not any(r.get("step") == "idle-after" and "footprint_gib" in r for r in rows):
        return False, "no idle-after record"
    return True, ""


STAGE_FUNCS = {
    "preflight": stage_preflight,
    "smoke": stage_smoke,
    "identity": stage_identity,
    "apc": stage_apc,
    "quality": stage_quality,
    "speed": stage_speed,
    "memory": stage_memory,
}
