"""The verification ladder: one function per stage, each returns a StageResult.

A stage writes its final `stage_complete` record only when its analysis finished
(fail closed: missing evidence = fail). Cells are gpuq jobs run by execute.Executor.
"""

from __future__ import annotations

import hashlib
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
from .core import sha as _sha
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


def _units(ctx: Ctx) -> list:
    """Server cells of the decode-style stages: [(tag, ctxs, kinds)]. A long suite splits per
    (ctx, kind) so every job stays under the 20 minute queue limit."""
    cfg = ctx.suite
    if not cfg.get("split_cells"):
        return [("", list(cfg["ctx"]), list(cfg["kinds"]))]
    return [(f"c{c}{k}", [c], [k]) for c in cfg["ctx"] for k in cfg["kinds"]]


def _dec_args(ctx: Ctx) -> list:
    out = ["--decode-tokens", str(int(ctx.suite.get("decode_tokens", 256)))]
    if ctx.suite.get("long_ask"):
        out.append("--long-ask")
    if ctx.suite.get("turn2_tokens"):
        out += ["--turn2-tokens", str(int(ctx.suite["turn2_tokens"]))]
    return out


def _unit_args(ctxs: list, rep: int = 0) -> list:
    return ["--part", "decode", "--rep", str(rep)] + [
        x for c in ctxs for x in ("--only-ctx", str(c))
    ]


def _kind_args(kinds: list, all_kinds: list) -> list:
    return [x for k in kinds for x in ("--only-kind", k)] if kinds != all_kinds else []


def _decode_est_min(
    ctx: Ctx, ctxs: list, kinds: int, n_dec: int = 256, n_t2: int = 0
) -> float:
    start = 2.5 if ctx.big else 0.7
    secs = 0.0
    for c in ctxs:
        per = (3 + c / 1000 * (1.4 if ctx.big else 0.15)) * 3  # cold, warm, follow-up
        secs += per * kinds
        secs += kinds * (2 * n_dec + (n_t2 or n_dec)) / (15.0 if ctx.big else 60.0)
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
                device="" if ctx.big else "any",
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
def _drafter_path() -> str:
    from .gate import local_env

    return os.environ.get("YV_DRAFTER") or local_env().get("D", "")


def mode_env(mode: str) -> dict:
    """Server env that selects a speculative-decoding method for an identity cell."""
    if mode == "default":
        return {}
    if mode in ("mtp", "off"):
        return {"YUNSHU_VLM_DRAFT": mode}
    if mode == "dflash":
        d = _drafter_path()
        if not d:
            raise InfraError(
                "--spec-modes dflash needs the drafter path (D in local.env)"
            )
        return {"YUNSHU_VLM_DRAFT": d}
    raise InfraError(f"unknown spec mode {mode!r} (default, mtp, dflash)")


def identity_cell_key(arm: str, mode: str, tag: str = "") -> str:
    # arm: base | cand | candoff; one server covers every context unless the suite splits cells
    return f"{arm}.{mode}" + (f"@{tag}" if tag else "")


def _identity_cells(ctx: Ctx) -> list:
    cfg = ctx.suite
    modes = cfg.get("spec_modes") or ["default"]
    cells = []
    for mode in modes:
        menv = mode_env(mode)
        variants = [("base", "base", menv), ("cand", "cand", menv)]
        eff = ctx.arm_env("cand", menv).get("YUNSHU_VLM_DRAFT", "").lower()
        if cfg["spec_off"] and eff not in ("off", "none"):
            variants.append(("candoff", "cand", dict(menv, YUNSHU_VLM_DRAFT="off")))
        for tag, ctxs, kinds in _units(ctx):
            for name, tree_arm, extra in variants:
                key = identity_cell_key(name, mode, tag)
                cells.append(
                    Cell(
                        "identity",
                        key,
                        _tfbench_argv(
                            ctx,
                            tree_arm,
                            "identity",
                            key,
                            _unit_args(ctxs)
                            + _kind_args(kinds, cfg["kinds"])
                            + _dec_args(ctx),
                            env=ctx.arm_env(tree_arm, extra),
                        ),
                        mem_gb=ctx.mem_gb,
                        timeout_min=_decode_est_min(
                            ctx,
                            ctxs,
                            len(kinds),
                            int(cfg.get("decode_tokens", 256)),
                            int(cfg.get("turn2_tokens", 0)),
                        ),
                        stall_min=12 if ctx.big else 4,
                        share_key=_share_key(ctx, tree_arm, extra, ctxs, tag)
                        if name == "base"
                        else "",
                    )
                )
    return cells


def _harness_hash() -> str:
    return hashlib.sha256(TFBENCH.read_bytes()).hexdigest() if TFBENCH.exists() else ""


def _share_key(ctx: Ctx, arm: str, extra: dict, c, tag: str = "") -> str:
    """Greedy digests of the base arm are deterministic: reuse them across runs. The key holds
    everything that could change them (code, env, model, harness, device) and no run path."""
    from .core import sha as _sha

    return _sha(
        "identity",
        ctx.tree(arm).key,
        ctx.arm_env(arm, extra),
        ctx.model,
        c,
        ctx.suite["kinds"],
        tag,
        _dec_args(ctx),
        hashlib.sha256(TFBENCH.read_bytes()).hexdigest() if TFBENCH.exists() else "",
        os.environ.get("GPUQ_DEVICE", "m5"),
        n=24,
    )


def _rows_for(res: dict, arm: str, mode: str) -> list:
    out = []
    for k, r in sorted(res.items()):
        if (
            (k == f"{arm}.{mode}" or k.startswith(f"{arm}.{mode}@"))
            and r.ok
            and r.evidence
        ):
            out += read_jsonl(r.evidence)
    return out


def stage_identity(ctx: Ctx) -> StageResult:
    _check_prompts(ctx.suite["ctx"], ctx.suite["kinds"])
    res = ctx.exe.run_cells(_identity_cells(ctx))
    reasons = _failed_cells(res)
    numbers: dict = {}
    if not reasons:
        for mode in ctx.suite.get("spec_modes") or ["default"]:
            tag = "" if mode == "default" else f"[{mode}] "
            base, cand = _rows_for(res, "base", mode), _rows_for(res, "cand", mode)
            cmp_ = analyze.compare_identity(base, cand, "base", "cand")
            numbers[f"{tag}base_vs_cand".strip()] = {
                "compared": cmp_["compared"],
                "mismatches": len(cmp_["mismatches"]),
            }
            if not cmp_["ok"]:
                reasons.append(f"{tag}base != cand: {cmp_['mismatches'][:3]}")
            offrows = _rows_for(res, "candoff", mode)
            if offrows:
                c2 = analyze.compare_identity(offrows, cand, "spec_off", "cand")
                numbers[f"{tag}spec_on_vs_off".strip()] = {
                    "compared": c2["compared"],
                    "mismatches": len(c2["mismatches"]),
                }
                if not c2["ok"]:
                    reasons.append(f"{tag}spec on != spec off: {c2['mismatches'][:3]}")
            numbers[f"{tag}engaged_spec_mode".strip()] = {
                "base": analyze.session_info(base).get("engaged_spec_mode"),
                "cand": analyze.session_info(cand).get("engaged_spec_mode"),
            }
    return _finish(ctx, StageResult("identity", not reasons, reasons, numbers))


# ── d. apc (analysis of the candidate's cold/warm pairs from the identity cells) ──
def stage_apc(ctx: Ctx) -> StageResult:
    rows = []
    for mode in ctx.suite.get("spec_modes") or ["default"]:
        for tag, _c, _k in _units(ctx):
            rows += read_jsonl(
                ctx.run.cell_path("identity", identity_cell_key("cand", mode, tag))
            )
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


def _quality_cache(ctx: Ctx, n: int) -> Path:
    key = _sha(
        "quality",
        ctx.base.key,
        ctx.arm_env("base"),
        ctx.model,
        n,
        ctx.suite.get("quality_max_tokens", 2048),
        hashlib.sha256(PAIRED.read_bytes()).hexdigest() if PAIRED.exists() else "",
        n=24,
    )
    return ctx.exe.cache_dir / f"quality-{key}.jsonl"


def stage_quality(ctx: Ctx, max_rounds: int = 6) -> StageResult:
    n = int(ctx.suite["mmlu_n"])
    budget = 14
    reasons: list = []
    cache = _quality_cache(ctx, n)
    base_file = _quality_arm_file(ctx, "base")
    if cache.exists() and _quality_counts(ctx, n)["base"] < n:
        base_file.parent.mkdir(parents=True, exist_ok=True)
        base_file.write_bytes(
            cache.read_bytes()
        )  # base answers from an earlier verdict
        ctx.run.append("quality", {"ev": "base_cached", "file": cache.name})
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
                        f"PAIRED_MAX_TOKENS={ctx.suite.get('quality_max_tokens', 2048)}",
                        f"PAIRED_THINKING={1 if ctx.suite.get('quality_thinking') else 0}",
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
    if counts["base"] >= n and not cache.exists():
        ctx.exe.cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(base_file.read_bytes())
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
    units = _units(ctx)
    for rep in range(int(cfg["reps"])):
        for arm in ("base", "cand"):  # A B A B ...
            for tag, ctxs, kinds in units:
                key = f"{arm}-r{rep}" + (f"@{tag}" if tag else "")
                cells.append(
                    Cell(
                        "speed",
                        key,
                        _tfbench_argv(
                            ctx,
                            arm,
                            "speed",
                            key,
                            _unit_args(ctxs, rep)
                            + _kind_args(kinds, cfg["kinds"])
                            + _dec_args(ctx),
                        ),
                        mem_gb=ctx.mem_gb,
                        timeout_min=_decode_est_min(
                            ctx,
                            ctxs,
                            len(kinds),
                            int(cfg.get("decode_tokens", 256)),
                            int(cfg.get("turn2_tokens", 0)),
                        ),
                        stall_min=12 if ctx.big else 4,
                        quiet=True,
                        share_key=_sha(
                            "speed",
                            ctx.base.key,
                            ctx.arm_env("base"),
                            ctx.model,
                            ctxs,
                            kinds,
                            _dec_args(ctx),
                            rep,
                            _harness_hash(),
                        )
                        if arm == "base" and cfg.get("reuse_base_speed")
                        else "",
                    )
                )
    res = ctx.exe.run_cells(cells)
    reasons = _failed_cells(res)
    numbers: dict = {}
    if not reasons:
        reps = int(cfg["reps"])

        def arm_rep(arm: str, i: int) -> list:
            rows: list = []
            for k, r in res.items():
                if k == f"{arm}-r{i}" or k.startswith(f"{arm}-r{i}@"):
                    rows += read_jsonl(r.evidence)
            return rows

        base = [arm_rep("base", i) for i in range(reps)]
        cand = [arm_rep("cand", i) for i in range(reps)]
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
    all_sizes = [str(s) for s in cfg["mem_sizes"]]
    groups = [[x] for x in all_sizes] if cfg.get("split_cells") else [all_sizes]
    cells = []
    for rep, sizes in [(r, g) for r in range(reps) for g in groups]:
        order = ("base", "cand") if rep % 2 == 0 else ("cand", "base")
        for arm in order:
            e = ctx.arm_env(arm)
            cells.append(
                Cell(
                    "memory",
                    f"{arm}-r{rep}" + (f"@{sizes[0]}" if len(groups) > 1 else ""),
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
                    share_key=_sha("memory", ctx.base.key, e, ctx.model, sizes, rep)
                    if arm == "base"
                    else "",
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


# ── h. longqa: long-context retrieval (needle) answers, base vs candidate ──
NEEDLE_COLD_S = {
    32768: 36.0,
    65536: 87.0,
    131072: 216.0,
}  # measured cold prefill per request


def needle_slices(ctxs: list) -> list:
    """[(ctx, lo, hi, timeout_min)]: every needle item may be a cold prefill (the prefix cache
    keeps few hybrid checkpoints), so a job takes only as many items as fit well inside the
    20 minute queue limit: start 150 s + items x cold prefill, times 1.5."""
    out = []
    for c in ctxs:
        cold = NEEDLE_COLD_S.get(c, 216.0 * c / 131072)
        per = max(1, min(10, int((20 * 60 / 1.5 - 150) // cold)))
        for lo in range(0, 10, per):
            hi = min(10, lo + per)
            out.append((c, lo, hi, min(20.0, (150 + (hi - lo) * cold) * 1.5 / 60)))
    return out


def stage_longqa(ctx: Ctx) -> StageResult:
    cfg = ctx.suite
    ctxs = list(cfg.get("needle_ctx") or [32768, 65536, 131072])
    _check_prompts(ctxs, ["prose"])
    cells = []
    for arm in ("base", "cand"):
        for c, lo, hi, tmo in needle_slices(ctxs):
            key = f"{arm}@c{c}i{lo}-{hi}"
            cells.append(
                Cell(
                    "longqa",
                    key,
                    _tfbench_argv(
                        ctx,
                        arm,
                        "longqa",
                        key,
                        ["--part", "needle", "--rep", "0", "--only-ctx", str(c)]
                        + ["--items", f"{lo}:{hi}"],
                    ),
                    mem_gb=ctx.mem_gb,
                    timeout_min=tmo if ctx.big else 6,
                    stall_min=12 if ctx.big else 4,
                    quiet=True,
                )
            )
    res = ctx.exe.run_cells(cells)
    reasons = _failed_cells(res)
    numbers: dict = {}
    if not reasons:

        def arm_rows(arm: str) -> list:
            rows: list = []
            for k, r in res.items():
                if k.startswith(f"{arm}@"):
                    rows += read_jsonl(r.evidence)
            return rows

        cmp_ = analyze.needle_compare(
            arm_rows("base"), arm_rows("cand"), int(cfg.get("quality_allowed", 1))
        )
        numbers = {
            k: cmp_[k]
            for k in ("items", "base_correct", "cand_correct", "net", "per_ctx")
        }
        if cmp_["missing"]:
            reasons += [f"missing {m}" for m in cmp_["missing"]]
        elif not cmp_["ok"]:
            reasons.append(
                f"retrieval base {cmp_['base_correct']} vs cand {cmp_['cand_correct']} "
                f"of {cmp_['items']} (allowed +-{cmp_['allowed']})"
            )
    return _finish(ctx, StageResult("longqa", not reasons, reasons, numbers))


# ── i. conc: two concurrent sub-agents, each with a warm 32K prefix + a ~2K new turn ──
def _median(xs: list) -> float:
    xs = sorted(x for x in xs if x is not None)
    return (
        xs[len(xs) // 2]
        if len(xs) % 2
        else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2
    )


def conc_summary(rows: list) -> dict:
    trials = [r for r in rows if r.get("part") == "conc32"]
    return {
        "trials": trials,
        "ttft_med": _median([t for r in trials for t in r["ttfts"]])
        if trials
        else None,
        "dec_med": _median([t for r in trials for t in r["per_req_dec"]])
        if trials
        else None,
    }


def stage_conc(ctx: Ctx) -> StageResult:
    _check_prompts([32768], ["prose", "code"])
    _check_prompts([8192], ["prose", "code"])
    cells = [
        Cell(
            "conc",
            arm,
            _tfbench_argv(ctx, arm, "conc", arm, ["--part", "conc32", "--rep", "0"]),
            mem_gb=ctx.mem_gb,
            timeout_min=20 if ctx.big else 6,
            stall_min=12 if ctx.big else 4,
            quiet=True,
        )
        for arm in ("base", "cand")
    ]
    res = ctx.exe.run_cells(cells)
    reasons = _failed_cells(res)
    numbers: dict = {}
    if not reasons:
        sb = conc_summary(read_jsonl(res["base"].evidence))
        sc = conc_summary(read_jsonl(res["cand"].evidence))
        numbers = {
            "trials": {"base": sb["trials"], "cand": sc["trials"]},
            "ttft_med": [sb["ttft_med"], sc["ttft_med"]],
            "dec_med": [sb["dec_med"], sc["dec_med"]],
        }
        tol = float(ctx.suite.get("conc_tol_pct", 10.0))
        if not sb["trials"] or not sc["trials"]:
            reasons.append("no conc32 records")
        else:
            if sc["ttft_med"] > sb["ttft_med"] * (1 + tol / 100):
                reasons.append(f"ttft {sb['ttft_med']} -> {sc['ttft_med']} s")
            if sc["dec_med"] < sb["dec_med"] * (1 - tol / 100):
                reasons.append(f"decode {sb['dec_med']} -> {sc['dec_med']} tok/s")
    return _finish(ctx, StageResult("conc", not reasons, reasons, numbers))


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
    "longqa": stage_longqa,
    "conc": stage_conc,
}
