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
from .snapshot import stage_snapshot

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


def client_routes_valid(path: Path):
    import json

    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return False, f"missing/invalid route evidence: {exc}"
    required = {
        "agent-custom-tools",
        "agent-shell-search",
        "agent-documents-citations",
        "agent-anthropic-client-tools",
        "agent-continuous-usage",
        "agent-template-props",
        "agent-http-video",
    }
    rows = data.get("checks", {})
    seen = {key.split("@")[0] for key in rows}
    if (
        data.get("complete") is not True
        or data.get("pass") is not True
        or required - seen
    ):
        return False, str(
            data.get("failures") or sorted(required - seen) or "incomplete routes"
        )
    if any(row.get("status") not in ("pass", "skip") for row in rows.values()):
        return False, "a route check failed"
    return True, ""


def stage_client_compat(ctx: Ctx) -> StageResult:
    import json

    stage = "client_compat"
    device = ctx.suite.get("client_compat_device", "m3")
    jobs, numbers, reasons = (
        [],
        {"device": device, "candidate_commit": ctx.cand.key},
        [],
    )
    tree_sha = subprocess.check_output(
        ["git", "-C", str(ctx.cand.path), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    numbers["tree_sha"] = tree_sha
    # One successful pilot before the second model; both use the commit-pinned tree.
    for model in ("Qwen3.5-0.8B-MLX-bf16", "Qwen2.5-3B-Instruct-4bit"):
        cell = Cell(
            stage,
            model,
            [
                ctx.py,
                str(ctx.cand.path / "scripts/research/client_compat_routes.py"),
                "--model",
                str(Path("/Volumes/P5Plus/models") / model),
                "--tree-sha",
                tree_sha,
                "--device",
                device,
                "--out",
                "{out}",
            ],
            mem_gb=8,
            timeout_min=35,
            stall_min=10,
            validate=client_routes_valid,
            device=device,
            cwd=ctx.cand.path,
        )
        result = ctx.exe.run_cells([cell])[cell.key]
        jobs.append(result.job)
        if result.evidence and result.evidence.exists():
            data = json.loads(result.evidence.read_text())
            numbers[model] = {
                "checks": data.get("checks"),
                "tree_sha": data.get("tree_sha"),
                "device": data.get("device"),
            }
            if data.get("tree_sha") != tree_sha or data.get("device") != device:
                reasons.append(f"{model}: wrong source tree/device")
        if not result.ok:
            reasons.append(f"{model}: {result.reason}")
        if reasons:
            break
    return _finish(ctx, StageResult(stage, not reasons, reasons, numbers, jobs))


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
                        timeout_min=cfg.get("identity_timeout_min")
                        or _decode_est_min(
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


def _spec_depth_valid(rows: list, mode: str) -> tuple[bool, str]:
    specs = [(row.get("xy") or {}).get("speculative") for row in rows]
    specs = [spec for spec in specs if isinstance(spec, dict) and spec.get("drafted")]
    if not specs:
        return False, "no speculative draft telemetry recorded"
    for spec in specs:
        if spec.get("mode") != mode or not spec.get("rounds"):
            return False, "speculative mode or rounds missing"
        depths = spec.get("per_depth") or []
        if not depths or any(not isinstance(row, dict) for row in depths):
            return False, "per-depth counters missing"
        if any(
            row.get("position") != pos
            or not isinstance(row.get("drafted"), int)
            or not isinstance(row.get("accepted"), int)
            or not 0 <= row["accepted"] <= row["drafted"]
            for pos, row in enumerate(depths)
        ):
            return False, "invalid per-depth counter"
        if sum(row["drafted"] for row in depths) != spec["drafted"] or sum(
            row["accepted"] for row in depths
        ) != spec.get("accepted"):
            return False, "per-depth counters do not sum to request totals"
    return True, ""


def stage_identity(ctx: Ctx) -> StageResult:
    _check_prompts(ctx.suite["ctx"], ctx.suite["kinds"])
    res = ctx.exe.run_cells(_identity_cells(ctx))
    reasons = _failed_cells(res)
    numbers: dict = {}
    if not reasons:
        for mode in ctx.suite.get("spec_modes") or ["default"]:
            tag = "" if mode == "default" else f"[{mode}] "
            base, cand = _rows_for(res, "base", mode), _rows_for(res, "cand", mode)
            if ctx.suite.get("require_spec_depth"):
                depth_ok, why = _spec_depth_valid(cand, mode)
                numbers[f"{tag}per_depth".strip()] = depth_ok
                if not depth_ok:
                    reasons.append(f"{tag}{why}")
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
def _speed_cells(ctx: Ctx, rep: int, units: list, only: set | None = None) -> list:
    """The cells of one rep, arms interleaved (A B). `only` limits a confirmation rep to the
    regressing (ctx, kind) cells; its keys always carry the cell tag."""
    cfg = ctx.suite
    cells = []
    for arm in ("base", "cand"):
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
                    if arm == "base" and cfg.get("reuse_base_speed") and only is None
                    else "",
                )
            )
    return cells


def _speed_reason(r: dict) -> str:
    return (
        f"{r['kind']}@{r['ctx']} {r['metric']}: {r['base_median']} -> {r['cand_median']} "
        f"({r['delta_pct']:+.1f}%, limit {r['limit_pct']:.1f}%)"
    )


def stage_speed(ctx: Ctx) -> StageResult:
    cfg = ctx.suite
    _check_prompts(cfg["ctx"], cfg["kinds"])
    cells = []
    units = _units(ctx)
    for rep in range(int(cfg["reps"])):
        cells += _speed_cells(ctx, rep, units)
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

        tol = float(cfg["speed_tol_pct"])
        base = [arm_rep("base", i) for i in range(reps)]
        cand = [arm_rep("cand", i) for i in range(reps)]
        sp = analyze.speed_compare(base, cand, tol)
        numbers = {k: sp[k] for k in ("cells", "reps", "tol_pct")}
        k_extra = int(cfg.get("speed_confirm_reps", 2))
        if sp["regressions"] and not sp["missing"] and k_extra > 0:
            # A stalled GPU request can spike both cand reps and look like a regression: rerun
            # only the regressing cells for K more paired reps, judge on all with robust stats.
            sus = sorted({(r["ctx"], r["kind"]) for r in sp["regressions"]})
            cunits = [(f"c{c}{k}", [c], [k]) for c, k in sus]
            ccells = []
            for rep in range(reps, reps + k_extra):
                ccells += _speed_cells(ctx, rep, cunits, set(sus))
            cres = ctx.exe.run_cells(ccells)
            reasons = _failed_cells(cres)
            if not reasons:
                allres = {**res, **cres}
                res = allres
                base = [arm_rep("base", i) for i in range(reps + k_extra)]
                cand = [arm_rep("cand", i) for i in range(reps + k_extra)]
                cf = analyze.speed_compare(base, cand, tol, robust=True, only=sus)
                keep = [c for c in sp["cells"] if (c["ctx"], c["kind"]) not in set(sus)]
                numbers = {
                    "cells": keep + cf["cells"],
                    "reps": reps,
                    "tol_pct": tol,
                    "confirmation": {
                        "extra_reps": k_extra,
                        "cells": [list(x) for x in sus],
                        "initial_verdict": "regression",
                        "initial_regressions": sp["regressions"],
                        "confirmed_verdict": "regression"
                        if cf["regressions"]
                        else "neutral",
                        "confirmed_cells": cf["cells"],
                    },
                }
                sp = {
                    "regressions": cf["regressions"],
                    "missing": cf["missing"],
                    "cells": numbers["cells"],
                }
        for r in sp["regressions"]:
            reasons.append(_speed_reason(r))
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
                        "18993",
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


def _multimodal_valid(
    path: Path, sizes: list, require_hit: bool, require_anthropic=True
):
    rows = read_jsonl(path)
    if not rows or not rows[-1].get("complete"):
        return False, "incomplete multimodal evidence"
    requests = {
        (r.get("size"), r.get("kind")): r for r in rows if r.get("event") == "request"
    }
    expected = {
        (n, k)
        for n in sizes
        for k in ("cold", "warm", "turn2-hit", "turn2-miss", "other-image")
    }
    if require_anthropic:
        expected.update(
            (0, k)
            for k in (
                "anthropic-cold",
                "anthropic-warm",
                "anthropic-turn2-hit",
                "anthropic-turn2-miss",
            )
        )
    if set(requests) != expected:
        return False, "missing multimodal requests"
    controls = [(n, [("cold", "warm"), ("turn2-miss", "turn2-hit")]) for n in sizes]
    if require_anthropic:
        controls += [
            (
                0,
                [
                    ("anthropic-cold", "anthropic-warm"),
                    ("anthropic-turn2-miss", "anthropic-turn2-hit"),
                ],
            )
        ]
    for n, pairs in controls:
        for miss, hit in pairs:
            a, b = requests[n, miss], requests[n, hit]
            if not a.get("ids") or a["ids"] != b.get("ids") or a.get("cached") != 0:
                return False, "raw token hit/miss identity failed"
            skipped = hit == "turn2-hit" and b.get("restore_skipped") is True
            if require_hit and b.get("cached", 0) <= 0 and not skipped:
                return False, "media APC not engaged"
    if any(requests[n, "other-image"].get("cached") != 0 for n in sizes):
        return False, "different image reused state"
    return True, ""


def _multimodal_quality_valid(path, sizes, n):
    ok, reason = _multimodal_valid(path, sizes, True)
    if not ok:
        return ok, reason
    rows = read_jsonl(path)
    items = [r for r in rows if r.get("event") == "parity_item"]
    if len(items) != n or {r.get("i") for r in items} != set(range(n)):
        return False, "incomplete paired image set"
    if any(
        not r.get("cached")
        or not r.get("cold_ids")
        or r["cold_ids"] != r.get("hit_ids")
        for r in items
    ):
        return False, "paired image raw-ID drift"
    summary = next((r for r in rows if r.get("event") == "quality"), {})
    scores = summary.get("scores", {})
    counts = {
        "cold": sum(bool(r.get("cold_correct")) for r in items),
        "hit": sum(bool(r.get("hit_correct")) for r in items),
    }
    if (
        summary.get("n") != n
        or any(scores.get(k) != v for k, v in counts.items())
        or abs(counts["cold"] - counts["hit"]) > 1
    ):
        return False, "invalid paired quality scores"
    return True, ""


def _multimodal_speed(numbers, reps, tolerance):
    def timing(arm, rep):
        phases = {"cold": "cold", "warm": "warm", "turn2-hit": "turn2"}
        return [
            dict(
                part="decode",
                ctx=r["size"],
                kind="media",
                phase=phases[r["kind"]],
                ttft_s=r["ttft_s"],
                dec_tps=1.0,
            )
            for r in numbers[f"{arm}-r{rep}"]
            if r["kind"] in phases
        ]

    return analyze.speed_compare(
        [timing("base", i) for i in range(reps)],
        [timing("cand", i) for i in range(reps)],
        tolerance,
        robust=True,
    )


def stage_multimodal(ctx: Ctx) -> StageResult:
    """Pinned media sessions: raw IDs, pixel isolation, short + long TTFT."""

    def cell(arm, rep):
        key = f"{arm}-r{rep}"
        scratch = ctx.run.path / "media" / key
        scratch.mkdir(parents=True, exist_ok=True)
        env = ctx.arm_env(
            arm,
            {
                "YUNSHU_VLM_APC_DISK": "0",
                "YUNSHU_VLM_APC_WARM": "off",
                "YUNSHU_KV_PRECISION": "bf16",
                "YUNSHU_VLM_DRAFT": "off",
                "YUNSHU_MEDIA_DIR": str(scratch),
                "HF_HUB_OFFLINE": "1",
            },
        )
        argv = ["env", f"PYTHONPATH={ctx.tree(arm).path / 'python'}"]
        argv += [f"{k}={v}" for k, v in env.items()]
        remote = (
            "gemma-4-e2b" in ctx.model
            and int(ctx.suite.get("reps", 3)) == 1
            and ctx.env.get("GPUQ_DEVICE") != "m5"
        )
        script = ctx.cand.path / "scripts/research/multimodal_apc.py"
        if remote:
            # gpuq snapshots cwd, not arbitrary external pinned trees. Run
            # the committed common harness in each arm's own snapshot; the
            # old arm need not contain this new measurement script.
            source = script.read_text()
            compile(source, str(script), "exec")  # CPU preflight before submit
            entry = [
                ctx.py,
                "-c",
                "import sys; __file__='scripts/research/multimodal_apc.py'; sys.path.insert(0, 'scripts/research'); "
                + f"exec(compile({source!r}, 'multimodal_apc.py', 'exec'))",
            ]
        else:
            entry = [ctx.py, str(script)]
        argv += [
            *entry,
            "--model",
            ctx.model,
            "--out",
            "{out}",
            "--sizes",
        ]
        argv += [str(n) for n in ctx.suite["ctx"]]
        if ctx.env.get("APC_PROBE_IMAGE_AUDIO") == "1":
            argv += ["--image-audio"]
        if arm == "cand":
            argv += ["--require-hit"]
        else:
            argv += ["--skip-anthropic"]
        cells = [
            Cell(
                "multimodal",
                key,
                argv,
                mem_gb=ctx.mem_gb,
                quiet=int(ctx.suite.get("reps", 3)) >= 3,
                timeout_min=20,
                device="m3" if remote else "m5",
                cwd=ctx.tree(arm).path if remote else None,
                validate=lambda path, hit=arm == "cand": _multimodal_valid(
                    path, ctx.suite["ctx"], hit, require_anthropic=hit
                ),
            )
        ]
        return cells[0]

    parity_n = (
        int(ctx.suite.get("mmlu_n", 200)) if ctx.suite.get("media_quality_items") else 0
    )
    quality_result = None
    if parity_n:
        quality_cell = cell("cand", "quality")
        quality_cell.argv += ["--parity-items", str(parity_n)]
        quality_cell.quiet = False
        quality_cell.validate = lambda path: _multimodal_quality_valid(
            path, ctx.suite["ctx"], parity_n
        )
        quality_result = ctx.exe.run_cells([quality_cell])[quality_cell.key]
        if not quality_result.ok:
            return _finish(
                ctx, StageResult("multimodal", False, [quality_result.reason])
            )
    cells = [
        cell(arm, rep)
        for rep in range(int(ctx.suite.get("reps", 3)))
        for arm in ("base", "cand")
    ]
    results = ctx.exe.run_cells(cells)
    reasons = _failed_cells(results)
    numbers = {}
    if quality_result is not None:
        numbers["paired_quality"] = next(
            r
            for r in read_jsonl(quality_result.evidence)
            if r.get("event") == "quality"
        )
    if not reasons:
        for key, result in results.items():
            rows = [
                r for r in read_jsonl(result.evidence) if r.get("event") == "request"
            ]
            numbers[key] = [
                {k: r[k] for k in ("kind", "size", "pt", "cached", "ttft_s", "sha")}
                for r in rows
            ]
        for rep in range(int(ctx.suite.get("reps", 3))):
            b, c = numbers[f"base-r{rep}"], numbers[f"cand-r{rep}"]
            c = [r for r in c if not r["kind"].startswith("anthropic-")]
            if [(r["kind"], r["size"], r["sha"]) for r in b] != [
                (r["kind"], r["size"], r["sha"]) for r in c
            ]:
                reasons.append(f"raw token identity differs base/cand rep {rep}")
            br = read_jsonl(results[f"base-r{rep}"].evidence)
            cr = read_jsonl(results[f"cand-r{rep}"].evidence)
            bv = next(
                (r.get("versions") for r in br if r.get("event") == "engaged"), None
            )
            cv = next(
                (r.get("versions") for r in cr if r.get("event") == "engaged"), None
            )
            if not bv or bv != cv:
                reasons.append(
                    f"dependencies changed across paired arms rep {rep}: {bv} / {cv}"
                )
            bd, cd = {r.get("device") for r in br}, {r.get("device") for r in cr}
            if len(bd) != 1 or bd != cd or not bd <= {"m3", "m5"}:
                reasons.append(f"mixed or missing devices rep {rep}: {bd} / {cd}")
    if not reasons and int(ctx.suite.get("reps", 3)) >= 3:
        speed = _multimodal_speed(
            numbers, int(ctx.suite["reps"]), float(ctx.suite["speed_tol_pct"])
        )
        numbers["speed"] = speed
        reasons.extend(_speed_reason(r) for r in speed["regressions"])
        reasons.extend(speed["missing"])
    return _finish(ctx, StageResult("multimodal", not reasons, reasons, numbers))


def stage_modelprobe(ctx: Ctx) -> StageResult:
    """Candidate-only absolute reserve/serving pilot; no relative-model verdict."""
    script = ctx.cand.path / "scripts/research/bigmoe_probe.py"
    cell = Cell(
        "modelprobe",
        "cand-pilot",
        [
            ctx.py,
            str(script),
            "--model",
            ctx.model,
            "--src",
            str(ctx.cand.path / "python"),
            "--out",
            "{out}",
        ],
        mem_gb=ctx.mem_gb,
        timeout_min=15,
        stall_min=8,
    )
    result = ctx.exe.run_cells([cell])
    reasons = _failed_cells(result)
    numbers = {
        "scope": "candidate-only absolute reserve and generic serving; not model superiority"
    }
    evidence = result["cand-pilot"].evidence
    if evidence:
        rows = read_jsonl(evidence)
        if rows:
            numbers.update(rows[-1])
    return _finish(ctx, StageResult("modelprobe", not reasons, reasons, numbers))


def _websearch_valid(path: Path):
    rows = read_jsonl(path)
    if not rows or rows[-1].get("complete") is not True:
        return False, "missing final complete record"
    checks = [r for r in rows if r.get("check")]
    if not checks or any(r.get("pass") is not True for r in checks):
        return False, "missing or failed web tool checks"
    return True, ""


def stage_websearch(ctx: Ctx) -> StageResult:
    # Candidate harness checks old search contract on base; new page actions on candidate.
    script = ctx.cand.path / "scripts/research/websearch_probe.py"
    cells = []
    for arm in ("base", "cand"):
        cells.append(
            Cell(
                "websearch",
                arm,
                [
                    "env",
                    *[
                        f"{key}={value}"
                        for key, value in sorted(ctx.arm_env(arm).items())
                    ],
                    "PYTHONPATH="
                    + str(ctx.tree(arm).path / "python")
                    + os.pathsep
                    + os.environ.get("PYTHONPATH", ""),
                    ctx.py,
                    str(script),
                    "--src",
                    str(ctx.tree(arm).path / "python"),
                    "--model",
                    ctx.model,
                    "--out",
                    "{out}",
                    *(
                        ["--baseline"]
                        if arm == "base"
                        else [
                            "--embedding-model",
                            "/Volumes/P5Plus/models/Qwen3-Embedding-0.6B",
                            "--eval-snapshot",
                            str(
                                ctx.cand.path
                                / "scripts/research/data/websearch_adversarial.jsonl"
                            ),
                        ]
                    ),
                ],
                mem_gb=14,
                timeout_min=8,
                stall_min=4,
                validate=_websearch_valid,
                device="m5",
            )
        )
    results = ctx.exe.run_cells(cells)
    reasons = _failed_cells(results)
    numbers = {
        key: read_jsonl(r.evidence) if r.evidence else [] for key, r in results.items()
    }
    return _finish(ctx, StageResult("websearch", not reasons, reasons, numbers))


def stage_rerank(ctx: Ctx) -> StageResult:
    """New capabilities: candidate HTTP output vs independent Transformers oracle.

    The baseline commit is recorded by yv, but cannot serve these trained heads;
    the numerical baseline for this stage is the original checkpoint on CPU.
    """
    import json

    tree = ctx.cand.path
    root = Path(ctx.env.get("RERANK_MODEL_ROOT", "/Volumes/P5Plus/models"))
    names = [
        "Qwen3-Reranker-0.6B",
        "bge-reranker-base",
        "ms-marco-MiniLM-L-6-v2",
        "bert-tiny-finetuned-sst2",
    ]
    import runpy

    cases = runpy.run_path(str(tree / "scripts/research/rerank_parity.py"))
    pairs, texts = cases["PAIRS"], cases["TEXTS"]
    numbers, reasons = {}, []
    for name in names:
        model = root / name
        if not (model / "config.json").exists():
            reasons.append(f"missing checkpoint {model}")
            break
        items = ctx.run.path / f"{name}.items.json"
        reference = ctx.run.path / f"{name}.reference.json"
        items.write_text(
            json.dumps(
                {"texts": texts} if name.startswith("bert-tiny") else {"pairs": pairs}
            )
        )
        # CPU oracle runs before occupying a GPU slot and stays under nice 15.
        if not reference.exists():
            result = subprocess.run(
                [
                    "nice",
                    "-n",
                    "15",
                    ctx.py,
                    str(tree / "scripts/research/rerank_reference.py"),
                    str(model),
                    "--items",
                    str(items),
                    "--out",
                    str(reference),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode:
                reasons.append(f"{name} oracle: {result.stderr[-1000:]}")
                break
        cell = Cell(
            "rerank",
            name,
            [
                "env",
                f"PYTHONPATH={tree / 'python'}",
                "HF_HUB_OFFLINE=1",
                ctx.py,
                str(tree / "scripts/research/rerank_parity.py"),
                "--model",
                str(model),
                "--reference",
                str(reference),
                "--out",
                "{out}",
            ],
            mem_gb=6,
            timeout_min=5,
            stall_min=3,
            priority=-1,
            device="m5",
        )
        results = ctx.exe.run_cells([cell])
        reasons += _failed_cells(results)
        if reasons:
            break
        rows = read_jsonl(results[name].evidence)
        numbers[name] = rows[-1]
        if not rows[-1].get("passed"):
            reasons.append(f"{name} oracle parity failed")
            break
    return _finish(ctx, StageResult("rerank", not reasons, reasons, numbers))


def stage_embedding(ctx: Ctx) -> StageResult:
    """Published-format loader and independent same-device upstream invocation."""
    reference = ctx.env.get(
        "EMBEDDING_REFERENCE",
        "/Volumes/P5Plus/yunshu-build/codex/priorfix/gemma-ref/ref.json",
    )
    root = Path(ctx.env.get("EMBEDDING_MODEL_ROOT", "/Volumes/P5Plus/models"))
    reasons, numbers = [], {}
    for name in (
        "embeddinggemma-2-bf16",
        "embeddinggemma-2-4bit",
        "embeddinggemma-2-bf16-multishard",
    ):
        cell = Cell(
            "embedding",
            name,
            [
                "env",
                f"PYTHONPATH={ctx.cand.path / 'python'}",
                "HF_HUB_OFFLINE=1",
                ctx.py,
                str(ctx.cand.path / "scripts/research/priorfix_embedding_parity.py"),
                "--model",
                str(root / name),
                "--reference",
                reference,
                "--out",
                "{out}",
            ],
            mem_gb=8,
            timeout_min=10,
            stall_min=5,
            priority=-1,
            device="m5",
        )
        results = ctx.exe.run_cells([cell])
        reasons += _failed_cells(results)
        if reasons:
            break
        rows = read_jsonl(results[name].evidence)
        numbers[name] = rows[-1]
        if not rows[-1].get("passed"):
            reasons.append(f"{name} parity failed")
            break
    return _finish(ctx, StageResult("embedding", not reasons, reasons, numbers))


def stage_priorart(ctx: Ctx) -> StageResult:
    reasons, numbers = [], {}
    kinds = ctx.env.get("PRIORART_KINDS", "retrieval,classifier,diffusion,omni").split(
        ","
    )
    for kind in kinds:
        prefix = str(ctx.cand.path / "python")
        if kind.startswith("diffusion") or kind == "capabilities":
            prefix += ":/Volumes/P5Plus/yunshu-build/codex/priorfix/mflux-deps"
        extra = []
        if kind in ("retrieval", "capabilities"):
            import urllib.request

            reference = ctx.run.path / "aperepel-cfe20b0-server.py"
            if not reference.exists():
                url = "https://raw.githubusercontent.com/aperepel/mlx-rerank/cfe20b0b0e2505240be91dbcf6e5575b8a8d7388/server.py"
                with urllib.request.urlopen(url, timeout=30) as response:
                    reference.write_bytes(response.read())
            compile(reference.read_text(), str(reference), "exec")
            extra = ["--rerank-reference", str(reference)]
        script = ctx.cand.path / "scripts/research/priorfix_runtime_parity.py"
        mode_args = ["--kind", kind]
        if kind == "diffusion-timing":
            script = ctx.cand.path / "scripts/research/priorfix_diffusion_timing.py"
            mode_args = []
        elif kind == "capabilities":
            script = ctx.cand.path / "scripts/research/priorfix_capabilities.py"
            mode_args = [
                "--model-root",
                ctx.env.get("EMBEDDING_MODEL_ROOT", "/Volumes/P5Plus/models"),
                "--reference",
                ctx.env.get(
                    "EMBEDDING_REFERENCE",
                    "/Volumes/P5Plus/yunshu-build/codex/priorfix/gemma-ref/ref.json",
                ),
            ]
        cell = Cell(
            "priorart",
            kind,
            [
                "env",
                f"PYTHONPATH={prefix}",
                "HF_HUB_OFFLINE=1",
                ctx.py,
                str(script),
                *mode_args,
                *extra,
                "--out",
                "{out}",
            ],
            # The floating two-runtime image pilot peaked at 57.3 (M5 gpuq RSS).
            mem_gb=64 if kind.startswith("diffusion") or kind == "capabilities" else 8,
            quiet=kind == "diffusion-timing",
            timeout_min=10,
            stall_min=5,
            priority=-1,
            device="m5",
        )
        results = ctx.exe.run_cells([cell])
        reasons += _failed_cells(results)
        if reasons:
            break
        rows = read_jsonl(results[kind].evidence)
        numbers[kind] = rows[-1]
        if not rows[-1].get("passed"):
            reasons.append(f"{kind} parity failed")
            break
    return _finish(ctx, StageResult("priorart", not reasons, reasons, numbers))


def stage_evals(ctx: Ctx) -> StageResult:
    """SDK shape + local engine route coverage; CPU probe tests run in preflight."""
    tree = ctx.tree("cand").path
    cell = Cell(
        "evals",
        "sdk-routes",
        [
            "env",
            f"PYTHONPATH={tree / 'python'}",
            "HF_HUB_OFFLINE=1",
            ctx.py,
            str(tree / "scripts/research/evals_verify.py"),
            "--model",
            ctx.model,
            "--src",
            str(tree / "python"),
            "--out",
            "{out}",
        ],
        mem_gb=6,
        timeout_min=10,
        stall_min=5,
        priority=-1,
        device="m5",
    )
    results = ctx.exe.run_cells([cell])
    reasons = _failed_cells(results)
    numbers = {}
    if not reasons:
        rows = read_jsonl(results["sdk-routes"].evidence)
        numbers = rows[-1]
        if not numbers.get("passed"):
            reasons.append("Evals SDK route smoke failed")
    return _finish(ctx, StageResult("evals", not reasons, reasons, numbers))


def stage_tavily(ctx: Ctx) -> StageResult:
    cells = [
        Cell(
            "tavily",
            arm,
            [
                "env",
                *[f"{key}={value}" for key, value in sorted(ctx.arm_env(arm).items())],
                "PYTHONPATH=" + str(ctx.tree(arm).path / "python"),
                ctx.py,
                str(ctx.cand.path / "scripts/research/tavily_probe.py"),
                "--src",
                str(ctx.tree(arm).path / "python"),
                "--model",
                ctx.model,
                "--out",
                "{out}",
                *(["--baseline"] if arm == "base" else []),
            ],
            mem_gb=ctx.mem_gb or 14,
            timeout_min=8,
            stall_min=4,
            validate=_websearch_valid,
            device="m5",
        )
        for arm in ("base", "cand")
    ]
    results = ctx.exe.run_cells(cells)
    reasons = _failed_cells(results)
    return _finish(
        ctx,
        StageResult(
            "tavily",
            not reasons,
            reasons,
            {
                key: read_jsonl(result.evidence) if result.evidence else []
                for key, result in results.items()
            },
        ),
    )


def _searchrank_valid(path):
    rows = read_jsonl(path)
    arms = {row.get("backend"): row for row in rows if "backend" in row}
    ok = bool(
        rows
        and rows[-1].get("complete") is True
        and all(
            arms.get(name, {}).get("status") == "ok"
            for name in ("cpu", "coreml_cpu_ne", "mlx")
        )
    )
    return ok, "All CPU/Core ML/MLX backends must finish with finite scores"


def stage_searchrank(ctx: Ctx) -> StageResult:
    interpreter = ctx.env.get("SEARCHRANK_PY", ctx.py)
    cell = Cell(
        "searchrank",
        "backends",
        [
            "env",
            "PYTHONPATH=" + str(ctx.cand.path / "python"),
            interpreter,
            str(ctx.cand.path / "scripts/research/searchrank_backends.py"),
            "--model",
            ctx.model,
            "--out",
            "{out}",
            "--cache",
            str(ctx.run.path / "coreml-cache"),
        ],
        mem_gb=4,
        timeout_min=10,
        stall_min=5,
        quiet=True,
        validate=_searchrank_valid,
        device="m5",
    )
    results = ctx.exe.run_cells([cell])
    reasons = _failed_cells(results)
    return _finish(
        ctx,
        StageResult(
            "searchrank",
            not reasons,
            reasons,
            {
                key: read_jsonl(result.evidence) if result.evidence else []
                for key, result in results.items()
            },
        ),
    )


def _console_validate(path):
    rows = read_jsonl(path)
    if not rows or not rows[-1].get("complete") or rows[-1].get("failures"):
        return False, "console route probe incomplete or failed"
    required = {
        "console_registration_cancel",
        "console_host_latency",
        "stream_latency",
        "single_stream_latency",
        "console_backend_gaps",
        "history_restart",
    }
    checks = rows[-1].get("checks", {})
    # gpuq adds device / execution_device / remote_host to every record: require the
    # four checks, each PASS, rather than an exact key set.
    missing = sorted(k for k in required if checks.get(k) != "PASS")
    if missing:
        return False, "console route checks missing or failing: " + ", ".join(missing)
    return True, ""


def stage_console(ctx: Ctx) -> StageResult:
    """Single-node console API correctness on a pinned candidate; no timing verdict."""

    validate = _console_validate

    cell = Cell(
        "console",
        "cand",
        [
            ctx.py,
            str(ctx.cand.path / "scripts/research/consolefeat_routes.py"),
            "--model",
            ctx.model,
            "--src",
            str(ctx.cand.path / "python"),
            "--out",
            "{out}",
        ],
        mem_gb=12,
        timeout_min=10,
        quiet=False,
        validate=validate,
    )
    result = ctx.exe.run_cells([cell])["cand"]
    numbers = read_jsonl(result.evidence)[-1] if result.ok and result.evidence else {}
    return _finish(
        ctx,
        StageResult(
            "console", result.ok, [] if result.ok else [result.reason], numbers
        ),
    )


def stage_respfeat(ctx: Ctx) -> StageResult:
    """One bounded M5 tiny server covers client-executed tools and transports."""
    import json

    tree_sha = subprocess.check_output(
        ["git", "-C", str(ctx.cand.path), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    argv = [
        ctx.py,
        str(ctx.cand.path / "scripts/research/respfeat_routes.py"),
        "--model",
        ctx.model,
        "--tree-sha",
        tree_sha,
        "--device",
        "m5",
        "--out",
        "{out}",
    ]
    if ctx.cand_env.get("RESPFEAT_EXTRA_PYTHONPATH"):
        argv += ["--extra-pythonpath", ctx.cand_env["RESPFEAT_EXTRA_PYTHONPATH"]]

    def valid(path):
        from scripts.research.respfeat_routes import judge

        try:
            return judge(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            return False, str(exc)

    cell = Cell(
        "respfeat",
        "tiny-routes",
        argv,
        mem_gb=8,
        timeout_min=10,
        validate=valid,
        device="m5",
        cwd=ctx.cand.path,
        retries=0,
    )
    result = ctx.exe.run_cells([cell])[cell.key]
    evidence = json.loads(result.evidence.read_text()) if result.evidence else {}
    return _finish(
        ctx,
        StageResult(
            "respfeat",
            result.ok,
            [] if result.ok else [result.reason],
            {"device": "m5", "candidate_commit": ctx.cand.key, "evidence": evidence},
            [result.job],
        ),
    )


def stage_telemetry(ctx: Ctx, *, pilot: bool = False) -> StageResult:
    """Unprivileged sensor plausibility + request receipt on the pinned candidate."""

    def validate(path):
        rows = read_jsonl(path)
        if not rows or rows[-1].get("complete") is not True:
            return False, "telemetry probe incomplete"
        summary = rows[-1].get("summary", {})
        if (
            not summary.get("gpu_peak_watts")
            or not summary.get("gpu_mhz_max")
            or not summary.get("die_max_c")
        ):
            return False, "missing power, frequency or temperature evidence"
        return True, ""

    name = "telemetry-tiny" if pilot else "telemetry"
    model = "/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16" if pilot else ctx.model
    big = False if pilot else ctx.big
    cell = Cell(
        name,
        "cand",
        [
            "env",
            f"TFB_YUNSHU_SRC={ctx.cand.path / 'python'}",
            f"TFB_OUT={ctx.run.path / 'tfb' / name}",
            ctx.py,
            str(ctx.cand.path / "scripts/research/telemetry_probe.py"),
            "--model",
            model,
            "--draft",
            "mtp" if big else "off",
            "--tokens",
            "512" if big else "2048",
            "--out",
            "{out}",
        ],
        mem_gb=14 if pilot else ctx.mem_gb,
        timeout_min=10,
        quiet=False,
        validate=validate,
    )
    result = ctx.exe.run_cells([cell])["cand"]
    numbers = (
        read_jsonl(result.evidence)[-1].get("summary", {})
        if result.ok and result.evidence
        else {}
    )
    return _finish(
        ctx,
        StageResult(name, result.ok, [] if result.ok else [result.reason], numbers),
    )


STAGE_FUNCS = {
    "snapshot": stage_snapshot,
    "console": stage_console,
    "telemetry": stage_telemetry,
    "telemetry-tiny": lambda ctx: stage_telemetry(ctx, pilot=True),
    "respfeat": stage_respfeat,
    "priorart": stage_priorart,
    "embedding": stage_embedding,
    "evals": stage_evals,
    "websearch": stage_websearch,
    "tavily": stage_tavily,
    "searchrank": stage_searchrank,
    "rerank": stage_rerank,
    "preflight": stage_preflight,
    "smoke": stage_smoke,
    "client_compat": stage_client_compat,
    "identity": stage_identity,
    "apc": stage_apc,
    "quality": stage_quality,
    "speed": stage_speed,
    "memory": stage_memory,
    "longqa": stage_longqa,
    "conc": stage_conc,
    "multimodal": stage_multimodal,
    "modelprobe": stage_modelprobe,
}
