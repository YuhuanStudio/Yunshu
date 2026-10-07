"""`yv gate`: scripts/release/gate.sh through gpuq, one job per stage, persisted per commit.

A rerun on the same commit skips stages that already passed (the 68-minute gate that failed
at its last stage used to start over). A stage passes when its check rows (results-*.jsonl)
have no FAIL / CONTENDED, at least one PASS, and the job exited 0.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from .core import (
    MAIN_CHECKOUT,
    REPO,
    RUNS,
    VENV_PY,
    Gpuq,
    InfraError,
    RunDir,
    git,
    now,
    read_jsonl,
    resolve_arm,
    write_json_atomic,
)

# stage -> (check-row prefix, timeout min, mem GB, stall min)
GATE_STAGES = {
    "install": ("install.", 25, 8, 15),
    "serve-27b": ("serve-27b.", 45, 65, 20),
    "families": ("families.", 45, 40, 20),
    "soak-mmlu": ("soak.mmlu", 90, 65, 20),
    "soak-realistic": ("soak.realistic", 60, 65, 20),
    "agent-sessions": ("agent.", 90, 65, 20),
    # not a gate.sh stage: `yv ab --suite long` (its own gpuq jobs), judged from its verdict
    "long": ("long.", 0, 0, 0),
}
DEFAULT_STAGES = [
    "install",
    "serve-27b",
    "families",
    "soak-mmlu",
    "soak-realistic",
    "agent-sessions",
    "long",
]
LONG_SUITE = "long"
GATE_PORT = "18993"


def local_env() -> dict:
    """KEY=VALUE lines of the main checkout's scripts/research/local.env (model paths etc.)."""
    out = {}
    p = MAIN_CHECKOUT / "scripts/research/local.env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k] = v.strip().strip('"')
    return out


def judge_rows(rows: list, prefix: str) -> tuple[bool, list]:
    """(passed, reasons) for the check rows of one gate stage (pure; unit-tested)."""
    mine = [r for r in rows if str(r.get("check", "")).startswith(prefix)]
    bad = [
        f"{r['check']}: {r.get('status')} {str(r.get('detail', ''))[:200]}"
        for r in mine
        if r.get("status") not in ("PASS", "SKIP")
    ]
    if not mine:
        return False, ["no check rows recorded (stage did not run to completion)"]
    if not any(r.get("status") == "PASS" for r in mine):
        bad.append("no PASS rows")
    return not bad, bad


def long_base(repo: Path) -> str:
    """The release reference for the long suite: the newest v* tag before HEAD (HEAD itself
    being tagged means the release under test, so the tag before it), else origin/main."""
    for rev in ("HEAD", "HEAD^"):
        try:
            tag = git(
                "describe", "--tags", "--abbrev=0", "--match", "v*", rev, cwd=repo
            )
        except Exception:  # noqa: BLE001 - no tag reachable
            continue
        if tag and git("rev-parse", tag + "^{commit}", cwd=repo) != git(
            "rev-parse", "HEAD", cwd=repo
        ):
            return tag
    return "origin/main"


def judge_long(verdict: dict | None, planned: list) -> tuple[bool, list]:
    """Fail closed: the long verdict must exist, be PASS / exit 0, and every planned stage PASS
    (a missing, NOT_RUN or skipped cell is a failure, never a pass). Pure; unit-tested."""
    if not verdict:
        return False, ["no verdict.json from yv ab --suite long"]
    bad = []
    if verdict.get("overall") != "PASS" or verdict.get("exit_code") != 0:
        bad.append(
            f"yv verdict {verdict.get('overall')} exit {verdict.get('exit_code')}"
            + (
                f": {verdict['infra_error'][:120]}"
                if verdict.get("infra_error")
                else ""
            )
        )
    got = {s.get("name"): s for s in verdict.get("stages", [])}
    for name in planned:
        s = got.get(name)
        if s is None:
            bad.append(f"{name}: missing from verdict")
        elif s.get("status") != "PASS":
            bad.append(
                f"{name}: {s.get('status')} {'; '.join(map(str, s.get('reasons', [])[:2]))[:160]}"
            )
    return not bad, bad


def run_long_stage(
    a, gq, log, runs, repo, priority, label_prefix="infra"
) -> tuple[bool, list, str]:
    """Run the long suite (candidate = the commit under test, base = long_base) and judge it."""
    import argparse

    from . import runner
    from .suites import parse_suite

    commit = git("rev-parse", "HEAD", cwd=repo)
    ns = argparse.Namespace(
        base=a["base"], cand=str(repo), env=[], cand_env=[], base_env=[],
        suite=LONG_SUITE, label=f"{label_prefix}-gate-long-{commit[:8]}", model=a["model"],
        model_name="", engaged=[], ctx=None, reps=None, mmlu_n=None, mem_sizes=None,
        mem_reps=None, speed_tol=None, spec_off=None, no_apc_hit_required=False,
        mem_gb=0, priority=priority,
    )  # fmt: skip
    rc = runner.run_ab(ns, gq=gq, log=log, runs=runs)
    cand = resolve_arm("cand", str(repo))
    vp = runner.run_dir_for(ns.label, cand, runs) / "verdict.json"
    try:
        v = json.loads(vp.read_text())
    except (OSError, ValueError):
        v = None
    ok, why = judge_long(v, parse_suite(LONG_SUITE)["stages"])
    if rc != 0 and ok:
        ok, why = False, [f"yv exit {rc}"]
    return ok, why, str(vp.parent)


def read_stage_rows(out_dir: Path) -> list:
    rows: list = []
    for f in sorted(out_dir.glob("results-*.jsonl")):
        rows += read_jsonl(f)
    return rows


def run_gate(
    stages: list | None = None,
    resume: bool = True,
    gq: Gpuq | None = None,
    log: Callable[[str], None] = print,
    runs: Path | None = None,
    repo: Path | None = None,
    extra_env: dict | None = None,
    priority: int = 0,
    label_prefix: str = "infra",
) -> int:
    gq = gq or Gpuq()
    repo = repo or REPO
    stages = stages or list(DEFAULT_STAGES)
    unknown = [s for s in stages if s not in GATE_STAGES]
    if unknown:
        raise InfraError(f"unknown gate stage {unknown}; {sorted(GATE_STAGES)}")
    commit = git("rev-parse", "HEAD", cwd=repo)
    if git("status", "--porcelain", "--untracked-files=no", cwd=repo):
        log(
            "warning: tracked changes in the tree; the gate record is keyed by HEAD only"
        )
    rd = RunDir((runs or RUNS) / f"gate-{commit[:12]}")
    env = dict(local_env())
    env.update(extra_env or {})
    gate_root = env.get("GATE_ROOT", os.path.expanduser("~/.cache/yunshu/gate"))
    marker = Path(gate_root) / ".yv-install-commit"
    results = []
    failed = None
    for name in stages:
        prefix, timeout, mem, stall = GATE_STAGES[name]
        prior = [r for r in rd.rows(name) if r.get("ev") == "stage_complete"]
        marker_ok = marker.exists() and marker.read_text().strip() == commit
        # the tools under $GATE_ROOT must still be this commit's install, else a pass is stale
        fresh = marker_ok or (name != "install" and "install" not in stages)
        if resume and prior and prior[-1].get("passed") and fresh:
            log(
                f"gate {name}: passed earlier on {commit[:12]} (job {prior[-1].get('job')}), skipped"
            )
            results.append(
                {
                    "name": name,
                    "status": "PASS",
                    "reused": True,
                    "job": prior[-1].get("job"),
                }
            )
            continue
        if name == "long":
            model = env.get("M", "")
            base = env.get("GATE_LONG_BASE") or long_base(repo)
            log(
                f"gate long: yv --suite {LONG_SUITE} base {base} cand HEAD {commit[:12]}"
            )
            if not model or not Path(model).is_dir():
                ok, reasons, where = (
                    False,
                    [f"M (Qwen3.8-27B) not set or missing: {model!r}"],
                    "-",
                )
            else:
                ok, reasons, where = run_long_stage(
                    {"base": base, "model": model},
                    gq,
                    log,
                    runs,
                    repo,
                    priority,
                    label_prefix,
                )
            rd.append(
                name,
                {
                    "ev": "stage_complete",
                    "passed": ok,
                    "reasons": reasons,
                    "job": where,
                },
            )
            results.append(
                {
                    "name": name,
                    "status": "PASS" if ok else "FAIL",
                    "reasons": reasons,
                    "job": where,
                }
            )
            log(f"gate long: {'PASS' if ok else 'FAIL'} {'; '.join(reasons)[:300]}")
            if not ok:
                failed = name
                break
            continue
        out_dir = rd.path / f"gate-{name}"
        out_dir.mkdir(exist_ok=True)
        for f in out_dir.glob("results-*.jsonl"):
            f.unlink()  # a stage's rows are all from this attempt
        stage_sel = {"soak-mmlu": "soak-mmlu", "soak-realistic": "soak-realistic"}.get(
            name, name
        )
        job_env = {
            "STAGE": stage_sel,
            "OUT": str(out_dir),
            "PORT": GATE_PORT,
            "PY": VENV_PY,
            "MMLU_DATA": str(
                MAIN_CHECKOUT / "reference/omlx/omlx/eval/data/mmlu_pro_test.jsonl"
            ),
            **env,
        }
        argv = [
            "env",
            *[f"{k}={v}" for k, v in sorted(job_env.items())],
            "zsh",
            str(repo / "scripts/release/gate.sh"),
        ]
        label = f"{label_prefix}-gate-{commit[:8]}-{name}-{int(now()) % 100000}"
        jid = gq.submit(
            label,
            argv,
            timeout_min=timeout,
            stall_min=stall,
            mem_gb=mem,
            priority=priority,
            cwd=repo,
            quiet=name == "serve-27b",
        )
        rd.append(name, {"ev": "cell_submitted", "cell": name, "job": jid})
        log(f"gate {name}: job {jid}")
        job = gq.wait(jid)
        ok, reasons = judge_rows(read_stage_rows(out_dir), prefix)
        if job.state != "done" or job.rc not in (0, None):
            if ok and job.state == "done" and job.rc == 1:
                pass  # rc 1 only reflects FAIL rows of other prefixes (none selected): rows decide
            else:
                ok = False
                reasons.append(f"job {jid} {job.state} rc={job.rc}")
        rd.append(
            name, {"ev": "stage_complete", "passed": ok, "reasons": reasons, "job": jid}
        )
        if ok and name == "install":
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(commit)
        results.append(
            {
                "name": name,
                "status": "PASS" if ok else "FAIL",
                "reasons": reasons,
                "job": jid,
            }
        )
        log(f"gate {name}: {'PASS' if ok else 'FAIL'} {'; '.join(reasons)[:300]}")
        if not ok:
            failed = name
            break
    done = {r["name"] for r in results}
    for s in stages:
        if s not in done:
            results.append({"name": s, "status": "NOT_RUN"})
    overall = "FAIL" if failed else "PASS"
    v = {
        "schema": 1,
        "kind": "gate",
        "commit": commit,
        "overall": overall,
        "exit_code": 1 if failed else 0,
        "stages": results,
        "ended": time.time(),
    }
    write_json_atomic(rd.path / "verdict.json", v)
    md = [f"# 發布關卡：{'通過' if not failed else '失敗'}（{commit[:12]}）", ""]
    for r in results:
        md.append(
            f"- {r['name']}：{r['status']}"
            + (
                f"（{'；'.join(r.get('reasons', []))[:200]}）"
                if r.get("reasons")
                else ""
            )
            + (" [沿用先前結果]" if r.get("reused") else "")
        )
    md += (
        ["", "```", f"release gate {commit[:12]}: {overall}"]
        + [f"  {r['name']}: {r['status']} job {r.get('job', '-')}" for r in results]
        + ["```"]
    )
    (rd.path / "verdict.md").write_text("\n".join(md) + "\n")
    log(f"gate {overall}: {rd.path / 'verdict.md'}")
    return v["exit_code"]
