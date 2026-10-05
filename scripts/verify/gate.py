"""`yv gate`: scripts/release/gate.sh through gpuq, one job per stage, persisted per commit.

A rerun on the same commit skips stages that already passed (the 68-minute gate that failed
at its last stage used to start over). A stage passes when its check rows (results-*.jsonl)
have no FAIL / CONTENDED, at least one PASS, and the job exited 0.
"""

from __future__ import annotations

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
}
DEFAULT_STAGES = [
    "install",
    "serve-27b",
    "families",
    "soak-mmlu",
    "soak-realistic",
    "agent-sessions",
]
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
        label = f"infra-gate-{commit[:8]}-{name}-{int(now()) % 100000}"
        jid = gq.submit(
            label,
            argv,
            timeout_min=timeout,
            stall_min=stall,
            mem_gb=mem,
            priority=priority,
            cwd=repo,
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
