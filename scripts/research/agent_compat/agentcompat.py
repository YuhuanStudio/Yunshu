"""One command, one fail-closed verdict: are the LATEST agent CLIs and SDKs still served correctly?

    scripts/dev/agentcompat [--stages install,census,m3] [--model Qwen3.5-9B-MLX-4bit] [--out DIR]

Stages (CPU on the M5; the model runs on the M3 lane):
  install  install the newest Claude Code / Codex / opencode into AGENTIC_CLIS_LATEST (never global, never ~)
  census   run the scripted census sessions with those CLIs against the recording mock server, diff it against
           the pinned census (new paths / headers / betas / body fields / tool and block types) and require every
           new item to be written down in docs/guides/AGENT_COMPAT.md
  m3       submit m3_serve.py to the gpuq M3 lane, reach it through an ssh port-forward on 127.0.0.1, replay every
           recorded census request (status + stream / body validity against the official event shapes), then run
           the scripted agent tasks (e2e_agents.py --url) with the latest CLIs; stop the job, kill the tunnel
Verdict: <out>/verdict.json; exit 0 only when every requested stage has positive evidence. No evidence of
success (missing file, zero requests, unfinished job) is a failure.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GPUQ = REPO / "scripts" / "dev" / "gpuq"
PY = os.environ.get("AGENTCOMPAT_PYTHON", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python")
CLIS_LATEST = Path(os.environ.get("AGENTIC_CLIS_LATEST", "/Volumes/P5Plus/yunshu-build/agentic-clis-latest"))
PINNED_CENSUS = os.environ.get("AGENTCOMPAT_PINNED_CENSUS", "")
DOC = REPO / "docs" / "guides" / "AGENT_COMPAT.md"
M3_MODELS_DIR = "/Volumes/P5Plus/models"
SSH = ["ssh", "-i", os.path.expanduser("~/.ssh/yunshu_m3"), "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes"]
M3_HOST = os.environ.get("M3_HOST", "yuhuan@192.168.50.55")
SERVER_PORT, CONTROL_PORT, MCP_PORT = 18994, 18995, 18996
E2E_SCENARIOS = "cc_edit,cx_edit,oc_edit,oc_bash,cc_mcp,cx_mcp,cc_image,cc_context"


def new_item_token(kind: str, item: str) -> str:
    """The text of a census-diff item that AGENT_COMPAT.md must mention."""
    tok = item.strip().split(" ")[-1]
    if kind in ("header_values", "types") and "=" in tok:
        tok = tok.split("=", 1)[1]
    if kind == "endpoints":
        tok = tok.split("?")[0]
    return tok


def doc_gaps(diff_new: dict[str, list[str]], doc_text: str) -> list[str]:
    """Items the latest census saw but the compat doc never names."""
    gaps = []
    for kind, items in diff_new.items():
        for it in items:
            tok = new_item_token(kind, it)
            if tok and tok not in doc_text:
                gaps.append(f"{kind}: {it}")
    return gaps


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, **kw)


# ---- stages ---------------------------------------------------------------------------------------------
def stage_install(out: Path) -> dict:
    env = {**os.environ, "AGENTIC_CLIS": str(CLIS_LATEST), "AGENTIC_INSTALL_HOME": str(CLIS_LATEST / "install-home"),
           "OPENCODE_V": "latest", "CLAUDE_V": "latest", "CODEX_V": "latest"}
    p = run(["sh", str(REPO / "scripts/research/agentic/install_agents.sh")], env=env, capture_output=True)
    (out / "install.log").write_text((p.stdout or "") + (p.stderr or ""))
    vers = {}
    for c in ("claude", "opencode", "codex"):
        v = run([str(CLIS_LATEST / "node_modules/.bin" / c), "--version"], capture_output=True, env={**env, "HOME": str(CLIS_LATEST / "install-home")})
        vers[c] = (v.stdout or "").strip()
    ok = p.returncode == 0 and all(vers.values())
    return dict(ok=ok, versions=vers, why="" if ok else "install failed, see install.log")


def stage_census(out: Path) -> dict:
    cdir = out / "census"
    cdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "AGENTIC_CLIS": str(CLIS_LATEST), "CENSUS_OUT": str(cdir)}
    p = run([PY, str(HERE / "census.py"), "run", "all"], env=env, cwd=HERE, capture_output=True, timeout=1800)
    (cdir / "run.log").write_text((p.stdout or "")[-20000:])
    sessions = sorted(d for d in cdir.iterdir() if (d / "meta.json").is_file())
    bad = []
    for d in sessions:
        meta = json.loads((d / "meta.json").read_text())
        n = sum(1 for _ in (d / "requests.jsonl").open()) if (d / "requests.jsonl").is_file() else 0
        if meta.get("rc") != 0 or n == 0:
            bad.append(f"{d.name}: rc={meta.get('rc')} requests={n}")
    if not sessions:
        return dict(ok=False, why="no census session ran")
    res: dict = dict(sessions=len(sessions), failed_sessions=bad)
    pinned = PINNED_CENSUS or _newest_pinned()
    if not pinned or not Path(pinned).is_dir():
        return dict(ok=False, why=f"no pinned census to diff against ({pinned!r})", **res)
    sys.path.insert(0, str(HERE))
    import census_diff

    new = census_diff.diff(census_diff.signature(Path(pinned)), census_diff.signature(cdir))
    gaps = doc_gaps(new, DOC.read_text())
    res.update(pinned=str(pinned), new_items=new, undocumented=gaps)
    (out / "census_diff.json").write_text(json.dumps(res, indent=1))
    res["ok"] = not bad and not gaps
    res["why"] = "; ".join(x for x in (f"failed sessions {bad}" if bad else "", f"{len(gaps)} census items not in AGENT_COMPAT.md" if gaps else "") if x)
    return res


def _newest_pinned() -> str:
    main = Path(os.environ.get("YUNSHU_MAIN", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu"))
    cands = sorted(p for p in (main / "docs/research/runs").glob("*-agent-census") if (p / "cc_plain").is_dir())
    return str(cands[0]) if cands else ""  # the oldest census is the pinned-CLI baseline


def port_free(p: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", p))
            return True
        except OSError:
            return False


def stage_m3(out: Path, model: str, minutes: float, scenarios: str) -> dict:
    mdir = out / "m3"
    mdir.mkdir(parents=True, exist_ok=True)
    sha = run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True).stdout.strip()
    dirty = run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True).stdout.strip()
    for p in (SERVER_PORT, CONTROL_PORT, MCP_PORT):
        if not port_free(p):
            return dict(ok=False, why=f"local port {p} is busy")
    receipt = out / "m3_serve.jsonl"
    receipt.unlink(missing_ok=True)
    sub = run(
        [str(GPUQ), "submit", "--device", "m3", "--label", f"agentcompat-serve-{sha}-{int(time.time()) % 100000}", "--mem-gb", "9",
         "--timeout", str(int(minutes) + 15), "--stall", "5", "--out", str(receipt), "--expect-complete", "--",
         "python", "scripts/research/agent_compat/m3_serve.py", "--model", f"{M3_MODELS_DIR}/{model}", "--port", str(SERVER_PORT),
         "--control-port", str(CONTROL_PORT), "--minutes", str(minutes), "--out", str(receipt)],
        capture_output=True,
    )
    job = (sub.stdout or "").strip().split()[-1] if sub.returncode == 0 and sub.stdout.strip() else ""
    if not job:
        return dict(ok=False, why="gpuq submit failed: " + ((sub.stdout or "") + (sub.stderr or ""))[-300:])
    print(f"m3 job {job} (commit {sha}{' + tracked edits' if dirty else ''})", flush=True)
    tunnel = subprocess.Popen(
        [*SSH, "-N", "-L", f"{SERVER_PORT}:127.0.0.1:{SERVER_PORT}", "-L", f"{CONTROL_PORT}:127.0.0.1:{CONTROL_PORT}", M3_HOST],
        stdin=subprocess.DEVNULL,
    )
    res: dict = dict(job=job, commit=sha, dirty=bool(dirty))
    try:
        url = f"http://127.0.0.1:{SERVER_PORT}"
        t0 = time.time()
        ready = False
        while time.time() - t0 < 45 * 60:
            if tunnel.poll() is not None:
                return dict(ok=False, why="ssh tunnel exited", **res)
            if run([str(GPUQ), "wait", "--max-seconds", "1", job], capture_output=True).returncode != 2:
                return dict(ok=False, why="M3 job finished before the server was ready: " + run([str(GPUQ), "log", job], capture_output=True).stdout[-400:], **res)
            with contextlib.suppress(Exception):
                if urllib.request.urlopen(url + "/v1/models", timeout=3).status == 200:
                    ready = True
                    break
            time.sleep(5)
        if not ready:
            return dict(ok=False, why="M3 server not ready in 45 min", **res)
        print(f"server ready after {time.time() - t0:.0f}s", flush=True)
        census_dir = out / "census"
        steps = {}
        steps["replay"] = run([PY, str(HERE / "replay.py"), "--url", url, "--census", str(census_dir), "--out", str(mdir / "replay")],
                              cwd=HERE, capture_output=True, timeout=3600)
        (mdir / "replay.log").write_text((steps["replay"].stdout or "")[-60000:] + (steps["replay"].stderr or "")[-4000:])
        env = {**os.environ, "AGENTIC_CLIS": str(CLIS_LATEST), "AGENTCOMPAT_MCP_PORT": str(MCP_PORT), "CENSUS_OUT": str(mdir / "census-scratch")}
        steps["agents"] = run([PY, str(HERE / "e2e_agents.py"), "--url", url, "--scenarios", scenarios, "--out", str(mdir / "e2e"), "--timeout", "600"],
                              cwd=HERE, env=env, capture_output=True, timeout=3 * 3600)
        (mdir / "agents.log").write_text((steps["agents"].stdout or "")[-60000:] + (steps["agents"].stderr or "")[-4000:])
        res["replay_rc"], res["agents_rc"] = steps["replay"].returncode, steps["agents"].returncode
        res["replay_tail"] = [ln for ln in (steps["replay"].stdout or "").splitlines() if ln.startswith(("PROBLEM", "replay:"))][-40:]
        res["agents_tail"] = [ln for ln in (steps["agents"].stdout or "").splitlines() if "PASS" in ln or "FAIL" in ln or ln.startswith("e2e_agents:")][-20:]
        res["ok"] = res["replay_rc"] == 0 and res["agents_rc"] == 0
        res["why"] = "" if res["ok"] else f"replay rc={res['replay_rc']}, agents rc={res['agents_rc']}"
    finally:
        with contextlib.suppress(Exception):
            with socket.create_connection(("127.0.0.1", CONTROL_PORT), timeout=5) as c:
                c.sendall(b"stop\n")
        done = run([str(GPUQ), "wait", "--max-seconds", "300", job], capture_output=True).returncode
        res["job_rc"] = done
        tunnel.terminate()
        with contextlib.suppress(Exception):
            tunnel.wait(5)
        if tunnel.poll() is None:
            tunnel.kill()
        if done == 2:
            run([str(GPUQ), "cancel", job], capture_output=True)
    if res.get("job_rc") != 0:
        res["ok"] = False
        res["why"] = (res.get("why", "") + f"; M3 job did not finish cleanly (rc={res.get('job_rc')})").strip("; ")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="census,m3")
    ap.add_argument("--model", default="Qwen3.5-9B-MLX-4bit")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--scenarios", default=E2E_SCENARIOS)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    sha = run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True).stdout.strip()
    out = Path(a.out or REPO / "docs/research/runs" / f"{time.strftime('%Y-%m-%d-%H%M')}-agentcompat-{sha}")
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in a.stages.split(",") if s]
    verdict: dict = dict(commit=sha, stages={})
    for s in stages:
        print(f"== stage {s}", flush=True)
        try:
            if s == "install":
                r = stage_install(out)
            elif s == "census":
                r = stage_census(out)
            elif s == "m3":
                r = stage_m3(out, a.model, a.minutes, a.scenarios)
            else:
                r = dict(ok=False, why="unknown stage")
        except Exception as e:  # noqa: BLE001
            r = dict(ok=False, why=f"exception: {e!r}")
        verdict["stages"][s] = r
        print(f"   {s}: {'OK' if r.get('ok') else 'FAIL'} {r.get('why', '')}", flush=True)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    verdict["ok"] = bool(stages) and all(verdict["stages"][s].get("ok") is True for s in stages)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(("AGENTCOMPAT PASS" if verdict["ok"] else "AGENTCOMPAT FAIL"), out / "verdict.json", flush=True)
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
