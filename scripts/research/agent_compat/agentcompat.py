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
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GPUQ = Path(os.environ.get("YV_GPUQ", str(REPO / "scripts/dev/gpuq")))
PY = os.environ.get(
    "AGENTCOMPAT_PYTHON", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python"
)
CLIS_LATEST = Path(
    os.environ.get(
        "AGENTIC_CLIS_LATEST", "/Volumes/P5Plus/yunshu-build/agentic-clis-latest"
    )
)
PINNED_CENSUS = os.environ.get("AGENTCOMPAT_PINNED_CENSUS", "")
DOC = REPO / "docs" / "guides" / "AGENT_COMPAT.md"
M3_MODELS_DIR = "/Volumes/P5Plus/models"
SSH = [
    "ssh",
    "-i",
    os.path.expanduser("~/.ssh/yunshu_m3"),
    "-o",
    "ExitOnForwardFailure=yes",
    "-o",
    "BatchMode=yes",
]
M3_HOST = os.environ.get("M3_HOST", "yuhuan@192.168.50.55")
# Ports on the laptop (its server and control socket) ...
SERVER_PORT, CONTROL_PORT = 18994, 18995
# ... and on this Mac: the forwards and the MCP fake use 18997-18999, outside the 18990-18996
# range of M5 gpuq jobs, which run at the same time as M3 jobs (2026-10-07: a forward on 18995
# made an M5 yv memory job fail to listen).
LOCAL_SERVER_PORT, LOCAL_CONTROL_PORT, MCP_PORT = 18997, 18998, 18999
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
    env = {
        **os.environ,
        "AGENTIC_CLIS": str(CLIS_LATEST),
        "AGENTIC_INSTALL_HOME": str(CLIS_LATEST / "install-home"),
        "OPENCODE_V": "latest",
        "CLAUDE_V": "latest",
        "CODEX_V": "latest",
    }
    p = run(
        ["sh", str(REPO / "scripts/research/agentic/install_agents.sh")],
        env=env,
        capture_output=True,
    )
    (out / "install.log").write_text((p.stdout or "") + (p.stderr or ""))
    vers = {}
    for c in ("claude", "opencode", "codex"):
        v = run(
            [str(CLIS_LATEST / "node_modules/.bin" / c), "--version"],
            capture_output=True,
            env={**env, "HOME": str(CLIS_LATEST / "install-home")},
        )
        vers[c] = (v.stdout or "").strip()
    ok = p.returncode == 0 and all(vers.values())
    return dict(
        ok=ok, versions=vers, why="" if ok else "install failed, see install.log"
    )


def stage_census(out: Path) -> dict:
    cdir = out / "census"
    cdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "AGENTIC_CLIS": str(CLIS_LATEST), "CENSUS_OUT": str(cdir)}
    p = run(
        [PY, str(HERE / "census.py"), "run", "all"],
        env=env,
        cwd=HERE,
        capture_output=True,
        timeout=1800,
    )
    (cdir / "run.log").write_text((p.stdout or "")[-20000:])
    sessions = sorted(d for d in cdir.iterdir() if (d / "meta.json").is_file())
    pinned = PINNED_CENSUS or _newest_pinned()
    bad = []
    for d in sessions:
        meta = json.loads((d / "meta.json").read_text())
        n = (
            sum(1 for _ in (d / "requests.jsonl").open())
            if (d / "requests.jsonl").is_file()
            else 0
        )
        # a session that made no request in the pinned census (a slash command) may make none now
        pf = Path(pinned) / d.name / "requests.jsonl" if pinned else None
        needs_requests = pf is None or (pf.is_file() and pf.stat().st_size > 0)
        if meta.get("rc") != 0 or (n == 0 and needs_requests):
            bad.append(f"{d.name}: rc={meta.get('rc')} requests={n}")
    if not sessions:
        return dict(ok=False, why="no census session ran")
    res: dict = dict(sessions=len(sessions), failed_sessions=bad)
    if not pinned or not Path(pinned).is_dir():
        return dict(
            ok=False, why=f"no pinned census to diff against ({pinned!r})", **res
        )
    sys.path.insert(0, str(HERE))
    import census_diff

    new = census_diff.diff(
        census_diff.signature(Path(pinned)), census_diff.signature(cdir)
    )
    gaps = doc_gaps(new, DOC.read_text())
    res.update(pinned=str(pinned), new_items=new, undocumented=gaps)
    (out / "census_diff.json").write_text(json.dumps(res, indent=1))
    res["ok"] = not bad and not gaps
    res["why"] = "; ".join(
        x
        for x in (
            f"failed sessions {bad}" if bad else "",
            f"{len(gaps)} census items not in AGENT_COMPAT.md" if gaps else "",
        )
        if x
    )
    return res


def _newest_pinned() -> str:
    main = Path(
        os.environ.get("YUNSHU_MAIN", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
    )
    cands = sorted(
        p
        for p in (main / "docs/research/runs").glob("*-agent-census")
        if (p / "cc_plain").is_dir()
    )
    return (
        str(cands[0]) if cands else ""
    )  # the oldest census is the pinned-CLI baseline


def port_free(p: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", p))
            return True
        except OSError:
            return False


def run_watched(cmd, tunnel, timeout, **kw):
    """Run a client command; kill it and fail if the ssh tunnel dies (a dropped tunnel must not hang the run)."""
    p = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        **kw,
    )
    out: list[str] = []
    t = threading.Thread(
        target=lambda: out.append(p.communicate()[0] or ""), daemon=True
    )
    t.start()
    t0 = time.time()
    dropped = False
    while t.is_alive():
        t.join(2)
        if tunnel.poll() is not None:
            dropped = True
        if dropped or time.time() - t0 > timeout:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            t.join(10)
            return 124 if not dropped else 125, "".join(out) + (
                "\nTUNNEL DROPPED" if dropped else "\nTIMEOUT"
            )
    return p.returncode, "".join(out)


def server_identity_ready(identity: str) -> bool:
    """The control socket must belong to this queued job, not another lane server."""
    try:
        with socket.create_connection(
            ("127.0.0.1", LOCAL_CONTROL_PORT), timeout=2
        ) as c:
            c.sendall(f"ready {identity}\n".encode())
            return c.recv(256).decode().strip() == f"ready {identity}"
    except OSError:
        return False


LABEL_PREFIX = "agentcompat"
PRIORITY = -1


def with_m3_server(out: Path, tag: str, model: str, minutes: float, work) -> dict:
    """One bounded M3 serve job (<= ~20 min) behind an ssh -L forward; `work(url, tunnel)` runs the M5-side clients."""
    sha = run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True
    ).stdout.strip()
    for p in (LOCAL_SERVER_PORT, LOCAL_CONTROL_PORT, MCP_PORT):
        if not port_free(p):
            return dict(ok=False, why=f"local port {p} is busy")
    identity = uuid.uuid4().hex
    receipt = out / f"m3_serve_{tag}.jsonl"
    receipt.unlink(missing_ok=True)
    sub = run(
        [
            str(GPUQ),
            "submit",
            "--priority",
            str(PRIORITY),
            "--device",
            "m3",
            "--label",
            f"{LABEL_PREFIX}-{tag}-{sha}-{int(time.time()) % 100000}",
            "--mem-gb",
            "9",
            "--timeout",
            str(int(minutes) + 8),
            "--stall",
            "5",
            "--out",
            str(receipt),
            "--expect-complete",
            "--",
            "python",
            "scripts/research/agent_compat/m3_serve.py",
            "--identity",
            identity,
            "--model",
            f"{M3_MODELS_DIR}/{model}",
            "--port",
            str(SERVER_PORT),
            "--control-port",
            str(CONTROL_PORT),
            "--minutes",
            str(minutes),
            "--out",
            str(receipt),
        ],
        capture_output=True,
    )
    job = (
        (sub.stdout or "").strip().split()[-1]
        if sub.returncode == 0 and sub.stdout.strip()
        else ""
    )
    if not job:
        return dict(
            ok=False,
            why="gpuq submit failed: "
            + ((sub.stdout or "") + (sub.stderr or ""))[-300:],
        )
    print(f"m3 job {job} ({tag}, commit {sha})", flush=True)
    tunnel = subprocess.Popen(
        [
            *SSH,
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-N",
            "-L",
            f"{LOCAL_SERVER_PORT}:127.0.0.1:{SERVER_PORT}",
            "-L",
            f"{LOCAL_CONTROL_PORT}:127.0.0.1:{CONTROL_PORT}",
            M3_HOST,
        ],
        stdin=subprocess.DEVNULL,
    )
    res: dict = dict(job=job, commit=sha, tag=tag)
    try:
        url = f"http://127.0.0.1:{LOCAL_SERVER_PORT}"
        t0 = time.time()
        ready = False
        while time.time() - t0 < 30 * 60:
            if tunnel.poll() is not None:
                return dict(ok=False, why="ssh tunnel exited before ready", **res)
            if (
                run(
                    [str(GPUQ), "wait", "--max-seconds", "1", job], capture_output=True
                ).returncode
                != 2
            ):
                return dict(
                    ok=False,
                    why="M3 job finished before ready: "
                    + run([str(GPUQ), "log", job], capture_output=True).stdout[-400:],
                    **res,
                )
            with contextlib.suppress(Exception):
                if (
                    server_identity_ready(identity)
                    and urllib.request.urlopen(url + "/v1/models", timeout=3).status
                    == 200
                ):
                    ready = True
                    break
            time.sleep(5)
        if not ready:
            return dict(ok=False, why="M3 server not ready in 30 min", **res)
        print(f"server ready after {time.time() - t0:.0f}s", flush=True)
        res.update(work(url, tunnel))
    finally:
        if ready:
            with contextlib.suppress(Exception):
                with socket.create_connection(
                    ("127.0.0.1", LOCAL_CONTROL_PORT), timeout=5
                ) as c:
                    c.sendall(f"stop {identity}\n".encode())
        done = run(
            [str(GPUQ), "wait", "--max-seconds", "300", job], capture_output=True
        ).returncode
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
        res["why"] = (
            res.get("why", "")
            + f"; M3 job did not finish cleanly (rc={res.get('job_rc')}); check the laptop before using the lane again"
        ).strip("; ")
    return res


def stage_m3(out: Path, model: str, minutes: float, scenarios: str) -> dict:
    """Two bounded serve jobs: spec replay (+ synthetic SDK shapes), then the scripted agent tasks."""
    mdir = out / "m3"
    mdir.mkdir(parents=True, exist_ok=True)
    census_dir = out / "census"

    def replay(url, tunnel):
        rc, text = run_watched(
            [
                PY,
                str(HERE / "replay.py"),
                "--url",
                url,
                "--census",
                str(census_dir),
                "--out",
                str(mdir / "replay"),
                "--per-session",
                "2",
            ],
            tunnel,
            minutes * 60 - 60,
            cwd=HERE,
        )
        (mdir / "replay.log").write_text(text[-80000:])
        tail = [
            ln for ln in text.splitlines() if ln.startswith(("PROBLEM", "replay:"))
        ][-40:]
        return dict(
            ok=rc == 0, rc=rc, tail=tail, why="" if rc == 0 else f"replay rc={rc}"
        )

    def agents(url, tunnel):
        env = {
            **os.environ,
            "AGENTIC_CLIS": str(CLIS_LATEST),
            "AGENTCOMPAT_MCP_PORT": str(MCP_PORT),
            "CENSUS_OUT": str(mdir / "census-scratch"),
        }
        rc, text = run_watched(
            [
                PY,
                str(HERE / "e2e_agents.py"),
                "--url",
                url,
                "--scenarios",
                scenarios,
                "--out",
                str(mdir / "e2e"),
                "--timeout",
                "300",
            ],
            tunnel,
            minutes * 60 - 60,
            cwd=HERE,
            env=env,
        )
        (mdir / "agents.log").write_text(text[-80000:])
        tail = [
            ln
            for ln in text.splitlines()
            if "PASS" in ln or "FAIL" in ln or ln.startswith("e2e_agents:")
        ][-20:]
        return dict(
            ok=rc == 0, rc=rc, tail=tail, why="" if rc == 0 else f"agents rc={rc}"
        )

    a = with_m3_server(out, "replay", model, minutes, replay)
    b = with_m3_server(out, "agents", model, minutes, agents)
    return dict(
        ok=a.get("ok") is True and b.get("ok") is True,
        replay=a,
        agents=b,
        why="; ".join(x for x in (a.get("why", ""), b.get("why", "")) if x),
    )


def main(argv=None) -> int:
    global LABEL_PREFIX, PRIORITY
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-prefix", default="agentcompat")
    ap.add_argument("--priority", type=int, default=-1)
    ap.add_argument("--stages", default="census,m3")
    ap.add_argument("--model", default="Qwen3.5-9B-MLX-4bit")
    ap.add_argument(
        "--minutes",
        type=float,
        default=20,
        help="length of each serve job (replay, agents)",
    )
    ap.add_argument("--scenarios", default=E2E_SCENARIOS)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    LABEL_PREFIX, PRIORITY = a.label_prefix, a.priority
    sha = run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True
    ).stdout.strip()
    out = Path(
        a.out
        or REPO
        / "docs/research/runs"
        / f"{time.strftime('%Y-%m-%d-%H%M')}-agentcompat-{sha}"
    )
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
        print(
            f"   {s}: {'OK' if r.get('ok') else 'FAIL'} {r.get('why', '')}", flush=True
        )
        (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    verdict["ok"] = bool(stages) and all(
        verdict["stages"][s].get("ok") is True for s in stages
    )
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(
        ("AGENTCOMPAT PASS" if verdict["ok"] else "AGENTCOMPAT FAIL"),
        out / "verdict.json",
        flush=True,
    )
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
