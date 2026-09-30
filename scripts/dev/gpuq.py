"""GPU job queue: one GPU, many benchmark jobs, run one at a time.

Benchmarks on a single Apple GPU are only meaningful when nothing else is on it,
so every GPU job goes through this queue. ``submit`` returns at once; a single
background daemon (started on demand, exits when idle) runs jobs in priority,
then FIFO order, each under a timeout, with output in a log file. Waiting is a
plain local process (``wait``), so whoever submitted can block on completion
without polling anything else.

    gpuq submit [--label L] [--timeout MIN] [--priority N] -- cmd args...   # prints job id
    gpuq wait ID [ID ...]        # block until all finish; exit 1 if any failed
    gpuq run [opts] -- cmd ...   # submit + wait (drop-in for the old gpu_run.sh)
    gpuq status                  # queue table
    gpuq log ID                  # print a job's log
    gpuq cancel ID               # drop a pending job or stop a running one

State lives in $GPUQ_DIR (default ~/.cache/yunshu/gpuq). Jobs keep the
submitter's cwd and environment.
"""

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("GPUQ_DIR", "~/.cache/yunshu/gpuq")).expanduser()
JOBS = ROOT / "jobs"
LOGS = ROOT / "logs"
IDLE_EXIT_S = 600
STALL_S = 600  # a job whose log stops growing this long is stopped as "stalled"
POLL_S = 2.0


def _now() -> float:
    return time.time()


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def _jobs() -> list[dict]:
    return sorted(
        (j for p in JOBS.glob("*.json") if (j := _read(p))),
        key=lambda j: (-j.get("priority", 0), j["submitted"], j["id"]),
    )


def _owner(job: dict) -> str:
    """Fair-share key: the agent worktree (or checkout) the job came from."""
    cwd = job.get("cwd", "")
    marker = "/.claude/worktrees/"
    if marker in cwd:
        return cwd.split(marker, 1)[1].split("/", 1)[0]
    return job.get("env", {}).get("GPUQ_OWNER", "main")


def _pick(jobs: list[dict]) -> dict | None:
    """Highest priority first; within it, the owner that last ran longest ago
    (round-robin across agents), then that owner's oldest job."""
    pending = [j for j in jobs if j["state"] == "pending"]
    if not pending:
        return None
    top = max(j.get("priority", 0) for j in pending)
    pending = [j for j in pending if j.get("priority", 0) == top]
    last: dict[str, float] = {}
    for j in jobs:
        if "started" in j:
            o = _owner(j)
            last[o] = max(last.get(o, 0.0), j["started"])
    return min(pending, key=lambda j: (last.get(_owner(j), 0.0), j["submitted"]))


def _new_id(label: str) -> str:
    stamp = time.strftime("%m%d-%H%M%S")
    n = 0
    while True:
        jid = f"{stamp}-{n:02d}-{label}"[:80]
        if not (JOBS / f"{jid}.json").exists():
            return jid
        n += 1


def _daemon_running() -> bool:
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with open(ROOT / "daemon.lock", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True


def _ensure_daemon() -> None:
    if _daemon_running():
        return
    log = open(ROOT / "daemon.log", "a")  # noqa: SIM115 - handed to the daemon process
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_daemon"],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def submit(
    cmd: list[str],
    label: str,
    timeout_min: float,
    priority: int,
    stall_min: float = STALL_S / 60,
) -> str:
    JOBS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    jid = _new_id(label)
    _write(
        JOBS / f"{jid}.json",
        {
            "id": jid,
            "label": label,
            "cmd": cmd,
            "cwd": os.getcwd(),
            "env": dict(os.environ),
            "timeout_s": timeout_min * 60,
            "stall_s": stall_min * 60,
            "priority": priority,
            "submitted": _now(),
            "state": "pending",
        },
    )
    _ensure_daemon()
    return jid


def _run_one(job: dict, path: Path) -> None:
    job.update(state="running", started=_now(), pid=None)
    log = open(LOGS / f"{job['id']}.log", "w")  # noqa: SIM115 - closed after the job ends
    try:
        rc_file = LOGS / f"{job['id']}.rc"
        proc = subprocess.Popen(
            ["/bin/sh", "-c", '"$@"; echo $? > "$GPUQ_RC"', "sh", *job["cmd"]],
            cwd=job["cwd"],
            env={**job["env"], "GPUQ_RC": str(rc_file)},
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        log.write(f"gpuq: failed to start: {e}\n")
        job.update(state="failed", rc=127, ended=_now())
        _write(path, job)
        return
    job["pid"] = proc.pid
    _write(path, job)
    deadline = job["started"] + job["timeout_s"]
    rc, why = None, None
    last_size, last_growth = -1, _now()
    while rc is None:
        try:
            rc = proc.wait(timeout=POLL_S)
        except subprocess.TimeoutExpired:
            log_path = LOGS / f"{job['id']}.log"
            size = log_path.stat().st_size if log_path.exists() else 0
            if size != last_size:
                last_size, last_growth = size, _now()
            if _read(path).get("cancel"):
                why = "cancelled"
            elif _now() > deadline:
                why = "timeout"
            elif _now() - last_growth > job.get("stall_s", STALL_S):
                why = "stalled"
            if why:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
    state = why or ("done" if rc == 0 else "failed")
    job.update(state=state, rc=rc, ended=_now())
    log.write(f"\ngpuq: {state} rc={rc} after {job['ended'] - job['started']:.0f}s\n")
    log.close()
    _write(path, job)


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _adopt(job: dict, path: Path) -> None:
    pid = job.get("pid")
    deadline = job.get("started", _now()) + job.get("timeout_s", 1200)
    why = None
    while _alive(pid):
        if not why and (_read(path).get("cancel") or _now() > deadline):
            why = "cancelled" if _read(path).get("cancel") else "timeout"
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGINT)
        time.sleep(POLL_S)
    rc_file = LOGS / f"{job['id']}.rc"
    try:
        rc = int(rc_file.read_text().strip())
    except (OSError, ValueError):
        rc = None
    if rc is None and not why:
        # A reboot or killed shell can leave an ordinary partial log without
        # writing the exit status. Absence of a traceback is not completion.
        state = "lost"
    else:
        state = why or ("done" if rc == 0 else "failed")
    job.update(state=state, rc=rc, ended=_now(), adopted=True)
    _write(path, job)


def daemon() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / "daemon.lock", "a")  # noqa: SIM115 - held for the daemon's lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    idle_since = _now()
    while True:
        # A job left running by a previous daemon is adopted: wait for its
        # process group to exit (still under timeout / cancel), then record it.
        for j in _jobs():
            if j["state"] == "running":
                _adopt(j, JOBS / f"{j['id']}.json")
        job = _pick(_jobs())
        if job is None:
            if _now() - idle_since > IDLE_EXIT_S:
                return
            time.sleep(POLL_S)
            continue
        path = JOBS / f"{job['id']}.json"
        if _read(path).get("cancel"):
            job.update(state="cancelled", ended=_now())
            _write(path, job)
            continue
        print(f"{time.strftime('%H:%M:%S')} run {job['id']}", flush=True)
        _run_one(job, path)
        print(f"{time.strftime('%H:%M:%S')} end {job['id']} {job['state']}", flush=True)
        idle_since = _now()


FINAL = {"done", "failed", "timeout", "stalled", "cancelled", "lost"}


def wait(ids: list[str]) -> int:
    bad = 0
    for jid in ids:
        path = JOBS / f"{jid}.json"
        while (j := _read(path)).get("state") not in FINAL:
            if not j:
                print(f"gpuq: no such job {jid}", file=sys.stderr)
                return 2
            time.sleep(POLL_S * 2)
        dur = j.get("ended", 0) - j.get("started", j.get("ended", 0))
        print(
            f"{jid}: {j['state']} rc={j.get('rc')} {dur:.0f}s log={LOGS / (jid + '.log')}"
        )
        bad += j["state"] != "done"
    return 1 if bad else 0


def status() -> None:
    now = _now()
    rows = _jobs()
    active = [j for j in rows if j["state"] in ("pending", "running")]
    recent = [
        j for j in rows if j["state"] in FINAL and now - j.get("ended", 0) < 6 * 3600
    ]
    print(f"daemon: {'up' if _daemon_running() else 'down'}")
    for j in active + recent[-15:]:
        t0 = j.get("started", j["submitted"])
        age = (j.get("ended") or now) - t0
        print(f"{j['state']:>9}  {age:6.0f}s  p{j.get('priority', 0)}  {j['id']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="op", required=True)
    for name in ("submit", "run"):
        p = sub.add_parser(name)
        p.add_argument("--label", default="job")
        p.add_argument(
            "--timeout", type=float, default=20.0, help="minutes (default 20)"
        )
        p.add_argument("--priority", type=int, default=0, help="higher runs first")
        p.add_argument(
            "--stall",
            type=float,
            default=STALL_S / 60,
            help="minutes without log output before stopping (default 10)",
        )
        p.add_argument("cmd", nargs=argparse.REMAINDER)
    sub.add_parser("wait").add_argument("ids", nargs="+")
    sub.add_parser("status")
    sub.add_parser("log").add_argument("id")
    sub.add_parser("cancel").add_argument("id")
    sub.add_parser("_daemon")
    a = ap.parse_args()
    if a.op in ("submit", "run"):
        cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        if not cmd:
            ap.error("missing command after --")
        jid = submit(cmd, a.label, a.timeout, a.priority, a.stall)
        print(jid, flush=True)
        return wait([jid]) if a.op == "run" else 0
    if a.op == "wait":
        return wait(a.ids)
    if a.op == "status":
        status()
    elif a.op == "log":
        sys.stdout.write((LOGS / f"{a.id}.log").read_text())
    elif a.op == "cancel":
        path = JOBS / f"{a.id}.json"
        j = _read(path)
        if j:
            j["cancel"] = True
            _write(path, j)
    elif a.op == "_daemon":
        daemon()
    return 0


if __name__ == "__main__":
    sys.exit(main())
