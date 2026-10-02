"""GPU job queue: one GPU, many benchmark jobs, run one at a time.

Benchmarks on a single Apple GPU are only meaningful when nothing else is on it,
so every GPU job goes through this queue. ``submit`` returns at once; a single
background daemon (started on demand, exits when idle) runs jobs in priority,
then FIFO order, each under a timeout, with output in a log file. Waiting is a
plain local process (``wait``), so whoever submitted can block on completion
without polling anything else.

    gpuq submit [--label L] [--timeout MIN] [--priority N] [--serving-ok] [--mem-gb G] -- cmd args...
    gpuq wait ID [ID ...]        # block until all finish; exit 1 if any failed
    gpuq run [opts] -- cmd ...   # submit + wait (drop-in for the old gpu_run.sh)
    gpuq status                  # queue table (paused / waiting-idle / waiting-mem columns)
    gpuq log ID                  # print a job's log
    gpuq cancel ID               # drop a pending job or stop a running one

Serving awareness (optional; docs/guides/SERVE_AND_DEVELOP.md): with production server URLs configured
(GPUQ_SERVING_URLS, or $GPUQ_DIR/serving.json) a job starts only after every server has been idle for
IDLE_START seconds, is SIGSTOPped (its own process group only) the moment a production request appears, and
SIGCONTed after IDLE_RESUME idle seconds. Pause intervals go to the job JSON and to the file named by
$GPUQ_PAUSE_FILE (read them with scripts/dev/gpuq_pause.py). ``--serving-ok`` bypasses the gate (CPU-only
jobs); ``--mem-gb`` is checked against free memory minus the production reserve. The poller reads only the
request counts from /v1/yunshu/status, never prompts.

State lives in $GPUQ_DIR (default ~/.cache/yunshu/gpuq). Jobs keep the
submitter's cwd and environment.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("GPUQ_DIR", "~/.cache/yunshu/gpuq")).expanduser()
JOBS = ROOT / "jobs"
LOGS = ROOT / "logs"
IDLE_EXIT_S = 600
STALL_S = 600  # a job whose log stops growing this long is stopped as "stalled"
POLL_S = 2.0
DEFAULT_MEM_GB = 24.0
DEFAULT_RESERVE_GB = 16.0


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


# ---------------------------------------------------------------- serving gate


def load_serving_config() -> dict:
    """Serving settings: $GPUQ_DIR/serving.json (re-read every loop) overridden by GPUQ_* env vars.

    Without URLs the gate is off and the queue behaves exactly as before."""
    cfg = _read(ROOT / "serving.json")
    env = os.environ
    if env.get("GPUQ_SERVING_URLS"):
        cfg["urls"] = [
            u.strip() for u in env["GPUQ_SERVING_URLS"].split(",") if u.strip()
        ]
    for key, var in (
        ("idle_start_s", "GPUQ_IDLE_START_S"),
        ("idle_resume_s", "GPUQ_IDLE_RESUME_S"),
        ("poll_s", "GPUQ_SERVING_POLL_S"),
        ("reserve_gb", "GPUQ_RESERVE_GB"),
    ):
        if env.get(var):
            cfg[key] = float(env[var])
    if env.get("GPUQ_SERVING_KEY"):
        cfg["api_key"] = env["GPUQ_SERVING_KEY"]
    if isinstance(cfg.get("urls"), str):
        cfg["urls"] = [cfg["urls"]]
    return cfg


def _fetch_busy(url: str, key: str | None, timeout: float = 2.0) -> bool:
    """True if the production server has active or queued requests. Reads counts only (never prompts).

    Connection refused = server down = nobody is being served = idle. Any other failure (timeout, 5xx, bad
    JSON) is treated as busy: an unresponsive server may well be mid-request, so fail closed."""
    req = urllib.request.Request(url.rstrip("/") + "/v1/yunshu/status")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reqs = json.loads(r.read()).get("requests") or {}
        return int(reqs.get("active") or 0) > 0 or int(reqs.get("queued") or 0) > 0
    except urllib.error.URLError as e:
        return not isinstance(e.reason, ConnectionRefusedError)
    except ConnectionRefusedError:
        return False
    except Exception:  # noqa: BLE001 - unknown state counts as busy
        return True


class ServingGate:
    """Tracks whether every production server has been idle, and for how long."""

    def __init__(self, fetch=None) -> None:
        self.cfg: dict = {}
        self.fetch = fetch or _fetch_busy
        self.idle_since: float | None = None

    def configure(self, cfg: dict) -> None:
        self.cfg = cfg
        if not self.enabled:
            self.idle_since = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("urls"))

    @property
    def poll_s(self) -> float:
        return float(self.cfg.get("poll_s", 1.0))

    @property
    def idle_start_s(self) -> float:
        return float(self.cfg.get("idle_start_s", 120.0))

    @property
    def idle_resume_s(self) -> float:
        return float(self.cfg.get("idle_resume_s", 30.0))

    @property
    def reserve_gb(self) -> float:
        return float(self.cfg.get("reserve_gb", DEFAULT_RESERVE_GB))

    def poll(self, now: float | None = None) -> bool:
        """Query every server; return True when any is busy."""
        now = _now() if now is None else now
        key = self.cfg.get("api_key")
        busy = False
        for url in self.cfg.get("urls", []):
            busy = self.fetch(url, key) or busy
        if busy:
            self.idle_since = None
        elif self.idle_since is None:
            self.idle_since = now
        return busy

    @property
    def busy(self) -> bool:
        return self.enabled and self.idle_since is None

    def idle_for(self, now: float | None = None) -> float:
        if self.idle_since is None:
            return 0.0
        return (_now() if now is None else now) - self.idle_since

    def may_start(self, now: float | None = None) -> bool:
        return (not self.enabled) or self.idle_for(now) >= self.idle_start_s

    def may_resume(self, now: float | None = None) -> bool:
        return (not self.enabled) or self.idle_for(now) >= self.idle_resume_s


def free_memory_gb() -> float | None:
    """Memory a new job could use without squeezing the production server (GB), or None if unknown."""
    try:
        import psutil  # type: ignore[import-not-found,unused-ignore]

        return float(psutil.virtual_memory().available) / 1e9
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
        page, pages = 16384, 0
        for line in out.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
            for k in (
                "Pages free",
                "Pages inactive",
                "Pages speculative",
                "Pages purgeable",
            ):
                if line.startswith(k):
                    pages += int(line.split(":")[1].strip().rstrip("."))
        return pages * page / 1e9
    except Exception:  # noqa: BLE001
        return None


def mem_blocked(job: dict, gate: ServingGate, free=free_memory_gb) -> bool:
    """True when starting `job` would leave less than the production reserve free.

    Only active with a serving config (or an explicit reserve_gb); otherwise the queue is unchanged."""
    if not gate.enabled and "reserve_gb" not in gate.cfg:
        return False
    need = float(job.get("mem_gb") or 0)
    if need <= 0:
        return False
    avail = free()
    if avail is None:  # cannot measure: do not wedge the queue
        return False
    return avail - need < gate.reserve_gb


def _blocker(job: dict, gate: ServingGate, free=free_memory_gb) -> str | None:
    """Why a pending job may not start now: 'idle' (production busy / not idle long enough), 'mem', or None."""
    if gate.enabled and not job.get("serving_ok") and not gate.may_start():
        return "idle"
    if mem_blocked(job, gate, free):
        return "mem"
    return None


class Pauser:
    """SIGSTOP / SIGCONT of one job's own process group, plus the pause-interval log."""

    def __init__(self, job: dict, path: Path) -> None:
        self.job, self.path = job, path
        self.pauses: list[list[float | None]] = [list(p) for p in job.get("pauses", [])]

    @property
    def paused(self) -> bool:
        return bool(self.pauses) and self.pauses[-1][1] is None

    def total(self, now: float) -> float:
        return sum((p[1] if p[1] is not None else now) - p[0] for p in self.pauses)

    def _save(self) -> None:
        self.job["pauses"] = self.pauses
        self.job["paused"] = self.paused
        data = _read(
            self.path
        )  # merge: never clobber a cancel flag the CLI set meanwhile
        data.update(pauses=self.pauses, paused=self.paused)
        _write(self.path, data)
        _write(LOGS / f"{self.job['id']}.pauses.json", {"pauses": self.pauses})

    def _signal(self, sig: int) -> None:
        pid = self.job.get("pid")
        if pid and pid > 1:  # only ever the job's own session / process group
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, sig)

    def pause(self, now: float) -> None:
        if self.paused:
            return
        self.pauses.append([now, None])
        self._save()  # record first, so a crash leaves a visible open interval
        self._signal(signal.SIGSTOP)

    def resume(self, now: float) -> None:
        if not self.paused:
            return
        self.pauses[-1][1] = now
        self._signal(signal.SIGCONT)
        self._save()

    def step(self, gate: ServingGate, serving_ok: bool, now: float) -> None:
        """Stop on production traffic; continue after the idle-resume window."""
        if serving_ok or not gate.enabled:
            self.resume(now)
        elif self.paused:
            if gate.may_resume(now):
                self.resume(now)
        elif gate.busy:
            self.pause(now)


# ---------------------------------------------------------------- queue


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


def _pick(jobs: list[dict], eligible=None) -> dict | None:
    """Highest priority first; within it, the owner that last ran longest ago
    (round-robin across agents), then that owner's oldest job. `eligible` filters
    out jobs the serving gate / memory admission currently blocks."""
    pending = [j for j in jobs if j["state"] == "pending"]
    if eligible is not None:
        pending = [j for j in pending if eligible(j)]
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
    serving_ok: bool = False,
    mem_gb: float | None = None,
) -> str:
    JOBS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    jid = _new_id(label)
    if mem_gb is None:
        mem_gb = 0.0 if serving_ok else DEFAULT_MEM_GB
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
            "serving_ok": serving_ok,
            "mem_gb": mem_gb,
            "submitted": _now(),
            "state": "pending",
        },
    )
    _ensure_daemon()
    return jid


def _run_one(job: dict, path: Path, gate: ServingGate | None = None) -> None:
    gate = gate or ServingGate()
    job.update(state="running", started=_now(), pid=None)
    log = open(LOGS / f"{job['id']}.log", "w")  # noqa: SIM115 - closed after the job ends
    pause_file = LOGS / f"{job['id']}.pauses.json"
    _write(pause_file, {"pauses": []})
    try:
        rc_file = LOGS / f"{job['id']}.rc"
        proc = subprocess.Popen(
            ["/bin/sh", "-c", '"$@"; echo $? > "$GPUQ_RC"', "sh", *job["cmd"]],
            cwd=job["cwd"],
            env={
                **job["env"],
                "GPUQ_RC": str(rc_file),
                "GPUQ_PAUSE_FILE": str(pause_file),
            },
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
    job["pauses"] = []
    job["waiting"] = None
    _write(path, job)
    pauser = Pauser(job, path)
    serving_ok = bool(job.get("serving_ok"))
    rc, why = None, None
    last_size, last_growth = -1, _now()
    while rc is None:
        try:
            rc = proc.wait(timeout=min(POLL_S, gate.poll_s) if gate.enabled else POLL_S)
        except subprocess.TimeoutExpired:
            now = _now()
            if gate.enabled:
                gate.poll(now)
            was_paused = pauser.paused
            cancelled = bool(_read(path).get("cancel"))
            if cancelled:
                pauser.resume(now)  # a stopped group cannot act on SIGINT
            else:
                pauser.step(gate, serving_ok, now)
            if was_paused and not pauser.paused:
                last_growth = now  # paused time is not stall time
            if pauser.paused:
                continue  # timeouts and stall detection only count active time
            deadline = job["started"] + job["timeout_s"] + pauser.total(now)
            log_path = LOGS / f"{job['id']}.log"
            size = log_path.stat().st_size if log_path.exists() else 0
            if size != last_size:
                last_size, last_growth = size, now
            if cancelled:
                why = "cancelled"
            elif now > deadline:
                why = "timeout"
            elif now - last_growth > job.get("stall_s", STALL_S):
                why = "stalled"
            if why:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
    state = why or ("done" if rc == 0 else "failed")
    ended = _now()
    if pauser.paused:  # the job died while stopped: close the interval
        pauser.pauses[-1][1] = ended
        pauser.job["pauses"], pauser.job["paused"] = pauser.pauses, False
    job.update(state=state, rc=rc, ended=ended, paused=False)
    log.write(
        f"\ngpuq: {state} rc={rc} after {ended - job['started']:.0f}s "
        f"({len(pauser.pauses)} pauses, {pauser.total(ended):.0f}s paused)\n"
    )
    log.close()
    _write(LOGS / f"{job['id']}.pauses.json", {"pauses": pauser.pauses})
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


def _adopt(job: dict, path: Path, gate: ServingGate | None = None) -> None:
    gate = gate or ServingGate()
    pid = job.get("pid")
    pauser = Pauser(job, path)
    serving_ok = bool(job.get("serving_ok"))
    why = None
    while _alive(pid):
        now = _now()
        if gate.enabled:
            gate.poll(now)
        cancelled = bool(_read(path).get("cancel"))
        if cancelled:
            pauser.resume(now)
        else:
            pauser.step(gate, serving_ok, now)
        deadline = (
            job.get("started", now) + job.get("timeout_s", 1200) + pauser.total(now)
        )
        if not why and (cancelled or (now > deadline and not pauser.paused)):
            why = "cancelled" if cancelled else "timeout"
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGINT)
        time.sleep(min(POLL_S, gate.poll_s) if gate.enabled else POLL_S)
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
    if pauser.paused:
        pauser.pauses[-1][1] = _now()
    job.update(
        state=state,
        rc=rc,
        ended=_now(),
        adopted=True,
        pauses=pauser.pauses,
        paused=False,
    )
    _write(path, job)


def daemon() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / "daemon.lock", "a")  # noqa: SIM115 - held for the daemon's lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    idle_since = _now()
    gate = ServingGate()
    while True:
        gate.configure(load_serving_config())
        # A job left running by a previous daemon is adopted: wait for its
        # process group to exit (still under timeout / cancel), then record it.
        for j in _jobs():
            if j["state"] == "running":
                _adopt(j, JOBS / f"{j['id']}.json", gate)
        if gate.enabled:
            gate.poll()
        for (
            j
        ) in _jobs():  # a cancelled pending job must go even while the gate blocks it
            if j["state"] == "pending" and j.get("cancel"):
                _patch_job(JOBS / f"{j['id']}.json", state="cancelled", ended=_now())
        jobs = _jobs()
        job = _pick(jobs, lambda j: _blocker(j, gate) is None)
        pending = [j for j in jobs if j["state"] == "pending"]
        for j in pending:  # surface why a job is waiting (gpuq status)
            want = _blocker(j, gate)
            if j.get("waiting") != want:
                _patch_job(JOBS / f"{j['id']}.json", waiting=want)
        if job is None:
            if pending:
                idle_since = _now()  # blocked jobs are not an idle queue
            elif _now() - idle_since > IDLE_EXIT_S:
                return
            time.sleep(min(POLL_S, gate.poll_s) if gate.enabled else POLL_S)
            continue
        path = JOBS / f"{job['id']}.json"
        if _read(path).get("cancel"):
            job.update(state="cancelled", ended=_now())
            _write(path, job)
            continue
        print(f"{time.strftime('%H:%M:%S')} run {job['id']}", flush=True)
        try:
            _run_one(job, path, gate)
        except Exception as e:  # noqa: BLE001 - a full disk etc. must not stop the queue
            print(
                f"{time.strftime('%H:%M:%S')} job error {job['id']}: {e!r}", flush=True
            )
            with contextlib.suppress(Exception):
                job.update(state="failed", rc=None, ended=_now(), error=repr(e))
                _write(path, job)
        print(f"{time.strftime('%H:%M:%S')} end {job['id']} {job['state']}", flush=True)
        idle_since = _now()


def _patch_job(path: Path, **kw) -> None:
    data = _read(path)
    if data:
        data.update(kw)
        _write(path, data)


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


def display_state(j: dict) -> str:
    """State column: running jobs may be 'paused'; pending ones 'wait-idle' / 'wait-mem'."""
    if j["state"] == "running" and j.get("paused"):
        return "paused"
    if j["state"] == "pending" and j.get("waiting"):
        return f"wait-{j['waiting']}"
    return j["state"]


def status() -> None:
    now = _now()
    rows = _jobs()
    active = [j for j in rows if j["state"] in ("pending", "running")]
    recent = [
        j for j in rows if j["state"] in FINAL and now - j.get("ended", 0) < 6 * 3600
    ]
    print(f"daemon: {'up' if _daemon_running() else 'down'}")
    cfg = load_serving_config()
    if cfg.get("urls"):
        print(f"serving gate: {', '.join(cfg['urls'])}")
    for j in active + recent[-15:]:
        t0 = j.get("started", j["submitted"])
        age = (j.get("ended") or now) - t0
        np = len(j.get("pauses", []))
        extra = f"  pauses={np}" if np else ""
        print(
            f"{display_state(j):>9}  {age:6.0f}s  p{j.get('priority', 0)}  {j['id']}{extra}"
        )


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
        p.add_argument(
            "--serving-ok",
            action="store_true",
            help="bypass the production-idle gate (CPU-only jobs); never paused",
        )
        p.add_argument(
            "--mem-gb",
            type=float,
            default=None,
            help=f"unified memory the job needs (default {DEFAULT_MEM_GB:g}, 0 with --serving-ok)",
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
        jid = submit(
            cmd, a.label, a.timeout, a.priority, a.stall, a.serving_ok, a.mem_gb
        )
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
