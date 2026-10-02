"""GPU job queue: one GPU, many benchmark jobs, run one at a time.

Benchmarks on a single Apple GPU are only meaningful when nothing else is on it,
so every GPU job goes through this queue. ``submit`` returns at once; a single
background daemon (started on demand, exits when idle) runs jobs in priority,
then FIFO order, each under a timeout, with output in a log file. Waiting is a
plain local process (``wait``), so whoever submitted can block on completion
without polling anything else.

    gpuq submit [--label L] [--timeout MIN] [--priority N] [--serving-ok] [--mem-gb G] -- cmd args...
    gpuq wait [--max-seconds N] ID [ID ...]  # exit 0 verified / 1 failed / 2 unfinished / 3 contended perf
    gpuq run [opts] -- cmd ...   # submit + wait (drop-in for the old gpu_run.sh)
    gpuq status                  # queue table (paused / waiting-idle / waiting-mem columns)
    gpuq log ID                  # print a job's log
    gpuq cancel ID               # drop a pending job or stop a running one
    gpuq digest [--label-prefix X] [--since 6h] [--peek]  # jobs finished since the last digest, failures and empty outputs flagged

Serving awareness (optional; docs/guides/SERVE_AND_DEVELOP.md): with production server URLs configured
(GPUQ_SERVING_URLS, or $GPUQ_DIR/serving.json) a job starts only after every server has been idle for
IDLE_START seconds, is SIGSTOPped (its own process group only) the moment a production request appears, and
SIGCONTed after IDLE_RESUME idle seconds. Pause intervals go to the job JSON and to the file named by
$GPUQ_PAUSE_FILE (read them with scripts/dev/gpuq_pause.py). ``--serving-ok`` bypasses the gate (CPU-only
jobs, still subject to priority preemption); ``--mem-gb`` is checked against free memory minus the production reserve. The poller reads only the
request counts from /v1/yunshu/status, never prompts.

Priority preemption (Q01): pending p>=0 jobs can pause a running p<=-1 job when free memory admits both
resident jobs, even without a serving config. Unknown/insufficient memory leaves the running job alone.
Submit-level repeatable ``--out PATH`` declares expected results; ``--expect-complete`` requires a line containing
``complete`` in each output. Wait and digest validate declared and legacy command-level output paths.
Duplicate active labels are refused; submitting never deletes earlier artifacts.

CPU contention (Q02): every job records load averages and the top foreign CPU consumers at start,
periodically and end, excluding its own descendants/process group. --quiet jobs (timing measurements) wait for a
continuous quiet window (default summed foreign CPU <150% for 20s, maximum wait 300s). --cpu-threshold,
--quiet-window and --quiet-max-wait override GPUQ_CPU_THRESHOLD / GPUQ_QUIET_WINDOW_S /
GPUQ_QUIET_MAX_WAIT_S. GPUQ_CPU_SAMPLE_S controls stored periodic samples (default 30s); monitoring polls
at most every 2s. Contention is latched in job.contended and GPUQ_CONTENTION_FILE; done is displayed as
contended, and wait returns 3 for otherwise successful contended perf/quiet jobs (failures still return 1).

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
import tempfile
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpuq_contention import (  # noqa: E402
    ENV,
    MAX_POLL_GAP_S,
    ContentionMonitor,
    QuietGate,
    contention_config,
    flag_sample_time,
    requires_quiet,
)


def _now() -> float:
    return time.time()


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write(path: Path, data: dict) -> None:
    # CLI cancellation and daemon updates can write the same job concurrently.
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as f:
        tmp = Path(f.name)
        try:
            json.dump(data, f, indent=1)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


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
        self.cpu = None
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
    return _cpu_blocker(job, gate)


def _cpu_blocker(job: dict, gate: ServingGate):
    if gate.cpu is None or not requires_quiet(job):
        return None
    now = _now()
    sample = gate.cpu.sample(now)
    blocked = gate.cpu.blocked(job, sample, now)
    if not blocked:
        job["cpu_admission_sample"] = sample
    _patch_job(
        JOBS / (job["id"] + ".json"),
        **{
            k: job[k]
            for k in (
                "quiet_wait_started",
                "quiet_since",
                "quiet_timeout",
                "quiet_last_poll",
            )
            if k in job
        },
    )
    return "cpu" if blocked else None


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
        data.update(
            pauses=self.pauses,
            paused=self.paused,
            pause_reason=self.job.get("pause_reason"),
        )
        _write(self.path, data)
        _write(LOGS / f"{self.job['id']}.pauses.json", {"pauses": self.pauses})

    def _signal(self, sig: int) -> None:
        pid = self.job.get("pid")
        if pid and pid > 1:  # only ever the job's own session / process group
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, sig)

    def pause(self, now: float, reason: str = "serving") -> None:
        self.job["pause_reason"] = reason
        if self.paused:
            self._save()
            return
        self.pauses.append([now, None])
        self._save()  # record first, so a crash leaves a visible open interval
        self._signal(signal.SIGSTOP)

    def resume(self, now: float) -> None:
        if not self.paused:
            return
        self.pauses[-1][1] = now
        self.job["pause_reason"] = None
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


AGE_S = float(os.environ.get("GPUQ_AGE_S", 2 * 3600))


def _eff_priority(job: dict, now: float | None = None) -> int:
    """Priority with one step of aging: a job below 0 that has waited AGE_S rises by
    one (never above 0), so p-1 work is not starved forever by a stream of p0 jobs
    while deep backlog (p-3) stays behind interactive work."""
    p = job.get("priority", 0)
    if p >= 0 or job.get("state") != "pending":
        return p
    waited = (now or time.time()) - job.get("submitted", now or time.time())
    return min(0, p + 1) if waited >= AGE_S else p


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
    now = time.time()
    top = max(_eff_priority(j, now) for j in pending)
    pending = [j for j in pending if _eff_priority(j, now) == top]
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
    outputs: list[str] | None = None,
    expect_complete: bool = False,
    quiet: bool = False,
    cpu_config: dict | None = None,
) -> str:
    cfg = contention_config({"contention_config": cpu_config or {}})
    JOBS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    # Serialize label admission and id allocation across all submitters.
    with open(ROOT / "submit.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(
            j.get("label") == label and j.get("state") in ("pending", "running")
            for j in _jobs()
        ):
            raise ValueError(f"duplicate active label: {label}")
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
                "outputs": [
                    str(Path(p).expanduser().resolve()) for p in (outputs or [])
                ],
                "expect_complete": expect_complete,
                "quiet": quiet,
                "contention_config": cfg,
                "contended": False,
                "submitted": _now(),
                "state": "pending",
            },
        )
    _ensure_daemon()
    return jid


def _read_rc(jid: str) -> int | None:
    try:
        return int((LOGS / f"{jid}.rc").read_text().strip())
    except (OSError, ValueError):
        return None


def _preempt_blocker(job: dict, gate: ServingGate) -> str | None:
    """Admission while the stopped low-priority job remains resident.

    Available memory already excludes resident allocations. Unlike ordinary
    serial admission, preemption always checks memory and fails closed if unknown.
    """
    need = float(job.get("mem_gb", DEFAULT_MEM_GB))
    if need > 0:
        available = free_memory_gb()
        if available is None or available - need < gate.reserve_gb:
            return "mem"
    if gate.enabled and not job.get("serving_ok") and not gate.may_start():
        return "idle"
    return _cpu_blocker(job, gate)


def _priority_step(pauser: Pauser, gate: ServingGate, now: float) -> bool:
    """Run admitted p>=0 work inside a p<=-1 pause; never kill to reclaim memory.

    Returns True while priority work still owns the pause. The nested runner
    cannot preempt again (its priority is >=0), so at most two jobs are resident.
    """
    if pauser.job.get("priority", 0) >= 0:
        return False
    pending = [
        j
        for j in _jobs()
        if j["state"] == "pending" and _eff_priority(j) >= 0 and not j.get("cancel")
    ]
    blockers = {j["id"]: _preempt_blocker(j, gate) for j in pending}
    for j in pending:
        why = blockers[j["id"]]
        if j.get("waiting") != why:
            _patch_job(JOBS / f"{j['id']}.json", waiting=why)
    # Memory admission decides whether to pause. The serving start window only
    # decides when the admitted high-priority job can start; it must not let the
    # low-priority job resume early through the shorter serving resume window.
    if any(why != "mem" for why in blockers.values()):
        pauser.pause(now, reason="priority")
    high = _pick(pending, lambda j: blockers[j["id"]] is None)
    if high is not None:
        path = JOBS / f"{high['id']}.json"
        if not _read(path).get("cancel"):
            _execute(high, path, gate)
        return True
    return (
        bool(pending) and pauser.paused and pauser.job.get("pause_reason") == "priority"
    )


def _run_one(job: dict, path: Path, gate: ServingGate | None = None) -> None:
    gate = gate or ServingGate()
    job.update(state="running", started=_now(), pid=None)
    log = open(LOGS / f"{job['id']}.log", "w")  # noqa: SIM115 - closed after the job ends
    pause_file = LOGS / f"{job['id']}.pauses.json"
    _write(pause_file, {"pauses": []})
    monitor = ContentionMonitor(job, path, LOGS, _write, _read)
    monitor.record("start")
    try:
        rc_file = LOGS / f"{job['id']}.rc"
        proc = subprocess.Popen(
            [
                "/bin/sh",
                "-c",
                '"$@"; rc=$?; echo $rc > "$GPUQ_RC"; exit $rc',
                "sh",
                *job["cmd"],
            ],
            cwd=job["cwd"],
            env={
                **job["env"],
                **{ENV[k]: str(v) for k, v in monitor.cfg.items()},
                "GPUQ_RC": str(rc_file),
                "GPUQ_PAUSE_FILE": str(pause_file),
                "GPUQ_CONTENTION_FILE": str(monitor.flag),
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        log.write(f"gpuq: failed to start: {e}\n")
        job.update(state="failed", rc=127, ended=_now())
        monitor.record("end")
        log.write(monitor.summary() + "\n")
        _write(path, job)
        log.close()
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
            cancelled = bool(_read(path).get("cancel"))
            if cancelled:
                pauser.resume(now)  # a stopped group cannot act on SIGINT
            elif not _priority_step(pauser, gate, now):
                pauser.step(gate, serving_ok, now)
            now = _now()
            monitor.poll(now)
            if pauser.paused:
                continue  # timeouts and stall detection only count active time
            deadline = job["started"] + job["timeout_s"] + pauser.total(now)
            log_path = LOGS / f"{job['id']}.log"
            size = log_path.stat().st_size if log_path.exists() else 0
            if size != last_size:
                last_size, last_growth = size, now - pauser.total(now)
            if cancelled:
                why = "cancelled"
            elif now > deadline:
                why = "timeout"
            elif now - pauser.total(now) - last_growth > job.get("stall_s", STALL_S):
                why = "stalled"
            if why:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
    # The command's own exit status is in the rc file; the shell's status alone is not
    # trusted (a wrapper that ends in `echo` exits 0 after a crash).
    rc_file_rc = _read_rc(job["id"])
    if rc_file_rc is not None:
        rc = rc_file_rc
    elif not why:
        rc = rc if rc not in (None, 0) else None  # no rc file: unknown, not success
    state = why or ("done" if rc == 0 else "failed")
    ended = _now()
    if pauser.paused:  # the job died while stopped: close the interval
        pauser.pauses[-1][1] = ended
        pauser.job["pauses"], pauser.job["paused"] = pauser.pauses, False
    job.update(state=state, rc=rc, ended=ended, paused=False, pause_reason=None)
    log.write(
        f"\ngpuq: {state} rc={rc} after {ended - job['started']:.0f}s "
        f"({len(pauser.pauses)} pauses, {pauser.total(ended):.0f}s paused)\n"
    )
    monitor.record("end", ended)
    log.write(monitor.summary() + "\n")
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
    monitor = ContentionMonitor(job, path, LOGS, _write, _read)
    last_sample = flag_sample_time(_read(monitor.flag))
    if (
        not job.get("cpu_samples")
        or last_sample is None
        or _now() - last_sample > MAX_POLL_GAP_S
    ):
        job["contended"] = True
        job.setdefault("contention_reasons", []).append("unmonitored_before_adoption")
        monitor.events.append(
            [
                last_sample if last_sample is not None else job.get("started", _now()),
                _now(),
            ]
        )
    monitor.record("adopt")
    why = None
    while _alive(pid):
        now = _now()
        if gate.enabled:
            gate.poll(now)
        cancelled = bool(_read(path).get("cancel"))
        if cancelled:
            pauser.resume(now)
        elif not _priority_step(pauser, gate, now):
            pauser.step(gate, serving_ok, now)
        now = _now()
        monitor.poll(now)
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
        pause_reason=None,
        pauses=pauser.pauses,
        paused=False,
    )
    monitor.record("end")
    with open(LOGS / f"{job['id']}.log", "a") as log:
        log.write("\n" + monitor.summary() + "\n")
    _write(LOGS / f"{job['id']}.pauses.json", {"pauses": pauser.pauses})
    _write(path, job)


def _execute(job: dict, path: Path, gate: ServingGate) -> None:
    """Contain a runner error for normal and preempting jobs alike."""
    print(f"{time.strftime('%H:%M:%S')} run {job['id']}", flush=True)
    try:
        _run_one(job, path, gate)
    except Exception as e:  # noqa: BLE001 - a full disk etc. must not stop the queue
        print(f"{time.strftime('%H:%M:%S')} job error {job['id']}: {e!r}", flush=True)
        # Never leave an untracked GPU process behind after a runner error.
        pid = job.get("pid")
        if pid and pid > 1:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            pauser = Pauser(job, path)
            ended = _now()
            if pauser.paused:
                pauser.pauses[-1][1] = ended
            job.update(
                state="failed",
                rc=None,
                ended=ended,
                error=repr(e),
                paused=False,
                pause_reason=None,
                pauses=pauser.pauses,
            )
            _write(LOGS / f"{job['id']}.pauses.json", {"pauses": pauser.pauses})
            _write(path, job)
    print(f"{time.strftime('%H:%M:%S')} end {job['id']} {job['state']}", flush=True)


def daemon() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / "daemon.lock", "a")  # noqa: SIM115 - held for the daemon's lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    idle_since = _now()
    gate = ServingGate()
    gate.cpu = QuietGate()
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
        _execute(job, path, gate)
        idle_since = _now()


def _patch_job(path: Path, **kw) -> None:
    data = _read(path)
    if data:
        data.update(kw)
        _write(path, data)


FINAL = {"done", "failed", "timeout", "stalled", "cancelled", "lost"}


def _output_issues(job: dict) -> list[str]:
    # Keep standalone execution and importlib-based tests on the same validator.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpuq_digest import output_issues  # noqa: PLC0415

    return output_issues(job)


def wait(ids: list[str], max_seconds: float | None = None) -> int:
    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    while True:
        jobs = [(jid, _read(JOBS / f"{jid}.json")) for jid in ids]
        pending = any(j and j.get("state") not in FINAL for _, j in jobs)
        if not pending or (deadline is not None and time.monotonic() >= deadline):
            break
        delay = POLL_S * 2
        if deadline is not None:
            delay = min(delay, max(0.0, deadline - time.monotonic()))
        time.sleep(delay)
    bad = False
    contended = False
    for jid, j in jobs:
        issues = _output_issues(j) if j else ["job not found"]
        state = j.get("state", "missing")
        print(
            f"{jid}: {display_state(j) if j else state} rc={j.get('rc')} missing_outputs={json.dumps(issues)} "
            f"log={LOGS / (jid + '.log')}"
        )
        contended = contended or (bool(j.get("contended")) and requires_quiet(j))
        bad = (
            bad
            or not j
            or (
                state in FINAL and (state != "done" or j.get("rc") != 0 or bool(issues))
            )
        )
    # An unfinished job takes precedence: callers must keep waiting even if a peer failed.
    return 2 if pending else (1 if bad else (3 if contended else 0))


def display_state(j: dict) -> str:
    """State column: running jobs may be 'paused'; pending ones 'wait-idle' / 'wait-mem'."""
    if j["state"] == "running" and j.get("paused"):
        return "paused"
    if j["state"] == "pending" and j.get("waiting"):
        return f"wait-{j['waiting']}"
    if j["state"] == "done" and j.get("contended"):
        return "contended"
    return j["state"]


def status() -> None:
    now = _now()
    rows = _jobs()
    active = [j for j in rows if j["state"] in ("pending", "running")]
    recent = [
        j for j in rows if j["state"] in FINAL and now - j.get("ended", 0) < 6 * 3600
    ]
    daemon_up = _daemon_running()
    print(f"daemon: {'up' if daemon_up else 'down'}")
    cfg = load_serving_config()
    if cfg.get("urls"):
        print(f"serving gate: {', '.join(cfg['urls'])}")
    pending = [j for j in active if j["state"] == "pending"]
    running = [j for j in active if j["state"] == "running"]
    if pending and not any(not j.get("paused") for j in running):
        reasons = []
        if not daemon_up:
            reasons.append("daemon down")
        if any(j.get("paused") for j in running):
            reasons.append(
                "pause: "
                + ", ".join(
                    f"{j['id']} ({j.get('pause_reason') or 'serving'})" for j in running
                )
            )
        for j in pending:
            why = j.get("waiting")
            explanation = {
                "mem": "memory admission (free memory minus job need is below reserve, or unknown during preemption)",
                "idle": "serving (busy or idle-start window)",
                "pause": "pause",
                "cpu": "CPU contention (waiting for continuous quiet window)",
            }.get(why)
            if explanation:
                reasons.append(f"{j['id']}: {explanation}")
        print(
            "idle: "
            + "; ".join(reasons or ["awaiting daemon scheduling/admission poll"])
        )
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
            help="bypass the production-idle gate and serving pauses (CPU-only jobs)",
        )
        p.add_argument(
            "--mem-gb",
            type=float,
            default=None,
            help=f"unified memory the job needs (default {DEFAULT_MEM_GB:g}, 0 with --serving-ok)",
        )
        p.add_argument(
            "--out",
            action="append",
            default=[],
            help="expected output path (repeatable)",
        )
        p.add_argument(
            "--expect-complete",
            action="store_true",
            help="require a line containing complete in every output",
        )
        p.add_argument(
            "--quiet", action="store_true", help="timing measurement: wait for a quiet CPU before starting and treat contention as untrustworthy"
        )
        p.add_argument(
            "--cpu-threshold",
            type=float,
            help="summed foreign CPU percent (default 150)",
        )
        p.add_argument(
            "--quiet-window", type=float, help="continuous quiet seconds (default 20)"
        )
        p.add_argument(
            "--quiet-max-wait",
            type=float,
            help="maximum CPU wait seconds (default 300)",
        )
        p.add_argument("cmd", nargs=argparse.REMAINDER)
    wp = sub.add_parser("wait")
    wp.add_argument(
        "--max-seconds",
        type=float,
        help="return 2 if still pending after this many seconds",
    )
    wp.add_argument("ids", nargs="+")
    sub.add_parser("status")
    sub.add_parser("log").add_argument("id")
    sub.add_parser("cancel").add_argument("id")
    sub.add_parser("_daemon")
    sub.add_parser("digest", add_help=False)  # flags handled by gpuq_digest.py
    if sys.argv[1:2] == ["digest"]:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import gpuq_digest  # noqa: PLC0415

        return gpuq_digest.main(sys.argv[2:])
    a = ap.parse_args()
    if a.op in ("submit", "run"):
        cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        if not cmd:
            ap.error("missing command after --")
        try:
            jid = submit(
                cmd,
                a.label,
                a.timeout,
                a.priority,
                a.stall,
                a.serving_ok,
                a.mem_gb,
                a.out,
                a.expect_complete,
                a.quiet,
                {
                    k: v
                    for k, v in dict(
                        threshold_pct=a.cpu_threshold,
                        window_s=a.quiet_window,
                        max_wait_s=a.quiet_max_wait,
                    ).items()
                    if v is not None
                },
            )
        except ValueError as e:
            print(f"gpuq: {e}", file=sys.stderr)
            return 1
        print(jid, flush=True)
        return wait([jid]) if a.op == "run" else 0
    if a.op == "wait":
        if a.max_seconds is not None and a.max_seconds < 0:
            ap.error("--max-seconds must be nonnegative")
        return wait(a.ids, a.max_seconds)
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
