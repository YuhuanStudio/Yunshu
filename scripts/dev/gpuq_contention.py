"""CPU contamination contracts for gpuq and harnesses (stdlib, Python 3.9).

CPU is summed in percent of one core, never divided by the machine's core count.
ps supplies the initial estimate; subsequent samples use CPU-time deltas over the
actual wall interval. A job's descendants and process group are excluded, including
children reparented while its session remains alive. Sampling errors fail closed.
Foreign processes with nice >= 10 are background work and excluded.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from gpuq_swap import host_available_bytes, memory_verdict, process_memory

MAX_POLL_GAP_S = 5.0
STALE_FLAG_S = 10.0

# Foreign CPU (percent of one core) at which a timing run counts as contended: only a
# machine that is nearly saturated, 90% of all cores (1620% on an 18-core M5 Max). Lower
# thresholds flagged ordinary background work that did not move GPU-bound timings.
_DEFAULT_THRESHOLD = 90.0 * (os.cpu_count() or 8)
DEFAULTS = dict(
    threshold_pct=_DEFAULT_THRESHOLD,
    window_s=20.0,
    max_wait_s=300.0,
    sample_s=30.0,
    foreign_model_cpu_pct=15.0,
    swap_mb=64.0,
    host_swap_mb=1073.741824,  # 1 GiB, decimal MB like the existing job JSON
    memory_floor_mb=4096.0,  # 4 GB free+inactive; independent of old swap stock
)
ENV = dict(
    threshold_pct="GPUQ_CPU_THRESHOLD",
    window_s="GPUQ_QUIET_WINDOW_S",
    max_wait_s="GPUQ_QUIET_MAX_WAIT_S",
    sample_s="GPUQ_CPU_SAMPLE_S",
    foreign_model_cpu_pct="GPUQ_FOREIGN_MODEL_CPU",
    swap_mb="GPUQ_SWAP_MB",
    host_swap_mb="GPUQ_HOST_SWAP_MB",
    memory_floor_mb="GPUQ_MEMORY_FLOOR_MB",
)

# Command lines of MLX model runtimes started outside gpuq. GPUQ_FOREIGN_MODEL_PATTERNS
# adds comma-separated regexes. Such a server can use the GPU while a timing job runs
# without ever reaching the summed-CPU threshold.
FOREIGN_MODEL_PATTERNS = (
    r"\byunshu(_cli)?\b.*\bserve\b",
    r"\buvicorn\b.*\byunshu_gateway\b",
    r"\bmlx_lm[./ ](server|generate)\b",
    r"\bmlx_vlm\b",
    r"\bmlx_audio\b",
)
_GPUQ_CMD = re.compile(r"(^|[/\s])gpuq(\.py)?(\s|$)")
# scripts/dev/ci-local runs the unit suite from a fresh clone under /tmp/ycl-<sha>: CPU only
# (the sandbox has no models), so its CLI and server-test processes are never GPU users.
_CI_SANDBOX = re.compile(r"(^|\s)/(private/)?tmp/ycl-[0-9a-f]+/")


def is_foreign_model_cmd(cmd, patterns):
    """True when `cmd` looks like a model runtime that can use the GPU. Plain `yunshu` CLI
    calls (config, model info, top) are not: only `serve` loads a model in-process."""
    if _CI_SANDBOX.search(cmd):
        return False
    return any(rx.search(cmd) for rx in patterns)


def foreign_model_patterns(env=None):
    env = os.environ if env is None else env
    extra = [
        x.strip() for x in str(env.get("GPUQ_FOREIGN_MODEL_PATTERNS", "")).split(",")
    ]
    out = []
    for p in (*FOREIGN_MODEL_PATTERNS, *[x for x in extra if x]):
        with contextlib.suppress(re.error):
            out.append(re.compile(p))
    return out


def process_args():
    """pid -> full command line (ps comm drops the arguments)."""
    text = subprocess.run(
        ["ps", "-A", "-ww", "-o", "pid=,args="],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout
    out = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            out[int(parts[0])] = parts[1]
    return out


def busy_foreign_models(sample, cfg):
    limit = cfg["foreign_model_cpu_pct"]
    return [m for m in sample.get("foreign_models") or [] if m["cpu_pct"] >= limit]


def foreign_model_reason(m):
    return f"foreign model process {m['pid']} busy {m['cpu_pct']:.1f}% ({m['cmd']})"


def contention_config(job=None):
    job = job or {}
    env = job.get("env", os.environ)
    cfg = {k: float(env.get(ENV[k], v)) for k, v in DEFAULTS.items()}
    cfg.update(job.get("contention_config", {}))
    for k, v in cfg.items():
        if not math.isfinite(v) or v < 0 or (k == "sample_s" and v == 0):
            raise ValueError(f"invalid CPU contention setting: {k}")
    return cfg


def requires_quiet(job):
    # Only jobs that compare timings ask for a quiet CPU (--quiet); correctness and
    # smoke jobs run at once. Contention is still sampled and recorded for every job.
    return bool(job.get("quiet"))


def swapout_bytes():
    """Lifetime HOST swap-IN bytes (legacy name), or None when unreadable.

    Old swapped-out apps may page in without job pressure. This counter is only
    the large host backstop; job footprint and free+inactive are independent gates.
    """
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
        page = int(re.search(r"page size of (\d+) bytes", out).group(1))
        pages = int(re.search(r"Swapins:\s+(\d+)", out).group(1))
        return pages * page
    except (OSError, subprocess.SubprocessError, AttributeError, ValueError):
        return None


def process_snapshot():
    return subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,pgid=,nice=,pcpu=,time=,comm="],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout


def _cpu_seconds(value):
    days, sep, rest = value.partition("-")
    if not sep:
        days, rest = "0", days
    parts = [float(x) for x in rest.split(":")]
    total = 0.0
    for part in parts:
        total = total * 60 + part
    return float(days) * 86400 + total


@contextlib.contextmanager
def job_lock(path):
    """Exclusive per-job lock for read-modify-write of a job file (CLI cancel vs daemon updates)."""
    with open(str(path) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class CpuSampler:
    def __init__(self, env=None):
        self.previous = {}
        self.previous_time = None
        self.owned = set()
        self.env = env

    def sample(self, pid=None, now=None):
        now = time.time() if now is None else now
        since = self.previous_time if self.previous_time is not None else now
        row = dict(
            time=now,
            since=since,
            foreign_cpu_pct=None,
            load_1m=None,
            load_5m=None,
            top_cpu=[],
            foreign_models=[],
        )
        with contextlib.suppress(OSError):
            row["load_1m"], row["load_5m"], _ = os.getloadavg()
        try:
            processes = {}
            for line in process_snapshot().splitlines():
                fields = line.split(None, 6)
                if len(fields) != 7:
                    raise ValueError("incomplete ps row")
                p, parent, group, nice, pct, seconds, command = fields
                processes[int(p)] = (
                    int(parent),
                    int(group),
                    float(pct),
                    _cpu_seconds(seconds),
                    Path(command).name,
                    int(nice),
                )
            if not processes:
                raise ValueError("empty ps snapshot")
            # Drop exited children so PID reuse cannot hide a foreign consumer.
            self.owned.intersection_update(processes)
            if pid:
                self.owned.add(pid)
                self.owned.update(p for p, v in processes.items() if v[1] == pid)
                while True:
                    children = {p for p, v in processes.items() if v[0] in self.owned}
                    if children <= self.owned:
                        break
                    self.owned.update(children)
            consumers = []
            pcts = {}
            dt = now - since
            for p, (_, _, pct, seconds, name, nice) in processes.items():
                if dt > 0 and p in self.previous:
                    pct = max(0.0, seconds - self.previous[p]) * 100 / dt
                elif dt > 0 and p not in self.previous:
                    # A new process cannot have used more than one interval of
                    # its total CPU time. It can use several cores in that interval.
                    pct = seconds * 100 / dt
                pcts[p] = pct
                if p not in self.owned and nice < 10:
                    consumers.append(dict(pid=p, name=name, cpu_pct=round(pct, 2)))
            row["foreign_cpu_pct"] = round(sum(p["cpu_pct"] for p in consumers), 2)
            row["top_cpu"] = sorted(consumers, key=lambda p: -p["cpu_pct"])[:8]
            row["foreign_models"] = self._foreign_models(processes, pcts)
            self.previous = {p: v[3] for p, v in processes.items()}
            self.previous_time = now
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            row["error"] = f"{type(e).__name__}: {e}"
        return row

    def _foreign_models(self, processes, pcts):
        try:
            patterns = foreign_model_patterns(self.env)
            args = process_args()
        except (OSError, ValueError, subprocess.SubprocessError):
            return []
        found = []
        for p, cmd in args.items():
            if p not in processes or p in self.owned or processes[p][5] >= 10:
                continue
            if p == os.getpid() or _GPUQ_CMD.search(cmd.split(" -- ")[0]):
                continue
            if is_foreign_model_cmd(cmd, patterns):
                found.append(
                    dict(pid=p, cmd=cmd[:120], cpu_pct=round(pcts.get(p, 0.0), 2))
                )
        return sorted(found, key=lambda m: -m["cpu_pct"])


class QuietGate:
    """One shared pending snapshot per poll, individual continuous quiet windows."""

    def __init__(self):
        self.sampler = CpuSampler(env=None)
        self.cached = None

    def sample(self, now):
        if self.cached is None or now - self.cached["time"] >= 1.0:
            self.cached = self.sampler.sample(now=now)
        return self.cached

    def blocked(self, job, sample, now):
        if not requires_quiet(job):
            return False
        cfg = contention_config(job)
        last = job.get("quiet_last_poll")
        if last is not None and now - last > MAX_POLL_GAP_S:
            # Waiting behind another GPU job is not observed CPU admission time.
            job["quiet_since"] = None
            if job.get("quiet_hold_started") is None:
                job.update(quiet_wait_started=now, quiet_timeout=False)
        job["quiet_last_poll"] = now
        job.setdefault("quiet_wait_started", now)
        cpu = sample["foreign_cpu_pct"]
        if (
            cpu is None
            or cpu >= cfg["threshold_pct"]
            or busy_foreign_models(sample, cfg)
        ):
            job["quiet_since"] = None
        elif job.get("quiet_since") is None:
            job["quiet_since"] = now
        if (
            job.get("quiet_since") is not None
            and now - job["quiet_since"] >= cfg["window_s"]
        ):
            job["quiet_timeout"] = False
            return False
        start = job.get("quiet_hold_started")
        limit = cfg["max_wait_s"]
        if start is not None:
            limit = min(limit, job["quiet_hold_limit_s"])
        else:
            start = job["quiet_wait_started"]
        if now - start >= limit:
            job["quiet_timeout"] = True
            return False
        return True


def _process_gone(pid):
    """True when pid has exited or is a zombie (its memory is already released)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    try:
        stat = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return not stat or stat.startswith("Z")


def _read_memory(pid, retry_s=0.2):
    """One process's memory row; None if it exited meanwhile; False if unreadable.

    Short-lived children (git, probes) often exit between the ps snapshot and the
    footprint read; 4 of the first 19 jobs under policy 2 latched job_swap_unreadable
    that way. A still-live process that stays unreadable after one retry is not
    evidence of zero swap and still fails closed.
    """
    for attempt in range(2):
        try:
            return process_memory(pid)
        except (OSError, ValueError, subprocess.SubprocessError):
            if _process_gone(pid):
                return None
            if attempt == 0:
                time.sleep(retry_s)
    return False


class ContentionMonitor:
    def __init__(self, job, path, logs, write, read):
        self.job, self.path, self.write, self.read = job, path, write, read
        self.flag = logs / (job["id"] + ".contention.json")
        self.cfg = contention_config(job)
        self.sampler = CpuSampler(env=job.get("env"))
        self.models = {m["pid"]: m for m in job.get("foreign_model_processes", [])}
        self.samples = list(job.get("cpu_samples", []))
        self.events = list(job.get("contention_events", []))
        self.count = job.get("foreign_cpu_sample_count", 0)
        self.total = job.get("foreign_cpu_sum_pct", 0.0)
        self.maximum = job.get("foreign_cpu_max_pct", 0.0)
        self.memory_processes = dict(job.get("memory_processes", {}))
        self.last_poll = None
        self.last_saved_sample = None
        if job.get("quiet_timeout"):
            job["contended"] = True
            job.setdefault("contention_reasons", []).append("quiet_window_timeout")
            self.events.append(
                [job.get("started", time.time()), job.get("started", time.time())]
            )

    def record(self, phase, now=None):
        now = time.time() if now is None else now
        row = self.sampler.sample(self.job.get("pid"), now=now)
        if phase == "start" and self.job.get("cpu_admission_sample"):
            # Admission already has interval-based CPU history. Avoid replacing
            # a verified quiet start with ps's historical first-sample estimate.
            row.update(self.job["cpu_admission_sample"])
        row["phase"] = phase
        self._check_swap(phase, row["since"], now)
        cpu = row["foreign_cpu_pct"]
        if cpu is not None:
            self.count += 1
            self.total += cpu
            self.maximum = max(self.maximum, cpu)
        if cpu is None or cpu >= self.cfg["threshold_pct"]:
            self.job["contended"] = True
            reason = "sampling_error" if cpu is None else "foreign_cpu"
            reasons = self.job.setdefault("contention_reasons", [])
            if reason not in reasons:
                reasons.append(reason)
            self.events.append([row["since"], row["time"]])
        for m in row.get("foreign_models") or []:
            seen = self.models.setdefault(
                m["pid"], dict(pid=m["pid"], cmd=m["cmd"], max_cpu_pct=0.0)
            )
            seen["max_cpu_pct"] = max(seen["max_cpu_pct"], m["cpu_pct"])
        for m in busy_foreign_models(row, self.cfg):
            self.job["contended"] = True
            reasons = self.job.setdefault("contention_reasons", [])
            reason = foreign_model_reason(m)
            if not any(
                r.startswith(f"foreign model process {m['pid']} ") for r in reasons
            ):
                reasons.append(reason)
            self.events.append([row["since"], row["time"]])
        if (
            phase != "poll"
            or self.last_saved_sample is None
            or now - self.last_saved_sample >= self.cfg["sample_s"]
        ):
            row["phase"] = "periodic" if phase == "poll" else phase
            self.samples.append(row)
            self.last_saved_sample = now
        fields = dict(
            contended=bool(self.job.get("contended")),
            contention_reasons=self.job.get("contention_reasons", []),
            cpu_samples=self.samples,
            contention_events=self.events,
            foreign_cpu_max_pct=self.maximum,
            foreign_cpu_mean_pct=self.total / self.count if self.count else None,
            foreign_cpu_sum_pct=self.total,
            foreign_cpu_sample_count=self.count,
            contention_config=self.cfg,
            foreign_model_processes=list(self.models.values()),
            swapouts_start=self.job.get("swapouts_start"),
            swap_mb=self.job.get("swap_mb", 0.0),
            host_swap_mb=self.job.get("swap_mb", 0.0),
            job_swap_mb=self.job.get("job_swap_mb"),
            available_min_mb=self.job.get("available_min_mb"),
            memory_processes=self.memory_processes,
            memory_policy_version=2,
        )
        self.job.update(fields)
        with job_lock(self.path):
            data = self.read(self.path)
            data.update(fields)
            self.write(self.path, data)
        self.write(
            self.flag,
            dict(
                contended=fields["contended"],
                events=self.events,
                last_sample=row,
                threshold_pct=self.cfg["threshold_pct"],
            ),
        )
        self.last_poll = now

    def _check_swap(self, phase, since, now):
        cur = swapout_bytes()
        available = host_available_bytes()
        if cur is not None:
            if self.job.get("swapouts_start") is None:
                self.job["swapouts_start"] = cur
            self.job["swap_mb"] = round(
                max(
                    self.job.get("swap_mb", 0.0),
                    (cur - self.job["swapouts_start"]) / 1e6,
                ),
                6,
            )
        if available is not None:
            self.job["available_min_mb"] = min(
                self.job.get("available_min_mb", available / 1e6), available / 1e6
            )
        readable = True
        for pid in self.sampler.owned:
            row = _read_memory(pid)
            if row is None:  # exited (or a zombie) between ps and footprint
                continue
            if row is False:
                readable = False
                continue
            key = f"{pid}:{row['start']}"
            old = self.memory_processes.get(key, {})
            row["swapped_peak_bytes"] = max(
                old.get("swapped_peak_bytes", 0), row["swapped_bytes"]
            )
            self.memory_processes[key] = row
        if not self.sampler.owned and phase != "start":
            readable = False
        if phase != "start" and not self.memory_processes:
            readable = False
        # Sum per-process peaks, including exited children: conservative for
        # non-overlapping lifetimes; never subtract a child's swap when it exits.
        own_mb = (
            sum(r["swapped_peak_bytes"] for r in self.memory_processes.values()) / 1e6
        )
        self.job["job_swap_mb"] = round(own_mb, 6) if readable else None
        failures = memory_verdict(
            self.job.get("swap_mb") if cur is not None else None,
            self.job["job_swap_mb"],
            self.job.get("available_min_mb") if available is not None else None,
            self.cfg,
        )
        for reason in failures:
            self.job["contended"] = True
            reasons = self.job.setdefault("contention_reasons", [])
            if reason not in reasons:
                reasons.append(reason)
            self.events.append([since, now])

    def poll(self, now):
        if self.last_poll is None or now - self.last_poll >= min(
            2.0, self.cfg["sample_s"]
        ):
            self.record("poll", now)

    def summary(self):
        mean = self.job.get("foreign_cpu_mean_pct")
        state = "contended" if self.job.get("contended") else "quiet"
        mean_text = "unknown" if mean is None else f"{mean:.1f}"
        return (
            f"gpuq: CPU {state}; foreign CPU max={self.maximum:.1f}% "
            f"mean={mean_text}% threshold={self.cfg['threshold_pct']:.1f}% "
            f"samples={self.count} host_swap={self.job.get('swap_mb', 0.0):g}MB "
            f"job_swap={self.job.get('job_swap_mb')}MB "
            f"available_min={self.job.get('available_min_mb')}MB "
            f"reasons={self.job.get('contention_reasons', [])}"
        )


def flag_sample_time(data):
    """Last monitoring timestamp, or None for a missing/structurally bad flag."""
    try:
        value = float(data["last_sample"]["time"])
        return value if math.isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def was_contended(t0=None, t1=None, path=None):
    """Latched job flag, or overlap with a single attempt. Bad files fail closed."""
    path = path or os.environ.get("GPUQ_CONTENTION_FILE")
    if not path:
        return False
    try:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            return True
        last = flag_sample_time(data)
        if "last_sample" in data and last is None:
            return True
        contended = bool(data["contended"])
        if last is not None and time.time() - float(last) > STALE_FLAG_S:
            return True
        if t0 is None or t1 is None:
            return contended
        if "events" not in data:
            return contended
        return any(a <= t1 and b >= t0 for a, b in data["events"])
    except (OSError, ValueError, TypeError, KeyError):
        return True


def wait_for_quiet(pid=None):
    """Bounded quiet wait before retry; never clear the latched job flag."""
    cfg = contention_config()
    gate = QuietGate()
    job = dict(priority=0, contention_config=cfg)
    while True:
        now = time.time()
        if not gate.blocked(job, gate.sampler.sample(pid, now), now):
            return not job.get("quiet_timeout", False)
        time.sleep(min(2.0, max(0.1, cfg["max_wait_s"])))


def server_path_attempts(run, before_retry=None):
    """Retry a contended timing result once. Text mismatch/errors still fail."""
    attempts = []
    for n in range(2):
        row = run()
        attempts.append(dict(row))
        correctness_ok = row.get("same_text") is True and row.get("spec_match", True)
        if row.get("status") not in ("PASS", "FAIL") or not correctness_ok:
            result = dict(row, status="FAIL")
            break
        if not row.get("contended"):
            result = dict(row)
            break
        result = dict(row, status="CONTENDED")
        if n == 0 and before_retry:
            before_retry()
    result["attempts"] = attempts
    return result
