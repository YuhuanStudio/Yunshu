"""CPU contamination contracts for gpuq and harnesses (stdlib, Python 3.9).

CPU is summed in percent of one core, never divided by the machine's core count.
ps supplies the initial estimate; subsequent samples use CPU-time deltas over the
actual wall interval. A job's descendants and process group are excluded, including
children reparented while its session remains alive. Sampling errors fail closed.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

MAX_POLL_GAP_S = 5.0
STALE_FLAG_S = 10.0

DEFAULTS = dict(threshold_pct=150.0, window_s=20.0, max_wait_s=300.0, sample_s=30.0)
ENV = dict(
    threshold_pct="GPUQ_CPU_THRESHOLD",
    window_s="GPUQ_QUIET_WINDOW_S",
    max_wait_s="GPUQ_QUIET_MAX_WAIT_S",
    sample_s="GPUQ_CPU_SAMPLE_S",
)


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


def process_snapshot():
    return subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,pgid=,pcpu=,time=,comm="],
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


class CpuSampler:
    def __init__(self):
        self.previous = {}
        self.previous_time = None
        self.owned = set()

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
        )
        with contextlib.suppress(OSError):
            row["load_1m"], row["load_5m"], _ = os.getloadavg()
        try:
            processes = {}
            for line in process_snapshot().splitlines():
                fields = line.split(None, 5)
                if len(fields) != 6:
                    raise ValueError("incomplete ps row")
                p, parent, group, pct, seconds, command = fields
                processes[int(p)] = (
                    int(parent),
                    int(group),
                    float(pct),
                    _cpu_seconds(seconds),
                    Path(command).name,
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
            dt = now - since
            for p, (_, _, pct, seconds, name) in processes.items():
                if dt > 0 and p in self.previous:
                    pct = max(0.0, seconds - self.previous[p]) * 100 / dt
                elif dt > 0 and p not in self.previous:
                    # A new process cannot have used more than one interval of
                    # its total CPU time. It can use several cores in that interval.
                    pct = seconds * 100 / dt
                if p not in self.owned:
                    consumers.append(dict(pid=p, name=name, cpu_pct=round(pct, 2)))
            row["foreign_cpu_pct"] = round(sum(p["cpu_pct"] for p in consumers), 2)
            row["top_cpu"] = sorted(consumers, key=lambda p: -p["cpu_pct"])[:8]
            self.previous = {p: v[3] for p, v in processes.items()}
            self.previous_time = now
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            row["error"] = f"{type(e).__name__}: {e}"
        return row


class QuietGate:
    """One shared pending snapshot per poll, individual continuous quiet windows."""

    def __init__(self):
        self.sampler = CpuSampler()
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
            job.update(quiet_since=None, quiet_wait_started=now, quiet_timeout=False)
        job["quiet_last_poll"] = now
        job.setdefault("quiet_wait_started", now)
        cpu = sample["foreign_cpu_pct"]
        if cpu is None or cpu >= cfg["threshold_pct"]:
            job["quiet_since"] = None
        elif job.get("quiet_since") is None:
            job["quiet_since"] = now
        if (
            job.get("quiet_since") is not None
            and now - job["quiet_since"] >= cfg["window_s"]
        ):
            job["quiet_timeout"] = False
            return False
        if now - job["quiet_wait_started"] >= cfg["max_wait_s"]:
            job["quiet_timeout"] = True
            return False
        return True


class ContentionMonitor:
    def __init__(self, job, path, logs, write, read):
        self.job, self.path, self.write, self.read = job, path, write, read
        self.flag = logs / (job["id"] + ".contention.json")
        self.cfg = contention_config(job)
        self.sampler = CpuSampler()
        self.samples = list(job.get("cpu_samples", []))
        self.events = list(job.get("contention_events", []))
        self.count = job.get("foreign_cpu_sample_count", 0)
        self.total = job.get("foreign_cpu_sum_pct", 0.0)
        self.maximum = job.get("foreign_cpu_max_pct", 0.0)
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
        )
        self.job.update(fields)
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
            f"samples={self.count} reasons={self.job.get('contention_reasons', [])}"
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
