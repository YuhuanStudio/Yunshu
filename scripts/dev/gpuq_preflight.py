"""gpuq job hygiene: catch broken jobs before they queue, size timeouts from history,
and reap what a finished job leaves behind.

Most wasted GPU time came from jobs that could never succeed (a syntax error, a
missing model path, a module not importable from the job's PYTHONPATH) but only
failed after waiting in the queue and loading a 27B model, from timeouts set below
what the same kind of job is known to take, and from servers a job started that
outlived it. Python 3.9 compatible (the daemon runs on the system interpreter).
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import time
from pathlib import Path

_PY = re.compile(r"(^|/)python[\d.]*$")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_MODEL_ROOTS = ("/Volumes/P5Plus/models/", "/Volumes/Micron/models/")


def _split_env(cmd: list[str]) -> tuple[dict[str, str], list[str]]:
    """Strip a leading `env A=B ...` (and bare A=B) prefix; return (assignments, rest)."""
    env: dict[str, str] = {}
    i = 0
    if cmd and os.path.basename(cmd[0]) == "env":
        i = 1
    while i < len(cmd) and _ASSIGN.match(cmd[i]):
        k, v = cmd[i].split("=", 1)
        env[k] = v
        i += 1
    return env, cmd[i:]


def _python_target(rest: list[str]) -> tuple[str | None, str | None, str | None]:
    """(interpreter, script, module) of a python invocation, else (None, None, None)."""
    if len(rest) >= 3 and os.path.basename(rest[0]) == "uv" and rest[1] == "run":
        tail = [t for t in rest[2:] if not t.startswith("-")]
        rest = tail
    if not rest or not _PY.search(rest[0]):
        return None, None, None
    py, args = rest[0], rest[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-m" and i + 1 < len(args):
            return py, None, args[i + 1]
        if a == "-c":
            return py, None, None
        if a.startswith("-"):
            i += 1
            continue
        return py, a, None
    return py, None, None


def _resolve_py(py: str, cwd: str, env: dict[str, str]) -> str | None:
    if os.path.isabs(py):
        return py if os.path.exists(py) else None
    if "/" in py:
        p = os.path.join(cwd, py)
        return p if os.path.exists(p) else None
    for d in env.get("PATH", os.environ.get("PATH", "")).split(os.pathsep):
        p = os.path.join(d, py)
        if os.path.exists(p):
            return p
    return None


_CHECK = (
    "import importlib.util, sys\n"
    "kind, target = sys.argv[1], sys.argv[2]\n"
    "if kind == 'script':\n"
    "    compile(open(target, encoding='utf-8').read(), target, 'exec')\n"
    "elif importlib.util.find_spec(target.split('.')[0]) is None:\n"
    "    sys.exit('module not found: ' + target)\n"
)


def preflight(cmd: list[str], cwd: str, env: dict[str, str]) -> list[str]:
    """Problems that make ``cmd`` certain to fail; empty when it may run. CPU only:
    compiles the job's script with the job's interpreter (no import, no GPU)."""
    problems: list[str] = []
    if not os.path.isdir(cwd):
        return [f"cwd does not exist: {cwd}"]
    prefix, rest = _split_env(cmd)
    if not rest:
        return ["empty command"]
    job_env = {**env, **prefix}
    for i, a in enumerate(rest):
        path = None
        if a.startswith(_MODEL_ROOTS):
            path = a
        elif i > 0 and rest[i - 1] == "--model" and ("/" in a):
            path = a if os.path.isabs(a) else os.path.join(cwd, a)
        if path and not os.path.exists(path):
            problems.append(f"model path does not exist: {a}")
    py, script, module = _python_target(rest)
    if py is None:
        exe = rest[0]
        if exe.endswith(".sh") and not os.path.exists(os.path.join(cwd, exe)):
            problems.append(f"script does not exist: {exe}")
        return problems
    interp = _resolve_py(py, cwd, job_env)
    if interp is None:
        return problems + [f"interpreter not found: {py}"]
    if script is not None:
        spath = script if os.path.isabs(script) else os.path.join(cwd, script)
        if not os.path.exists(spath):
            return problems + [f"script does not exist: {script}"]
        target, kind = spath, "script"
    elif module is not None:
        target, kind = module, "module"
    else:
        return problems
    run_env = dict(job_env)
    pp = run_env.get("PYTHONPATH")
    if pp:
        run_env["PYTHONPATH"] = os.pathsep.join(
            p if os.path.isabs(p) else os.path.join(cwd, p)
            for p in pp.split(os.pathsep)
        )
    try:
        out = subprocess.run(
            [interp, "-I" if kind == "script" else "-s", "-c", _CHECK, kind, target],
            cwd=cwd,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return problems + [f"preflight could not run {interp}: {e}"]
    if out.returncode != 0:
        lines = [ln for ln in (out.stderr or out.stdout).strip().splitlines() if ln]
        problems.append(f"{kind} {target}: " + " / ".join(lines[-3:]))
    return problems


_OUT_FLAGS = ("--out", "--output", "--out-dir", "--outdir", "--log")


def ensure_out_dirs(cmd: list[str], cwd: str) -> list[str]:
    """Create the missing parent directory of every output path the job names
    (``--out F`` / ``--out=F``); a job must not fail after its queue wait because
    its results directory was never made. Returns the directories created."""
    made: list[str] = []
    for i, a in enumerate(cmd):
        path = None
        if a in _OUT_FLAGS and i + 1 < len(cmd):
            path = cmd[i + 1]
        elif a.startswith(tuple(f + "=" for f in _OUT_FLAGS)):
            path = a.split("=", 1)[1]
        if not path or path.startswith("-") or "$" in path:
            continue
        full = path if os.path.isabs(path) else os.path.join(cwd, path)
        parent = (
            full
            if a.startswith("--out-dir") or a.startswith("--outdir")
            else os.path.dirname(full)
        )
        if parent and not os.path.isdir(parent):
            with contextlib.suppress(OSError):
                os.makedirs(parent, exist_ok=True)
                made.append(parent)
    return made


# ---------------------------------------------------------------- timeouts

_NOISE = re.compile(r"^(r?\d+[a-z]?|\d{3,}[a-z0-9]*|[a-z]|[a-z]\d+|[0-9a-f]{4,})$")
# Words that say how a run went, not what it measures.
_GENERIC = frozenset(
    ["final", "pilot", "fixed", "main", "base", "cand", "new", "old", "test", "run"]
    + ["quick", "full", "retry", "again"]
)


def kind(label: str) -> frozenset[str]:
    """What a job does, without its worker prefix and run counters:
    'wide8-quality200-1' -> {'quality200'}; 'prefill6-readme-http32k-r3-0800' ->
    {'readme', 'http32k'}."""
    toks = (label or "").lower().split("-")[1:]
    return frozenset(t for t in toks if t and not _NOISE.match(t) and t not in _GENERIC)


def active_seconds(job: dict) -> float | None:
    st, en = job.get("started"), job.get("ended")
    if not st or not en:
        return None
    paused = sum((b or en) - a for a, b in job.get("pauses") or [])
    return max(0.0, en - st - paused)


def learned_timeout(
    label: str, requested_s: float, history: list[dict], now: float | None = None
) -> tuple[float, str | None]:
    """Raise ``requested_s`` to 1.5x the longest active run of the same kind of job
    in the last 7 days when it is below 1.3x of it (a timed-out run counts as 1.5x
    what it got). Returns (timeout_s, note or None)."""
    want = kind(label)
    if not want:
        return requested_s, None
    now = now or time.time()
    # A finished run says how long the kind takes; a timed-out one only bounds it
    # from below and is used when no run finished (a badly sized timed-out job
    # must not inflate every later timeout of its kind).
    best: dict[str, tuple[float, str | None]] = {
        "done": (0.0, None),
        "timeout": (0.0, None),
    }
    for j in history:
        state = j.get("state")
        if state not in best or now - (j.get("ended") or 0) > 7 * 86400:
            continue
        have = kind(j.get("label") or "")
        if not have or not (want <= have or have <= want):
            continue
        secs = active_seconds(j)
        if secs and state == "timeout":
            secs *= 1.5  # it needed more than it got
        if secs and secs > best[state][0]:
            best[state] = (secs, j.get("id"))
    longest, source = best["done"] if best["done"][0] else best["timeout"]
    if longest and requested_s < 1.3 * longest:
        new = float(int(1.5 * longest / 60 + 1) * 60)
        return (
            new,
            f"raised from {requested_s / 60:g} min: {source} ran {longest / 60:.1f} min",
        )
    return requested_s, None


# ---------------------------------------------------------------- leftovers


def leftover_pids(rc_path: str, exclude: set[int] | None = None) -> list[int]:
    """Processes still carrying this job's GPUQ_RC in their environment."""
    try:
        out = subprocess.run(
            ["ps", "eww", "-ax", "-o", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    needle = f"GPUQ_RC={rc_path}"
    pids = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            if needle + " " in parts[1] + " ":
                pids.append(int(parts[0]))
    skip = (exclude or set()) | {os.getpid()}
    return [p for p in pids if p not in skip]


def reap(rc_path: str, grace_s: float = 10.0) -> list[int]:
    """SIGTERM (then SIGKILL) every process left behind by a finished job."""
    pids = leftover_pids(rc_path)
    for p in pids:
        for sig in (signal.SIGCONT, signal.SIGTERM):
            with contextlib.suppress(OSError):
                os.kill(p, sig)
    end = time.time() + grace_s
    alive = pids
    while alive and time.time() < end:
        time.sleep(0.2)
        alive = [p for p in alive if _alive(p)]
    for p in alive:
        with contextlib.suppress(OSError):
            os.kill(p, signal.SIGKILL)
    return pids


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def rc_path_for(logs: Path, jid: str) -> str:
    return str(logs / f"{jid}.rc")
