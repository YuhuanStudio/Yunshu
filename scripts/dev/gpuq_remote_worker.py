"""Remote supervisor: shared POSIX lock, cancellation tombstone and owned groups."""

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

p = json.loads(sys.argv[1])
os.environ.update(p["env"])
os.makedirs(p["env"]["TMPDIR"], exist_ok=True)
if os.getpgrp() != os.getpid():
    os.setsid()
proc = None
rc = 125
deadline = time.monotonic() + p["timeout"]


def stop(*args):
    raise InterruptedError("remote job cancelled")


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGHUP, stop)
Path(p["pid"]).write_text(str(os.getpid()))
with open(p["lock"], "a") as lock:
    try:
        while True:
            if os.path.exists(p["pid"] + ".cancel"):
                raise InterruptedError("cancelled before lock admission")
            if time.monotonic() >= deadline:
                raise TimeoutError("remote lock timeout")
            try:
                fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        if os.path.exists(p["pid"] + ".cancel"):
            raise InterruptedError("cancelled before command start")
        proc = subprocess.Popen(
            p["cmd"], cwd=p["cwd"], env=p["env"], start_new_session=True
        )
        rc = proc.wait(timeout=max(0.01, deadline - time.monotonic()))
    except (subprocess.TimeoutExpired, TimeoutError):
        rc = 124
    except InterruptedError:
        rc = 130
    except Exception:
        traceback.print_exc()
    finally:
        # Keep the lock until the command group is gone, including on cancellation.
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        # Descendants that left the group (own session, e.g. a test server) carry the tag.
        tag = p["env"].get("GPUQ_JOB_ID")
        if tag:
            out = subprocess.run(
                ["ps", "eww", "-ax", "-o", "pid=,command="],
                capture_output=True,
                text=True,
            ).stdout
            for line in out.splitlines():
                pid = line.split(None, 1)[0] if line.strip() else ""
                if (
                    "GPUQ_JOB_ID=" + tag in line
                    and pid.isdigit()
                    and int(pid) != os.getpid()
                ):
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.kill(int(pid), signal.SIGKILL)
        Path(p["rc"]).write_text(str(rc))
        os.unlink(p["pid"])
