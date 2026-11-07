"""real-1104 B2 (2026-11-05): a restart cell SIGKILLed its first server 0.4 s after the reply, before any engine
had written its disk prefix tier; every engine (Yunshu, Splash, mlx-serve, oMLX) then measured a cold prefill
after the restart. The restart now waits for the cache dirs to settle and stops the server with SIGTERM first."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import tfbench  # noqa: E402


def test_wait_disk_settled_waits_while_the_cache_grows(tmp_path):
    cache = tmp_path / ".yunshu" / "cache" / "apc"
    cache.mkdir(parents=True)
    f = cache / "blk"

    def writer():
        for _ in range(6):
            with f.open("ab") as h:
                h.write(b"x" * 1024)
            time.sleep(0.2)

    t = threading.Thread(target=writer)
    t.start()
    waited = tfbench.wait_disk_settled(tmp_path, quiet_s=0.5, max_s=10, poll_s=0.05)
    t.join()
    assert waited >= 1.0  # the writer ran ~1.2 s; the wait outlasted it
    assert f.stat().st_size == 6 * 1024


def test_wait_disk_settled_is_bounded(tmp_path):
    assert tfbench.wait_disk_settled(tmp_path, quiet_s=0.2, max_s=5, poll_s=0.05) < 1.0
    assert tfbench.wait_disk_settled(None) == 0.0


def test_graceful_kill_sends_sigterm_and_lets_the_handler_run(tmp_path):
    done = tmp_path / "flushed"
    code = (
        "import signal,sys,time,pathlib\n"
        f"p=pathlib.Path({str(done)!r})\n"
        "def h(*a):\n    p.write_text('ok'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM,h)\n"
        "print('ready',flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, start_new_session=True
    )
    assert proc.stdout.readline().strip() == b"ready"

    class S:
        pass

    s = S()
    s.proc, s.home, s.mem = proc, tmp_path, None
    tfbench.Srv.kill(s, keep_disk=True, graceful=True)
    assert (
        done.read_text() == "ok"
    )  # SIGTERM reached the handler: a SIGKILL would have left no file
    assert s.term_exit == "exited"
    assert proc.returncode == 0 and proc.returncode != -signal.SIGKILL
