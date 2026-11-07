"""TFB_STUB_SERVER (test-only): tfbench launches the stub in place of an engine. CPU only."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import tfbench  # noqa: E402


def test_stub_command_names_the_engine_port_and_server_log(tmp_path):
    cmd = tfbench.stub_server_cmd("mlx-serve-26.11.1", 18999, tmp_path / "s.log", HERE)
    assert cmd[0] == sys.executable
    assert cmd[1] == str(HERE / "fullcmp_stub_server.py")
    assert cmd[cmd.index("--engine") + 1] == "mlx-serve-26.11.1"
    assert cmd[cmd.index("--port") + 1] == "18999"
    assert cmd[cmd.index("--log") + 1] == str(tmp_path / "s.log")


def test_stub_process_serves_on_the_requested_port_and_logs_engine_lines(tmp_path):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = tmp_path / "server.log"
    cmd = tfbench.stub_server_cmd("omlx-0.7.3", port, log, HERE)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while True:
            try:
                data = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=2
                ).read()
                break
            except Exception:
                if time.time() - t0 > 20:
                    raise
                time.sleep(0.2)
        assert b"stub-model" in data
        assert "DFlash enabled" in log.read_text()
    finally:
        proc.kill()
        proc.wait(timeout=10)
