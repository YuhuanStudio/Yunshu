"""m3_serve control loop: stop line, deadline and a dying server each end the wait with their own reason."""

import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts/research/agent_compat")
)
import m3_serve  # noqa: E402


def _listener():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(2)
    return s, s.getsockname()[1]


def test_stop_line():
    s, port = _listener()

    def client():
        time.sleep(0.2)
        c = socket.create_connection(("127.0.0.1", port))
        c.sendall(b"stop\n")
        assert c.recv(32) == b"stopping\n"

    t = threading.Thread(target=client)
    t.start()
    assert m3_serve.wait_stop(s, time.monotonic() + 10, lambda: True) == "stop"
    t.join()


def test_garbage_does_not_stop_and_deadline_wins():
    s, port = _listener()
    threading.Thread(
        target=lambda: socket.create_connection(("127.0.0.1", port)).sendall(b"hello\n")
    ).start()
    assert m3_serve.wait_stop(s, time.monotonic() + 1.5, lambda: True) == "deadline"


def test_server_death_is_reported():
    s, _ = _listener()
    assert m3_serve.wait_stop(s, time.monotonic() + 10, lambda: False) == "server-died"


def test_port_outside_range_refused(tmp_path):
    assert (
        m3_serve.main(
            [
                "--model",
                "x",
                "--port",
                "8000",
                "--control-port",
                "18995",
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 2
    )
