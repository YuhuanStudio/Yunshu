"""Port-pool exhaustion must wait, without releasing a successful reservation."""

from types import SimpleNamespace

import pytest

from . import bound_listener as bl


def fake_pool(monkeypatch, busy):
    sockets, sleeps = [], []
    clock = [0.0]

    class Listener:
        def __init__(self):
            self.index = len(sockets)
            self.closed = False
            self.calls = []
            sockets.append(self)

        def setsockopt(self, *args):
            self.calls.append(("reuse", args))

        def bind(self, address):
            self.calls.append(("bind", address))
            if self.index < busy:
                raise OSError("busy")

        def listen(self, backlog):
            self.calls.append(("listen", backlog))

        def close(self):
            self.closed = True

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(bl.socket, "socket", lambda *args: Listener())
    monkeypatch.setattr(
        bl,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep),
        raising=False,
    )
    return sockets, sleeps


def test_reserved_listener_reuses_closed_ports_before_binding(monkeypatch):
    sockets, sleeps = fake_pool(monkeypatch, 0)
    result = bl.reserve_listener()
    assert result is sockets[0] and not result.closed
    assert result.calls[0] == (
        "reuse",
        (bl.socket.SOL_SOCKET, bl.socket.SO_REUSEADDR, 1),
    )
    assert [call[0] for call in result.calls] == ["reuse", "bind", "listen"]
    assert sleeps == []


def test_busy_pool_retries_and_keeps_only_owned_listener(monkeypatch):
    sockets, sleeps = fake_pool(monkeypatch, 10)
    result = bl.reserve_listener(timeout=2, poll_interval=0.25)
    assert result is sockets[-1] and len(sockets) == 11
    assert all(s.closed for s in sockets[:-1]) and not result.closed
    assert sleeps == [0.25]


def test_busy_pool_timeout_closes_every_attempt(monkeypatch):
    sockets, sleeps = fake_pool(monkeypatch, 100)
    with pytest.raises(RuntimeError, match="no free port"):
        bl.reserve_listener(timeout=0)
    assert len(sockets) == 10 and all(s.closed for s in sockets)
    assert sleeps == []
