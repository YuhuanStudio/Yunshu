"""Per-request host memory reads use syscalls, never a fork+exec of the serving process.

A subprocess (`vm_stat`, `sysctl`) per request cost 8-17 ms of warm-request admission on 27B (lead tailprof
2026-10-28: the MLX executor sat in subprocess._execute_child from memory_admission.read_host while a
back-to-back request waited)."""

import subprocess
import sys

import pytest

from yunshu_engine import memory_admission

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="mach host_statistics64"
)


def test_available_bytes_matches_vm_stat():
    fast = memory_admission.available_bytes()
    slow = memory_admission.parse_vm_stat(
        subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    )
    assert fast is not None and slow is not None
    # the two reads are a moment apart; page counts drift a little between them
    assert abs(fast - slow) <= max(256 * 2**20, 0.02 * slow)


def test_read_host_does_not_spawn_a_process(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("read_host spawned a process")

    monkeypatch.setattr(memory_admission.subprocess, "run", boom)
    host = memory_admission.read_host()
    assert host is not None and host.available > 0 and host.total > host.available


def test_request_memory_check_does_not_spawn_a_process(monkeypatch):
    from yunshu_gateway.middleware import request_logging

    calls = []
    # the check swallows exceptions, so count the calls instead of raising
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    request_logging._check_memory_pressure()
    request_logging._check_memory_pressure()
    assert calls == []
    assert request_logging._total_memory() > 0
