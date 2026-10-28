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


@pytest.mark.serial  # other tests allocate between the reads when run in parallel
def test_available_bytes_matches_vm_stat():
    def slow():
        return memory_admission.parse_vm_stat(
            subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=True
            ).stdout
        )

    # the same formula read a moment apart: the syscall value must fall between two vm_stat reads
    # taken around it (plus slack for pages that move while vm_stat itself runs)
    before = slow()
    fast = memory_admission.available_bytes()
    after = slow()
    assert fast is not None and before is not None and after is not None
    slack = 2**30
    assert min(before, after) - slack <= fast <= max(before, after) + slack


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
