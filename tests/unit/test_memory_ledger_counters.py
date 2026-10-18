"""An impossible OS counter cannot be used as deployment headroom."""

import sys
from types import SimpleNamespace

import pytest

from yunshu_gateway import memory_ledger as ledger


@pytest.mark.parametrize("available", [-1, 101])
def test_available_outside_physical_ram_is_unknown(monkeypatch, available):
    monkeypatch.setattr(
        ledger, "_sysctl_int", lambda name: 100 if name == "hw.memsize" else None
    )
    monkeypatch.setattr(ledger, "_sysctl_raw", lambda *a: None)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=available)),
    )
    sample = ledger._host_uncached()
    assert sample["available_bytes"] is None
    assert sample["available_reported_bytes"] == available


def test_valid_available_counter_is_preserved(monkeypatch):
    monkeypatch.setattr(
        ledger, "_sysctl_int", lambda name: 100 if name == "hw.memsize" else None
    )
    monkeypatch.setattr(ledger, "_sysctl_raw", lambda *a: None)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=40)),
    )
    assert ledger._host_uncached()["available_bytes"] == 40


def test_drafter_weights_exclude_target_and_internal_aliases():
    target_array = SimpleNamespace(nbytes=100)
    draft_array = SimpleNamespace(nbytes=40)
    target = SimpleNamespace(parameters=lambda: {"weight": target_array})
    draft = SimpleNamespace(
        parameters=lambda: {
            "tied": target_array,
            "weight": draft_array,
            "alias": draft_array,
        }
    )
    engine = SimpleNamespace(
        _model=target, _batch_runner=SimpleNamespace(drafter=draft)
    )
    assert ledger.measured_drafter_bytes(engine) == 40
    assert ledger.measured_drafter_bytes(SimpleNamespace(_model=target)) is None
