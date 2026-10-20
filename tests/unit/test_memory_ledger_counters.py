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


def test_draft_window_store_is_attributed_once(monkeypatch):
    from yunshu_engine import dflash_apc_window

    monkeypatch.setattr(dflash_apc_window, "resident_bytes", lambda: 123456)
    monkeypatch.setattr(ledger, "_loaded_models", lambda *args: [])
    monkeypatch.setattr(
        ledger, "mlx_counters", lambda: {"active": 123456, "cache": 0, "peak": 123456}
    )
    sample = ledger.collect(None, None)
    windows = [o for o in sample["owners"] if o["kind"] == "apc_draft_window"]
    assert len(windows) == 1
    assert windows[0]["bytes"] == 123456
    assert windows[0]["estimated"] is False
    assert next(o for o in sample["owners"] if o["kind"] == "other")["bytes"] == 0


def test_grammar_token_tables_are_a_named_native_owner(monkeypatch):
    import types

    from yunshu_engine import tool_call_grammar as tcg

    tokenizer = types.SimpleNamespace(backend_tokenizer=object())
    held = []
    fake = types.ModuleType("llguidance.hf")

    def build(hf, n_vocab):
        held.append(bytearray(96 * 2**20))
        return object()

    fake.from_tokenizer = build
    parent = types.ModuleType("llguidance")
    parent.hf = fake
    monkeypatch.setitem(sys.modules, "llguidance", parent)
    monkeypatch.setitem(sys.modules, "llguidance.hf", fake)
    monkeypatch.setattr(tcg, "_TOKENIZERS", {})
    monkeypatch.setattr(tcg, "_TOKENIZER_BYTES", {})
    assert tcg.resident_tokenizer_bytes() == 0
    tcg.llg_tokenizer(tokenizer, 1000)
    tcg.llg_tokenizer(tokenizer, 1000)  # cached: counted once
    assert len(held) == 1
    if tcg._heap_pages_bytes():  # macOS malloc zone readable
        assert tcg.resident_tokenizer_bytes() >= 48 * 2**20

    monkeypatch.setattr(tcg, "resident_tokenizer_bytes", lambda: 4242)
    monkeypatch.setattr(ledger, "_loaded_models", lambda *args: [])
    monkeypatch.setattr(
        ledger, "mlx_counters", lambda: {"active": 10, "cache": 0, "peak": 10}
    )
    sample = ledger.collect(None, None)
    owner = [o for o in sample["native_owners"] if o["kind"] == "grammar_tokenizer"]
    assert len(owner) == 1 and owner[0]["bytes"] == 4242
    assert all(o["kind"] != "grammar_tokenizer" for o in sample["owners"])
    assert next(o for o in sample["owners"] if o["kind"] == "other")["bytes"] == 10
