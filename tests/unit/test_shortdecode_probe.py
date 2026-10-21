"""CPU-only regression checks for short-context round attribution."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/research/shortdecode_probe.py"
spec = importlib.util.spec_from_file_location("shortdecode_probe", SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_rounds_use_emitted_tokens_including_bonus():
    summary = probe.summarize_rounds(
        "DEBUG DFlash round: at=1 kind=model rows=16 accepted=4 match=0 ms=45.50\n"
        "DEBUG DFlash round: at=5 kind=copy rows=32 accepted=21 match=8 ms=60.00\n"
        "DEBUG DFlash round: at=26 kind=model rows=8 accepted=3 match=-1 ms=42.50\n"
        "DFlash round profile: 3 rounds\n"
    )
    assert summary["round_count"] == 3
    assert summary["by_kind"]["model"] == dict(
        rounds=2,
        tokens=7,
        tokens_per_round=3.5,
        ms_per_round=44.0,
        total_ms=88.0,
        mean_rows=12.0,
    )
    assert summary["by_kind"]["copy"]["tokens_per_round"] == 21


def test_fail_closed_if_fast_tree_missing_or_response_incomplete():
    result = dict(invalid=[], completion_tokens=512)
    with pytest.raises(RuntimeError, match="fast tree not engaged"):
        probe.validate(result, probe.summarize_rounds("MTP decoder"), "yunshu")
    with pytest.raises(RuntimeError, match="invalid request"):
        probe.validate(dict(invalid=[], completion_tokens=10), {}, "kapitsa")


def test_never_persist_response_text():
    original = dict(text="code", reasoning="private", completion_tokens=512)
    result = probe.safe_result(original)
    assert "text" not in result and "reasoning" not in result
    assert len(result["text_sha256"]) == 64
    assert result["reasoning_chars"] == 7
    assert original["text"] == "code"


def test_cli_dry_run_and_source_are_explicit(tmp_path):
    args = probe.parse_args(
        ["--out", str(tmp_path), "--dry-run", "--source", str(tmp_path)]
    )
    assert args.dry_run and args.source == tmp_path
    assert args.runs == 5


def test_chain_trace_counts_bonus_and_observes_final_round(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    trace_spec = importlib.util.spec_from_file_location(
        "shortdecode_chain_trace", SOURCE.with_name("shortdecode_chain_trace.py")
    )
    trace = importlib.util.module_from_spec(trace_spec)
    trace_spec.loader.exec_module(trace)

    class Proxy:
        def __init__(self):
            self._seen_rounds = 0
            self._previous = None
            self._draft = SimpleNamespace(accept_lens=[])
            self._copy = SimpleNamespace(last_match=0)

        def observe(self):
            self._seen_rounds = len(self._draft.accept_lens)

        def choose(self, block, remaining):
            self.observe()
            self._previous = (False, block - 1)
            return block

    ticks = iter([1.0, 1.05])
    monkeypatch.setattr(trace.time, "perf_counter", lambda: next(ticks))
    trace.install(Proxy)
    proxy = Proxy()
    proxy.choose(8, 512)
    proxy._draft.accept_lens.append(3)
    with caplog.at_level(logging.DEBUG):
        proxy.observe()
        proxy.observe()  # no duplicate at close
    summary = probe.summarize_rounds(caplog.text)
    assert summary["round_count"] == 1
    assert summary["rounds"][0]["accepted"] == 4
    assert summary["rounds"][0]["ms"] == 50.0


def test_raw_receipt_uses_ids_and_plain_switch(caplog):
    import logging

    trace_spec = importlib.util.spec_from_file_location(
        "shortdecode_chain_receipt", SOURCE.with_name("shortdecode_chain_trace.py")
    )
    trace = importlib.util.module_from_spec(trace_spec)
    trace_spec.loader.exec_module(trace)
    calls = []

    class Runner:
        def iter_tokens(self, input_ids, **kwargs):
            calls.append(kwargs)
            yield from [42, 7, 999]

    trace.install_receipts(Runner, plain=True)
    with caplog.at_level(logging.DEBUG):
        assert list(Runner().iter_tokens([1, 2])) == [42, 7, 999]
    assert calls == [{"allow_draft": False}]
    assert "Raw token receipt: count=3 sha256=" in caplog.text
    assert "token_ids" not in caplog.text


def test_competitor_native_bonus_conventions():
    mlx = probe.competitor_rounds(
        "[spec-stats] mode=dflash attempts=76 accepts=436 avg_per_round=5.74",
        "mlxserve",
        512,
    )
    assert mlx["rounds"] == 76
    assert mlx["tokens_per_round"] == 512 / 76
    kapitsa = probe.competitor_rounds(
        "[kapitsa] chat: prompt 85 (cached 0) in 348 ms (+0.8 ms template/tokenize), 512 tokens in 3563 ms (143.4 tok/s), length, 5.32 tok/cycle",
        "kapitsa",
        512,
    )
    assert kapitsa["rounds"] == 96
    assert kapitsa["ms_per_round"] == 3563 / 96
