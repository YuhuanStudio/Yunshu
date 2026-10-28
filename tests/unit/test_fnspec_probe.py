"""CPU checks of the Flash-Next sampled-spec probe's verdict logic."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/research/fnspec_probe.py"
spec = importlib.util.spec_from_file_location("fnspec_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def row(text="a", lps=(0.1, 0.2), ct=3, rounds=2):
    return dict(
        _text=text,
        lp_tokens=["a", "b"],
        lps=list(lps),
        finish="length",
        ct=ct,
        pt=10,
        cached=0,
        ttft_s=1.0,
        dec_tps=20.0,
        xy={"speculative": {"rounds": rounds}},
    )


def test_digest_covers_text_and_logprobs():
    assert probe.digest(row()) == probe.digest(row())
    assert probe.digest(row()) != probe.digest(row(text="b"))
    assert probe.digest(row()) != probe.digest(row(lps=(0.1, 0.25)))


def test_compare_requires_identity_and_engagement():
    on = dict(cells={"c": probe.summarize(row())})
    off = dict(cells={"c": probe.summarize(row(rounds=0))})
    assert probe.compare(on, off)[0]
    assert not probe.compare(on, dict(cells={"c": probe.summarize(row(text="x"))}))[0]
    unengaged = dict(cells={"c": probe.summarize(row(rounds=0))})
    assert not probe.compare(unengaged, off)[0]
    assert not probe.compare(on, dict(cells={}))[0]


def test_incomplete_cell_fails_closed():
    with pytest.raises(RuntimeError):
        probe.check_row(dict(finish="stop", ct=2), 3, False)
    with pytest.raises(RuntimeError):
        probe.check_row(dict(finish="length", ct=3, lps=[0.1]), 3, True)
    probe.check_row(dict(finish="length", ct=3, lps=[0.1, 0.2]), 3, True)
