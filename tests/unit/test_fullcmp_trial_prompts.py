"""Every conc/mixed trial's long prompt differs from token 0 of the previous trial (no prefix-cache hit
across trials on one server), and trial k is the same text for every engine. CPU only."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tb(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "tfbench_trials", ROOT / "scripts" / "research" / "tfbench.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tfbench_trials"] = mod
    spec.loader.exec_module(mod)
    import snapshot_prompts as sp

    monkeypatch.setattr(
        sp, "exact_chat_prompt", lambda src, target, suffix="": f"{src}|{target}"
    )
    monkeypatch.setattr(
        mod, "load_variant_prompt", lambda name, variant=0: f"BODY-{name}-v{variant}"
    )
    return mod


def test_each_trial_starts_differently_from_token_zero(tb):
    t0 = tb.mixed_long_text(0, 8192)
    t1 = tb.mixed_long_text(1, 8192)
    assert t0 != t1
    assert t0[:24] != t1[:24]  # the prefix differs at the front, not only the suffix


def test_trial_text_is_the_same_for_every_engine(tb):
    # the function takes no engine: trial k is one text for all engines
    assert tb.mixed_long_text(2, 131072, 0) == tb.mixed_long_text(2, 131072, 0)


def test_the_long_request_is_sized_to_the_cell_context(tb):
    assert tb.mixed_long_text(0, 131072).endswith("|131072")


def test_smoke_keeps_the_short_hello(tb):
    assert tb.mixed_long_text(0, 512, smoke=True) == "Say hello."


# --- the real tokenizer: conc prompts are salted from token 0 and sized to the cell context ----

CTX = 8192


@pytest.fixture
def real_tb(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "tfbench_real", ROOT / "scripts" / "research" / "tfbench.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tfbench_real"] = mod
    spec.loader.exec_module(mod)
    import snapshot_prompts as sp

    try:
        sp.chat_tokenizer()
    except Exception as exc:  # the tokenizer files are local; a missing model dir is a skip, not a pass
        pytest.skip(f"no local tokenizer: {exc}")
    monkeypatch.setattr(
        mod,
        "load_variant_prompt",
        lambda name, variant=0: "lorem ipsum dolor sit amet " * 4000,
    )
    return mod


def _first_tokens(text, n=8):
    import snapshot_prompts as sp

    return sp.chat_tokenizer().encode(text)[:n]


def test_conc_distinct_prompts_are_salted_from_token_zero_and_sized_to_the_context(
    real_tb,
):
    import snapshot_prompts as sp

    for trial in (0, 1):
        texts = [real_tb.concurrent_prompt(i, trial, CTX) for i in range(3)]
        for text in texts:
            assert (
                sp.chat_token_count(text) == CTX
            )  # the real tokenizer, rendered with the chat template
    a, b = real_tb.concurrent_prompt(0, 0, CTX), real_tb.concurrent_prompt(0, 1, CTX)
    assert _first_tokens(a) != _first_tokens(
        b
    )  # the salt is at token 0, not only in the suffix


def test_conc_same_prompt_is_salted_per_trial_and_sized_to_the_context(real_tb):
    import snapshot_prompts as sp

    shared = [real_tb.concurrent_prompt(0, trial, CTX) for trial in (0, 1, 2)]
    for text in shared:
        assert sp.chat_token_count(text) == CTX
    # trial k must not hit trial k-1's cache: the prefix differs from token 0 in every trial
    assert (
        _first_tokens(shared[0]) != _first_tokens(shared[1]) != _first_tokens(shared[2])
    )


def test_sweeps_with_different_concurrency_never_share_a_prompt(tb):
    # n=2 trial 0 and n=4 trial 0 must differ from token 0 (one server, two sweeps: a cache hit otherwise)
    a = tb.concurrent_prompt(0, 0, 8192, run=2)
    b = tb.concurrent_prompt(0, 0, 8192, run=4)
    assert a[:24] != b[:24]
    assert tb.concurrent_prompt(0, 0, 8192, run=2) == a  # the same run is reproducible
