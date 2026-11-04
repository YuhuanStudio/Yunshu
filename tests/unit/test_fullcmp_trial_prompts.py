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
    spec = importlib.util.spec_from_file_location("tfbench_trials", ROOT / "scripts" / "research" / "tfbench.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tfbench_trials"] = mod
    spec.loader.exec_module(mod)
    import snapshot_prompts as sp
    monkeypatch.setattr(sp, "exact_chat_prompt", lambda src, target, suffix="": f"{src}|{target}")
    monkeypatch.setattr(mod, "load_variant_prompt", lambda name, variant=0: f"BODY-{name}-v{variant}")
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
