"""The no-backbone embedding fallback must refuse vocabulary logits."""

from __future__ import annotations

import mlx.core as mx
import pytest

from tests.unit.test_engine_embed_pool import _make_loaded_engine


def _engine(vocab):
    eng = _make_loaded_engine()
    eng._model.config = type("Cfg", (), {"hidden_size": 64, "vocab_size": vocab})()
    return eng


def test_vocab_sized_output_raises():
    with pytest.raises(RuntimeError, match="vocabulary"):
        _engine(100)._extract_hidden_states(mx.zeros((1, 3, 100)))


def test_hidden_sized_output_is_returned():
    out = mx.zeros((1, 3, 64))
    assert _engine(100)._extract_hidden_states(out) is out
