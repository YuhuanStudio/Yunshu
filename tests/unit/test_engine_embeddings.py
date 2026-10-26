"""Behaviour of engine_embeddings: pooling-mode detection, backbone / hidden-size resolution and pooling."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402

from yunshu_engine.engine_embeddings import EngineEmbeddingsMixin as E  # noqa: E402


def _pooling_dir(tmp_path, **flags):
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling" / "config.json").write_text(json.dumps(flags))
    return str(tmp_path)


@pytest.mark.parametrize(
    "flags,expected",
    [
        ({"pooling_mode_cls_token": True}, "CLS"),
        ({"pooling_mode_lasttoken": True}, "LAST"),
        ({"pooling_mode_mean_tokens": True}, "MEAN"),
        ({}, "MEAN"),
    ],
)
def test_resolve_embedding_pooling_from_sentence_transformers_config(
    tmp_path, flags, expected
):
    eng = SimpleNamespace(model_name=_pooling_dir(tmp_path, **flags))
    assert E._resolve_embedding_pooling(eng) == expected
    assert eng._embed_pooling == expected  # cached on the engine


def test_resolve_embedding_pooling_defaults_to_mean_without_config(tmp_path):
    assert (
        E._resolve_embedding_pooling(SimpleNamespace(model_name=str(tmp_path)))
        == "MEAN"
    )


def test_resolve_embedding_pooling_uses_the_cached_value(tmp_path):
    eng = SimpleNamespace(
        model_name=_pooling_dir(tmp_path, pooling_mode_cls_token=True),
        _embed_pooling="LAST",
    )
    assert E._resolve_embedding_pooling(eng) == "LAST"


def test_get_hidden_size_precedence():
    cfg = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=512))
    )
    assert E._get_hidden_size(SimpleNamespace(_model=cfg.model)) == 512
    nested = SimpleNamespace(
        args=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=640), hidden_size=1
        )
    )
    assert E._get_hidden_size(SimpleNamespace(_model=nested)) == 640
    assert E._get_hidden_size(SimpleNamespace(_model=SimpleNamespace())) == 768


def test_get_backbone_prefers_language_model_then_model_then_transformer():
    def inner():
        return None

    other = lambda x: x  # noqa: E731
    lm = SimpleNamespace(language_model=SimpleNamespace(model=inner), model=other)
    assert E._get_backbone(SimpleNamespace(_model=lm)) is inner
    assert (
        E._get_backbone(SimpleNamespace(_model=SimpleNamespace(model=other))) is other
    )
    assert (
        E._get_backbone(SimpleNamespace(_model=SimpleNamespace(transformer=other)))
        is other
    )
    assert E._get_backbone(SimpleNamespace(_model=SimpleNamespace())) is None


def test_extract_hidden_states_raises_on_vocab_space_logits():
    eng = SimpleNamespace(
        _model=SimpleNamespace(config=SimpleNamespace(vocab_size=11)),
        model_name="m",
    )
    with pytest.raises(RuntimeError, match="vocabulary dimension"):
        E._extract_hidden_states(eng, mx.zeros((1, 3, 11)))


def test_extract_hidden_states_accepts_array_tuple_and_last_hidden_state():
    eng = SimpleNamespace(
        _model=SimpleNamespace(config=SimpleNamespace(vocab_size=100, hidden_size=4)),
        model_name="m",
        _get_hidden_size=lambda: 4,
    )
    h = mx.ones((1, 3, 4))
    assert E._extract_hidden_states(eng, h).shape == (1, 3, 4)
    assert E._extract_hidden_states(eng, (h, None)).shape == (1, 3, 4)
    obj = SimpleNamespace(last_hidden_state=h)
    assert E._extract_hidden_states(eng, obj) is h


class _Tok:
    def encode(self, text):
        return [ord(c) % 7 for c in text]


def _pool_engine(backbone):
    return SimpleNamespace(
        _loaded=True,
        _model=object(),
        _tokenizer=_Tok(),
        _get_backbone=lambda: backbone,
        _get_hidden_size=lambda: 3,
    )


def test_pool_modes_and_empty_text():
    # hidden[t] = [t, 10 + t, 20 + t] for token position t
    def backbone(ids):
        n = ids.shape[1]
        base = mx.arange(n).reshape(1, n, 1) + mx.array([0.0, 10.0, 20.0]).reshape(
            1, 1, 3
        )
        return base

    eng = _pool_engine(backbone)
    assert E.pool(eng, ["abcd"], "CLS") == [[0.0, 10.0, 20.0]]
    assert E.pool(eng, ["abcd"], "last") == [[3.0, 13.0, 23.0]]
    assert E.pool(eng, ["abcd"], "MEAN") == [[1.5, 11.5, 21.5]]
    assert E.pool(eng, [""], "MEAN") == [
        [0.0, 0.0, 0.0]
    ]  # empty text -> zero vector of hidden size


def test_pool_requires_a_loaded_engine():
    eng = SimpleNamespace(_loaded=False, _model=None, _tokenizer=None)
    with pytest.raises(RuntimeError, match="not loaded"):
        E.pool(eng, ["x"])
