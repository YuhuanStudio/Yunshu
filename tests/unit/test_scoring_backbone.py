"""the scoring/pooling embedding FALLBACK resolved model.language_model as the
backbone, but for mlx-vlm models that wrapper applies lm_head and returns vocab-space
LOGITS, not hidden states — so it pooled logits silently → wrong-dimensioned, meaningless
embeddings (the W614 logits→backbone bug, never propagated to this fallback). Now it
descends to language_model.model (the real text backbone), mirroring the engine's
_get_backbone, and warns loudly if it ever still gets logits."""
from __future__ import annotations

import asyncio
import inspect

import mlx.core as mx

from yunshu_gateway.routers import scoring
from yunshu_gateway.routers.scoring import _fallback_embeddings

_HIDDEN_DIM = 8
_VOCAB_DIM = 1000


class _HiddenBackbone:
    def __call__(self, input_ids):
        return mx.ones([1, 3, _HIDDEN_DIM])  # [batch, seq, hidden]


class _LMOut:
    def __init__(self, logits):
        self.logits = logits


class _LangModelWrapper:
    """mlx-vlm-style: callable applies lm_head → returns LOGITS; .model is the real
    hidden-state backbone."""
    def __init__(self):
        self.model = _HiddenBackbone()

    def __call__(self, input_ids):
        return _LMOut(mx.ones([1, 3, _VOCAB_DIM]))  # vocab-space logits


class _VLMModel:
    def __init__(self):
        self.language_model = _LangModelWrapper()


class _StdModel:
    """Standard HF text model: model.model is the backbone."""
    def __init__(self):
        self.model = _HiddenBackbone()


class _Tok:
    def encode(self, text):
        return [1, 2, 3]


class _Engine:
    def __init__(self, model):
        self._tokenizer = _Tok()
        self._model = model


def test_vlm_fallback_uses_hidden_not_logits():
    out = asyncio.run(_fallback_embeddings(_Engine(_VLMModel()), ["hi"], "MEAN", False))
    # the pooled vector must be the HIDDEN dim (8), NOT the vocab/logits dim (1000)
    assert len(out[0]) == _HIDDEN_DIM, f"pooled logits ({len(out[0])}) instead of hidden state"


def test_standard_model_backbone_still_resolves():
    out = asyncio.run(_fallback_embeddings(_Engine(_StdModel()), ["hi"], "MEAN", False))
    assert len(out[0]) == _HIDDEN_DIM


def test_logits_branch_warns_not_silent():
    src = inspect.getsource(scoring)
    # descends to language_model.model
    assert "_lm.model" not in src or "getattr(_lm" in src  # resolution present
    assert 'getattr(model, "language_model"' in src
    # the logits fallback warns loudly now (was silent)
    i = src.index("hasattr(out, 'logits')")
    window = src[i:i + 400]
    assert "logger.warning" in window
