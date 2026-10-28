"""_wired_limit_ctx sizes the model from its flattened parameters.

``sum(p.nbytes for p in model.parameters())`` iterated the nested dict's keys, raised,
and the DEBUG-logged exception meant the wired limit was never raised for large models.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from yunshu_engine import batched_engine


def test_large_model_raises_and_restores_the_wired_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mx.metal, "recommended_max_working_memory_size", lambda: 100, raising=False
    )
    monkeypatch.setattr(mx, "set_wired_limit", lambda v: calls.append(v) or 7)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))  # 2 * (64 + 8) * 4 B > 50
    with batched_engine._wired_limit_ctx(model):
        assert calls == [100]
    assert calls == [100, 7]


def test_small_model_leaves_the_limit_alone(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mx.metal, "recommended_max_working_memory_size", lambda: 10**12, raising=False
    )
    monkeypatch.setattr(mx, "set_wired_limit", lambda v: calls.append(v) or 7)
    with batched_engine._wired_limit_ctx(nn.Linear(2, 2)):
        pass
    assert calls == []
