"""Flash-Next shared_expert_gate: a row's bits must not depend on how many rows share the prefill segment.

The same body runs on base and on the fix: it calls the gate through the layer object, and the only
difference is what the engine installed there (stock QuantizedLinear on base, RowwiseGate after the fix)."""

import importlib

import mlx.core as mx
import mlx.nn as nn


def _installed_gate_layer():
    """A layer whose gate is whatever the engine installs; base has no installer, so it stays stock."""
    mx.random.seed(0)
    lin = nn.QuantizedLinear(2560, 1, bias=False, group_size=64, bits=8)

    class Mlp:
        pass

    class Layer:
        pass

    class Model:
        pass

    class LM:
        pass

    layer = Layer()
    layer.mlp = Mlp()
    layer.mlp.shared_expert_gate = lin
    lm = LM()
    lm.model = Model()
    lm.model.layers = [layer]
    try:
        sgate = importlib.import_module("yunshu_engine.kernels.sgate_rowwise")
    except (
        ImportError
    ):  # base tree: the engine installs nothing, the stock gate is what prefill calls
        sgate = None
    if sgate is not None:
        sgate.install(lm)
    return lm.model.layers[0]


def test_installed_gate_rows_are_segment_length_invariant():
    layer = _installed_gate_layer()
    mx.random.seed(1)
    x = (mx.random.normal((512, 2560)) * 0.1).astype(mx.bfloat16)
    # prefill calls the gate per segment: mlp(x[:, s:e]) with M = segment length
    short = layer.mlp.shared_expert_gate(x[None, :5])[0]
    full = layer.mlp.shared_expert_gate(x[None])[0][:5]
    mx.eval(short, full)
    assert short.dtype == full.dtype
    assert mx.array_equal(short, full).item(), "gate rows differ between M=5 and M=512"
