"""A lenient weight load may ignore extra checkpoint tensors but must never leave a model
parameter randomly initialised (the MTP target load and the TTS load used a blanket retry)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from yunshu_engine.checkpoint_keys import extras_only, lenient_extras_load
from yunshu_engine.mlxvlm_mtp import _tolerant_target_load


def _weights(lin, drop=None, extra=False):
    w = {"weight": mx.ones(lin.weight.shape), "bias": mx.ones(lin.bias.shape)}
    if drop:
        w.pop(drop)
    if extra:
        w["mtp.fc.weight"] = mx.zeros((2, 2))
    return list(w.items())


@pytest.mark.parametrize("ctx", [lenient_extras_load, _tolerant_target_load])
def test_extra_tensors_are_ignored(ctx):
    lin = nn.Linear(3, 2)
    with ctx():
        lin.load_weights(_weights(lin, extra=True))
    assert mx.all(lin.weight == 1).item()


@pytest.mark.parametrize("ctx", [lenient_extras_load, _tolerant_target_load])
def test_missing_parameter_still_fails(ctx):
    lin = nn.Linear(3, 2)
    with ctx(), pytest.raises(ValueError, match="does not cover 1 model parameters"):
        lin.load_weights(_weights(lin, drop="bias", extra=True))


def test_patch_is_restored():
    orig = nn.Module.load_weights
    with lenient_extras_load():
        assert nn.Module.load_weights is not orig
    assert nn.Module.load_weights is orig


def test_extras_only_names():
    extras_only(["a", "b", "x"], ["a", "b"])
    with pytest.raises(ValueError):
        extras_only(["a"], ["a", "b"])
