import pytest

from yunshu_engine.checkpoint_keys import key_rename

MODEL = {
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.0.mlp.up.weight",
}


def test_unprefixed_backbone_checkpoint_is_renamed():
    ck = {"embed_tokens.weight", "norm.weight", "layers.0.mlp.up.weight"}
    ren = key_rename(ck, MODEL)
    assert {ren.get(k, k) for k in ck} == MODEL


def test_extras_alone_need_no_rename():
    assert key_rename(MODEL | {"model.layers.5.vestigial.k"}, MODEL) == {}


def test_missing_parameters_fail_closed():
    with pytest.raises(ValueError, match="does not cover"):
        key_rename({"something.else"}, MODEL)
