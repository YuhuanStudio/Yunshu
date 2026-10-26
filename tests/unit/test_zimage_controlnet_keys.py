"""Checkpoint-key remapping of the Z-Image ControlNet port (the checkpoint uses Sequential indices)."""

import pytest

pytest.importorskip("mlx.core")

from yunshu_engine.zimage_controlnet import _remap_controlnet_key  # noqa: E402


@pytest.mark.parametrize(
    "key,expected",
    [
        ("control_all_x_embedder.2-1.weight", "control_x_embedder.weight"),
        (
            "control_noise_refiner.0.before_proj.weight",
            "noise_refiner.0.before_proj.weight",
        ),
        ("control_layers.14.after_proj.bias", "layers.14.after_proj.bias"),
        (
            "control_layers.3.attention.to_out.0.weight",
            "layers.3.attention.to_out.weight",
        ),
        ("control_layers.3.adaLN_modulation.0.bias", "layers.3.adaLN_modulation.bias"),
        ("something.else", "something.else"),
    ],
)
def test_remap_controlnet_key(key, expected):
    assert _remap_controlnet_key(key) == expected


def test_remap_leaves_other_numeric_indices_alone():
    # only the `.0` of to_out / adaLN_modulation is a Sequential index
    assert (
        _remap_controlnet_key("control_layers.0.attention.to_q.weight")
        == "layers.0.attention.to_q.weight"
    )
    assert (
        _remap_controlnet_key("control_layers.0.feed_forward.w1.0.weight")
        == "layers.0.feed_forward.w1.0.weight"
    )
