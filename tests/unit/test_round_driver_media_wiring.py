"""Runner -> round driver wiring of media prompts: embeddings, mRoPE positions, decode offset."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from yunshu_engine.vlm_batch_runner import VLMBatchRunner  # noqa: E402


def runner():
    return VLMBatchRunner(object(), object(), apc_manager=None)


def test_text_prompt_has_no_media_fields_and_every_prompt_is_taken_by_the_driver():
    r = runner()
    assert r._media_request(SimpleNamespace(prompt_kwargs=None)) == {}
    r.driver = object()
    # media prompts are no longer excluded from the driver
    assert r._driver_takes(SimpleNamespace(prompt_kwargs={"inputs_embeds": 1}))
    assert r._driver_takes(SimpleNamespace(prompt_kwargs=None))


def test_media_prompt_maps_to_request_fields():
    embeds = mx.zeros((1, 6, 4))
    pos = mx.arange(18, dtype=mx.int32).reshape(3, 1, 6)
    job = SimpleNamespace(
        prompt_kwargs={
            "inputs_embeds": embeds,
            "position_ids": pos,
            "rope_deltas": mx.array([[-3.0]]),
            "image_grid_thw": mx.array([[1, 2, 2]]),
        }
    )
    got = runner()._media_request(job)
    assert got["embeds"].shape == (6, 4)
    assert got["position_ids"].shape == (3, 6)
    assert bool(mx.array_equal(got["position_ids"], pos[:, 0, :]))
    assert got["rope_delta"] == -3


def test_unsupported_prompt_features_are_refused_not_dropped():
    job = SimpleNamespace(
        prompt_kwargs={
            "inputs_embeds": mx.zeros((1, 2, 4)),
            "deepstack_visual_embeds": [mx.zeros((1, 2, 4))],
        }
    )
    with pytest.raises(ValueError, match="deepstack_visual_embeds"):
        runner()._media_request(job)
