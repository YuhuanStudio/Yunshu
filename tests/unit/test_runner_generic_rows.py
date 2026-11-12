"""Regression (M1 C8): the runner builds generic-layer rows with the model's prompt inputs.

``GenericLayer.admit`` requires ``prompt_kwargs``; the runner must pick the generic request from the layer
(``isinstance(getattr(driver, "layer", None), GenericLayer)``; test fakes may lack a layer), not from a driver attribute that M1 removed."""

import inspect


def test_runner_reads_the_layer_not_a_driver_flag():
    from yunshu_engine import vlm_batch_runner

    src = inspect.getsource(vlm_batch_runner)
    assert 'getattr(driver, "generic"' not in src
    assert "isinstance(getattr(driver, \"layer\", None), GenericLayer)" in src


def test_generic_layer_rows_get_prompt_inputs_on_admit():
    import pytest

    from yunshu_engine.round_driver.layer import GenericLayer

    layer = GenericLayer.__new__(GenericLayer)
    row = type("Row", (), {})()
    req = type("Req", (), {"prompt_kwargs": None, "ids": [1, 2]})()
    with pytest.raises(ValueError, match="prompt_kwargs are required"):
        layer.admit(row, req)
