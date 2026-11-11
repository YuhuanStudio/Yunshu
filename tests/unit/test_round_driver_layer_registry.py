"""Round-driver layer registry (M1 C1): ``select`` picks the right layer for the three CPU fixtures the driver
tests use (generic tiny, tiny Qwen3.5, tiny Flash-Next), and each ``supports`` agrees with the predicate the
driver and engine use today. Nothing in the driver calls the registry yet, so this is the only consumer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from yunshu_engine.round_driver import flashnext, forward  # noqa: E402
from yunshu_engine.round_driver.layer import (
    REGISTRY,
    FlashNextLayer,
    GenericLayer,
    PackedLayer,
)  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def generic_lm():
    from tests.unit.test_round_driver_generic import TinyLM

    return TinyLM()


@pytest.fixture(scope="module")
def packed_lm():
    # the tiny Qwen3.5 config only: no kernel install, so no process-global state leaks into other tests
    from mlx_vlm.models.qwen3_5.language import LanguageModel
    from scripts.research.phaseb_verify import tiny_config

    return LanguageModel(tiny_config())


@pytest.fixture(scope="module")
def flash_lm():
    from tests.unit.test_round_driver_flashnext import make_tiny

    lm, _head = make_tiny()
    return lm


def test_select_picks_the_layer_for_each_fixture(generic_lm, packed_lm, flash_lm):
    assert type(REGISTRY.select(generic_lm)) is GenericLayer
    assert type(REGISTRY.select(packed_lm)) is PackedLayer
    assert type(REGISTRY.select(flash_lm)) is FlashNextLayer


def test_select_unwraps_a_model_with_a_language_model(packed_lm):
    wrapper = SimpleNamespace(language_model=packed_lm)
    assert type(REGISTRY.select(wrapper)) is PackedLayer
    assert REGISTRY.select(wrapper).lm is packed_lm


def test_supports_agrees_with_the_existing_predicates(generic_lm, packed_lm, flash_lm):
    for lm in (generic_lm, packed_lm, flash_lm):
        assert PackedLayer.supports(lm) is forward.supports(lm)
        assert FlashNextLayer.supports(lm) is flashnext.is_flash_next(lm)
    # Flash-Next and packed claim disjoint models; generic is the fallback behind them in select order
    assert [cls.supports(flash_lm) for cls in REGISTRY.layers[:2]] == [True, False]
    assert [cls.supports(packed_lm) for cls in REGISTRY.layers[:2]] == [False, True]
    assert [cls.supports(generic_lm) for cls in REGISTRY.layers[:2]] == [False, False]


def test_flash_next_text_sub_config_is_flash_next():
    # the engine's Flash-Next checkpoint names its text config qwen4_exp_text (driver.py:466 same predicate)
    lm = SimpleNamespace(args=SimpleNamespace(model_type="qwen4_exp_text"))
    assert FlashNextLayer.supports(lm) is flashnext.is_flash_next(lm) is True
    assert PackedLayer.supports(lm) is False


def test_generic_is_the_fallback_for_any_model(generic_lm):
    assert GenericLayer.supports(generic_lm) is True
    assert GenericLayer.supports(SimpleNamespace()) is True


def test_adapters_return_the_drivers_cache_and_logits(generic_lm, packed_lm, flash_lm):
    assert isinstance(REGISTRY.select(generic_lm).new_cache(), list)
    assert len(REGISTRY.select(packed_lm).new_cache()) == len(packed_lm.model.layers)
    # generic: the rows already are logits, so the adapter is the identity (driver.py _logits)
    rows = mx.zeros((2, 8))
    assert REGISTRY.select(generic_lm).logits(rows) is rows
    assert REGISTRY.select(flash_lm).lm is flash_lm


def test_declared_caps_match_the_census():
    assert PackedLayer.caps.slot_decode and not PackedLayer.caps.identity_logits
    assert FlashNextLayer.caps.lazy_head and not FlashNextLayer.caps.slot_decode
    assert GenericLayer.caps.identity_logits and GenericLayer.caps.ngram
    assert not GenericLayer.caps.per_token_rows
