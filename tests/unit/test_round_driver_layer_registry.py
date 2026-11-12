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


def test_driver_builds_the_layer_its_flags_name(generic_lm, packed_lm, flash_lm):
    # M1 C2: the old flags build the same layer, and the layer sets the driver's head, batch and draft width
    from yunshu_engine.round_driver import driver as drv
    from yunshu_engine.round_driver.generic import GenericBatch

    generic_driver = drv.RoundDriver(generic_lm, generic_layer=True)
    assert type(generic_driver.layer) is GenericLayer
    assert isinstance(generic_driver.batch, GenericBatch)
    assert generic_driver.generic and not generic_driver.flash
    assert generic_driver.max_draft == 0 and generic_driver.head is None

    packed_driver = drv.RoundDriver(packed_lm)
    assert type(packed_driver.layer) is PackedLayer
    assert not packed_driver.flash and not packed_driver.generic
    assert packed_driver.max_draft == 7

    flash_driver = drv.RoundDriver(SimpleNamespace(language_model=flash_lm))
    assert type(flash_driver.layer) is FlashNextLayer
    assert flash_driver.flash and flash_driver.max_draft == 1

    explicit = drv.RoundDriver(generic_lm, layer=GenericLayer(generic_lm))
    assert type(explicit.layer) is GenericLayer and isinstance(
        explicit.batch, GenericBatch
    )


def test_layer_arguments_are_checked(generic_lm, packed_lm):
    from yunshu_engine.round_driver import driver as drv

    with pytest.raises(ValueError, match="not both"):
        drv.RoundDriver(generic_lm, generic_layer=True, layer=GenericLayer(generic_lm))
    with pytest.raises(ValueError, match="another language model"):
        drv.RoundDriver(packed_lm, layer=PackedLayer(generic_lm))


def test_drafters_contract_per_layer(generic_lm, packed_lm):
    plain = dict(
        drafter=None, dflash_drafter=None, plain_verify=False, kv_format="bf16"
    )
    assert GenericLayer(generic_lm).drafters(generic_lm, **plain).max_draft == 0
    with pytest.raises(ValueError, match="generic layer"):
        GenericLayer(generic_lm).drafters(generic_lm, **{**plain, "drafter": object()})
    packed = PackedLayer(packed_lm).drafters(packed_lm, **plain)
    assert packed.head is None and packed.dflash is None and packed.max_draft == 7


def test_prefill_dispatches_to_each_family_forward(
    generic_lm, packed_lm, flash_lm, monkeypatch
):
    """C5: each adapter's ``prefill`` calls its family's own forward (the driver's old branches)."""
    from yunshu_engine.round_driver import generic

    calls: list[str] = []
    monkeypatch.setattr(
        forward, "lane_forward", lambda lm, seg, **kw: calls.append("lane") or "h"
    )
    monkeypatch.setattr(
        flashnext, "prefill", lambda lm, seg, **kw: calls.append("flash") or "h"
    )
    monkeypatch.setattr(
        generic, "prefill", lambda lm, row, s, e: calls.append("generic") or None
    )
    seg = SimpleNamespace()
    row = SimpleNamespace()
    assert PackedLayer(packed_lm).prefill(row, seg, 0, 4) == "h"
    assert FlashNextLayer(flash_lm).prefill(row, seg, 0, 4) == "h"
    assert GenericLayer(generic_lm).prefill(row, seg, 0, 4) is None
    assert calls == ["lane", "flash", "generic"]


def test_c6_caps_and_hooks_match_the_census(
    generic_lm, packed_lm, flash_lm, monkeypatch
):
    """C6: replay / lone pipelining / lazy head / fn_trace caps, and the APC suffix and step hooks per layer."""
    monkeypatch.delenv("YUNSHU_JUMP_FORWARD", raising=False)
    assert PackedLayer.caps.replay and FlashNextLayer.caps.replay
    assert not GenericLayer.caps.replay
    assert GenericLayer.caps.lone_pipelining and not PackedLayer.caps.lone_pipelining
    assert not PackedLayer.caps.lazy_head and FlashNextLayer.caps.lazy_head
    assert FlashNextLayer.caps.fn_trace and not PackedLayer.caps.fn_trace
    assert not GenericLayer.caps.fn_trace
    assert PackedLayer(packed_lm).apc_suffix() == ""
    assert FlashNextLayer(flash_lm).apc_suffix() == ""
    assert GenericLayer(generic_lm).apc_suffix() == ":resume-rope-v2"
    monkeypatch.setenv("YUNSHU_JUMP_FORWARD", "1")
    assert GenericLayer(generic_lm).apc_suffix() == ":resume-rope-v2:jump-forward"


def test_c7_release_cut_out_is_the_decode_set_method():
    from yunshu_engine.round_driver.batch import DecodeBatch
    from yunshu_engine.round_driver.generic import GenericBatch

    # M1 C7: the release path calls cut_out on the decode set; only packed layers take the slot cut-out
    assert callable(DecodeBatch.cut_out) and callable(GenericBatch.cut_out)
    assert not hasattr(DecodeBatch, "restore") and not hasattr(GenericBatch, "restore")
    assert PackedLayer.caps.slot_decode and not GenericLayer.caps.slot_decode
    assert not FlashNextLayer.caps.slot_decode
