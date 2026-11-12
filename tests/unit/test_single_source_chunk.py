"""HARN 436 (A-10): the round-driver prefill atom has one definition, round_driver/driver.py CHUNK."""

from yunshu_engine import vlm_engine
from yunshu_engine.round_driver import driver


def test_vlm_engine_has_no_second_chunk_constant():
    assert not hasattr(vlm_engine, "ROUND_PREFILL_CHUNK")


def test_driver_chunk_is_the_prefill_atom():
    assert vlm_engine._round_prefill_chunk() == driver.CHUNK == 512
