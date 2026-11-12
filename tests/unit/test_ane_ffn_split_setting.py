"""The ANE FFN split setting documents its hidden-size restriction (whole 2560-channel segments)."""

from yunshu_engine import settings
from yunshu_engine.ane import ffn_program


def test_ane_ffn_split_description_states_segment_restriction():
    text = settings.REGISTRY["YUNSHU_ANE_FFN_SPLIT"].description
    assert f"{ffn_program.SEGMENT}-channel" in text
