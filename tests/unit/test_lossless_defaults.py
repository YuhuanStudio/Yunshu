"""Lossy optimizations are user options with a lossless default.

Rule (CLAUDE.md, Settings): anything that changes output quality (KV / state
quantization, weight quantization) is opt-in; only lossless optimizations may be
on by default. This pins the defaults of every lossy knob in the registry.
"""

from __future__ import annotations

import pytest

from yunshu_engine import settings

LOSSLESS_DEFAULTS = {
    "YUNSHU_KV_QUANT_BITS": "off",  # text engine KV quantization
    "YUNSHU_PREFIX_HOT_LIMIT": 0,  # text prefix cache 4-bit WARM tier
    "YUNSHU_SSD_CACHE_PRECISION": "native",  # text SSD prefix-cache storage
    "YUNSHU_KV_PRECISION": "bf16",  # VLM runner shared-batch KV
    "YUNSHU_QUANT_MODE": "",  # in-memory weight quantization at load
    "YUNSHU_QUANT_CONFIG": None,
}


@pytest.mark.parametrize(("name", "lossless"), sorted(LOSSLESS_DEFAULTS.items()))
def test_lossy_setting_defaults_lossless(monkeypatch, name, lossless):
    monkeypatch.delenv(name, raising=False)
    assert settings.get(name) == lossless


@pytest.mark.parametrize(
    "name",
    [
        "YUNSHU_KV_QUANT_BITS",
        "YUNSHU_PREFIX_HOT_LIMIT",
        "YUNSHU_SSD_CACHE_PRECISION",
        "YUNSHU_QUANT_MODE",
    ],
)
def test_lossy_setting_says_so(name):
    assert "lossy" in settings.REGISTRY[name].description
