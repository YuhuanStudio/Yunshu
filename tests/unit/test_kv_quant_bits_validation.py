"""(MED): kv_cache_quant_bits from model_settings.json / env-model-override
bypassed the (2,3,4,8) validation that the YUNSHU_KV_QUANT_BITS env path enforces.

The env path (batched_engine.py ~886) rejects unsupported bit-widths with a warning,
but _apply_settings assigned s.kv_cache_quant_bits directly, so a settings file with
{"kv_cache_quant_bits": 5} set _kv_quant_bits=5 → to_quantized(bits=5) raises on every
request that hits the quant threshold (500). _apply_settings now validates first,
mirroring the env path.
"""

from __future__ import annotations

import types

from yunshu_engine.batched_engine import BatchedEngine


def _settings(bits):
    return types.SimpleNamespace(
        kv_cache_quant_bits=bits,
        kv_cache_quant_group_size=64,
        prefix_cache_enabled=True,  # True → no-op (only False clears the cache)
        spec_decode_enabled=False,
        spec_prefill_enabled=False,
        ssd_cache_enabled=False,
        enable_thinking=None,
        moe_top_k=0,
    )


def _engine():
    eng = BatchedEngine.__new__(BatchedEngine)
    eng._kv_quant_bits = None
    eng._kv_quant_group_size = 64
    eng._kv_prefix_cache = None
    eng._model = None
    eng.model_name = "m"
    eng.enable_thinking = False
    return eng


def test_invalid_bits_from_settings_rejected():
    eng = _engine()
    eng._settings = _settings(5)  # invalid — MLX supports only 2/3/4/8
    eng._apply_settings()
    assert eng._kv_quant_bits is None  # ignored, not assigned → no to_quantized crash


def test_valid_bits_from_settings_applied():
    for b in (2, 3, 4, 8):
        eng = _engine()
        eng._settings = _settings(b)
        eng._apply_settings()
        assert eng._kv_quant_bits == b


def test_none_bits_leaves_unchanged():
    eng = _engine()
    eng._kv_quant_bits = 4  # e.g. a validated env value already set
    eng._settings = _settings(None)
    eng._apply_settings()
    assert eng._kv_quant_bits == 4  # None settings doesn't clobber


def test_every_hardware_tier_is_lossless_kv():
    from yunshu_engine.utils.hardware import HardwareInfo, compute_adaptive_defaults

    for gb in (8.0, 16.0, 36.0, 64.0):
        hw = HardwareInfo(
            chip_name="Apple M1",
            total_memory_gb=gb,
            max_working_set_bytes=int(gb * 0.7 * 1024**3),
        )
        assert "kv_cache_quant_bits" not in compute_adaptive_defaults(hw), gb


def test_explicit_off_beats_model_settings(monkeypatch):
    monkeypatch.setenv("YUNSHU_KV_QUANT_BITS", "off")
    eng = _engine()
    eng._settings = _settings(4)
    eng._apply_settings()
    assert eng._kv_quant_bits is None


def test_model_settings_still_applies_when_env_unset(monkeypatch):
    monkeypatch.delenv("YUNSHU_KV_QUANT_BITS", raising=False)
    eng = _engine()
    eng._settings = _settings(4)
    eng._apply_settings()
    assert eng._kv_quant_bits == 4
