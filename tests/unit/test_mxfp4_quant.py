"""on-the-fly weight quantization at load (YUNSHU_QUANT_MODE)."""

from __future__ import annotations

import mlx.nn as nn

from yunshu_engine.batched_engine import BatchedEngine


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64, bias=False)  # 64 % 32 == 0 → quantizable


def _eng(model):
    e = BatchedEngine.__new__(BatchedEngine)
    e._model = model
    e.model_name = "test"
    return e


def test_mxfp4_quantizes_linear():
    m = _Tiny()
    assert isinstance(m.fc, nn.Linear) and not isinstance(m.fc, nn.QuantizedLinear)
    _eng(m)._quantize_on_load("mxfp4")
    # The Linear should now be a QuantizedLinear with mxfp4 params.
    assert isinstance(m.fc, nn.QuantizedLinear)
    assert m.fc.bits == 4 and m.fc.group_size == 32


def test_affine_defaults():
    m = _Tiny()
    _eng(m)._quantize_on_load("affine")
    assert isinstance(m.fc, nn.QuantizedLinear)
    assert m.fc.bits == 4 and m.fc.group_size == 64


def test_already_quantized_is_noop():
    m = _Tiny()
    _eng(m)._quantize_on_load("mxfp4")  # now QuantizedLinear
    # Re-applying must be a safe no-op (QuantizedLinear has no to_quantized).
    _eng(m)._quantize_on_load("affine")
    assert isinstance(m.fc, nn.QuantizedLinear)
    assert m.fc.group_size == 32  # unchanged from the mxfp4 pass (not re-quantized)


def test_non_group_aligned_skipped():
    class _Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(48, 48, bias=False)  # 48 % 32 != 0 → skipped for mxfp4

    m = _Odd()
    _eng(m)._quantize_on_load("mxfp4")
    assert isinstance(m.fc, nn.Linear) and not isinstance(m.fc, nn.QuantizedLinear)
