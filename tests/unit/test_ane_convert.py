import numpy as np

from yunshu_engine.ane.convert import dequantize_affine, split_channels, unpack_affine


def _pack(codes: np.ndarray, bits: int) -> np.ndarray:
    out, n = codes.shape
    if bits in (2, 4, 8):
        per = 32 // bits
        c = codes.reshape(out, n // per, per).astype(np.uint64)
        w = np.zeros((out, n // per), dtype=np.uint64)
        for i in range(per):
            w |= c[:, :, i] << np.uint64(bits * i)
        return w.astype(np.uint32)
    c = codes.reshape(out, n // 8, 8).astype(np.uint64)
    w = np.zeros((out, n // 8), dtype=np.uint64)
    for i in range(8):
        w |= c[:, :, i] << np.uint64(bits * i)
    raw = np.stack(
        [(w >> np.uint64(8 * i)) & np.uint64(255) for i in range(bits)], axis=-1
    ).astype(np.uint8)
    return np.ascontiguousarray(raw.reshape(out, -1)).view(np.uint32)


def test_unpack_roundtrip():
    rng = np.random.default_rng(0)
    for bits in (2, 4, 5, 6, 8):
        codes = rng.integers(0, 1 << bits, size=(3, 128), dtype=np.uint8)
        assert np.array_equal(unpack_affine(_pack(codes, bits), bits), codes)


def test_dequant_affine():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 16, size=(2, 128), dtype=np.uint8)
    s = rng.random((2, 2), dtype=np.float32)
    b = rng.random((2, 2), dtype=np.float32)
    w = dequantize_affine(_pack(codes, 4), s, b, 4)
    assert np.allclose(w[1, 70], codes[1, 70] * s[1, 1] + b[1, 1])


def test_split_channels():
    assert split_channels(17408, 0.25) == 4352
    assert split_channels(17408, 0.0) == 64
