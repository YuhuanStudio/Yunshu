"""CPU tests of the Splash-derived ANE FFN split (program text, calibration model)."""

import numpy as np
import pytest

from yunshu_engine.ane import ffn_calibration as cal
from yunshu_engine.ane import ffn_program as prog


def test_mt19937_matches_cpp_reference_stream():
    # std::mt19937(5489)() == 3499211612 (the default-seed first output)
    rs = np.random.RandomState(5489)
    assert int(rs.randint(0, 2**32, size=1, dtype=np.uint64)[0]) == 3499211612


def test_signs_and_rotation_are_orthogonal():
    signs = prog.rotation_signs()
    assert signs.shape == (512,) and set(np.unique(signs)) <= {-1.0, 1.0}
    for n in (128, 512):
        r = prog.rotation_matrix(n, signs)
        assert np.allclose(r @ r.T, np.eye(n), atol=1e-5)
    w = prog.rotation_weight(1024, signs)
    assert w.shape == (1024, 512, 1, 1) and w.dtype == np.float16
    assert np.array_equal(w[512, :, 0, 0], w[0, :, 0, 0])


def test_shape_surfaces_and_inputs():
    shape = prog.Shape.of(5120, 17408, 64, 8)
    assert (
        shape.ane == 4096 and shape.gpu == 13312 and shape.down_segments == (2560, 1536)
    )
    mem = prog.allocate(shape)
    names = [i.name for i in prog.program_inputs(shape, mem)]
    assert names == [
        "x0",
        "x1",
        "tx",
        "wg0",
        "wu0",
        "wg1",
        "wu1",
        "sg",
        "su",
        "wd0",
        "wd1",
        "sd",
    ]
    assert mem.partial.rows == 5121 and mem.inputs[0].element == "int8"
    with pytest.raises(ValueError):
        prog.Shape.of(5120, 17408, 64, 34)


def test_function_text_structure():
    shape = prog.Shape.of(5120, 17408, 64, 8)
    mem = prog.allocate(shape)
    ins = prog.program_inputs(shape, mem)
    text = prog.program_text(shape, ins, mem.partial, [512, 640])
    assert text.count("func ffn") == 2 and "func ffn512<ios18>(" in text
    assert text.count("matmul(") == 2 * (2 * 2 + 2)
    assert "conv(" in text and "groups = int32(8)" in text and "reduce_max" in text
    assert "tensor<int8, [1, 1, 4096, 512]> hq = quantize(" in text
    assert "values = (ds, hscale)" in text
    assert prog.fp16(0.5) == "fp16(0x1p-1)" and prog.fp16(2.0**-9) == "fp16(0x1p-9)"
    # narrower function reads leading columns of the max-row surface (stride of 2048 rows)
    assert (
        "strides=[5242880, 5242880, 2048, 1]"
        in text.split("func ffn512")[1].split("\n")[0]
    )


def test_calibration_model_and_choice():
    low = cal.Timing(0.4, 8.0, 9.0)
    high = cal.Timing(0.8, 5.0, 14.0)
    model = cal.fit(low, high)
    assert model.ane(0.8) == pytest.approx(14.0)
    units = cal.choose(model, 19.0, 34, 33)
    assert units is not None and 1 < units < 34
    assert cal.choose(model, 8.0, 34, 33) is None
    assert cal.nearest_units(0.4, 34) == 14


def test_minimum_rows_breaker_and_calibration_text():
    gpu = (cal.GpuPoint(512, 5.0), cal.GpuPoint(2048, 19.0))
    funcs = [cal.FunctionTiming(1921, 2048, 14.0), cal.FunctionTiming(1793, 1920, 13.0)]
    t = cal.ChunkTimings(gpu, funcs)
    assert cal.minimum_rows(t) == 1793
    t2 = cal.ChunkTimings(gpu, [cal.FunctionTiming(1921, 2048, 25.0)])
    assert cal.minimum_rows(t2) is None
    c = cal.Calibration(8, 1793, gpu)
    assert cal.Calibration.parse(c.text()) == c
    assert cal.Calibration.parse("garbage") is None
    b = cal.Breaker(gpu)
    out = ""
    for _ in range(cal.WINDOW):
        out = b.add(2048, 4, 4 * 25.0)
    assert out.startswith("losing to the GPU alone")
    ok = cal.Breaker(gpu)
    assert all(ok.add(2048, 4, 4 * 10.0) == "" for _ in range(2 * cal.WINDOW))


def test_confirm_retimes_noisy_function_so_minimum_rows_reaches_the_bottom():
    from yunshu_engine.ane import ffn_calibration as cal

    gpu = (cal.GpuPoint(512, 4.8), cal.GpuPoint(2048, 19.2))
    # 1920's first sample is a slow outlier; true cost is 0.9 of the GPU alone.
    funcs = [
        cal.FunctionTiming(1921, 2048, 17.0),
        cal.FunctionTiming(1793, 1920, 25.0),
        cal.FunctionTiming(1665, 1792, 14.0),
    ]
    t = cal.ChunkTimings(gpu, funcs)
    assert cal.minimum_rows(t) == 1921
    made = cal.confirm(t, lambda f: 0.9 * (4.8 + 0.01 * (f.rows - 512)))
    assert made == 1
    assert cal.minimum_rows(t) == 1665


def test_confirm_keeps_a_real_loss():
    from yunshu_engine.ane import ffn_calibration as cal

    gpu = (cal.GpuPoint(512, 4.8), cal.GpuPoint(2048, 19.2))
    t = cal.ChunkTimings(gpu, [cal.FunctionTiming(512, 2048, 30.0)])
    cal.confirm(t, lambda f: 30.0)
    assert cal.minimum_rows(t) is None


def test_padded_rows_may_lose_a_little_below_a_winning_function_top():
    from yunshu_engine.ane import ffn_calibration as cal

    gpu = (cal.GpuPoint(512, 4.8), cal.GpuPoint(2048, 19.2))
    line = cal.gpu_alone_line(gpu)
    # Each function takes 0.95 of the GPU alone at its top row: its lowest row loses ~5%.
    funcs = [
        cal.FunctionTiming(1921, 2048, 0.95 * line(2048)),
        cal.FunctionTiming(1793, 1920, 0.95 * line(1920)),
        cal.FunctionTiming(1665, 1792, 0.95 * line(1792)),
        cal.FunctionTiming(512, 1664, 0.95 * line(1664)),
    ]
    # The last function spans too far down (padding loses >10% at 512): the walk stops inside it.
    assert 1300 < cal.minimum_rows(cal.ChunkTimings(gpu, funcs)) < 1665
