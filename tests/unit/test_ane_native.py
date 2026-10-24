"""CPU-only contracts for the optional ANE runtime; no MLX/Metal loads."""

import importlib.util
import struct
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from yunshu_engine.ane.native import NativeSession, mil


def test_blob_records_are_aligned_and_typed():
    b = mil.Blob()
    values = [
        np.arange(9, dtype=np.float16).reshape(3, 3),
        np.ones((2, 5), dtype=np.int8),
    ]
    offsets = [b.add(v) for v in values]
    data = b.finish()
    assert struct.unpack_from("<II", data) == (2, 2)
    for offset, value, dtype in zip(offsets, values, [1, 4], strict=True):
        assert offset % 64 == 0
        assert struct.unpack_from("<IIQQ", data, offset) == (
            0xDEADBEEF,
            dtype,
            value.nbytes,
            offset + 64,
        )
        assert data[offset + 64 : offset + 64 + value.nbytes] == value.tobytes()


def test_strides_match_iosurface_64byte_padding():
    text, stride = mil.surface_type(64, 129)
    assert stride == "[10240, 10240, 160, 1]"
    assert "shape=[1, 1, 64, 129]" in text


@pytest.mark.parametrize("mode", ["fp16", "int8"])
def test_mlp_scaling_and_shape(mode):
    g, u, d = (
        np.ones((32, 64), dtype=np.float16),
        np.ones((32, 64), dtype=np.float16),
        np.ones((64, 32), dtype=np.float16),
    )
    text, blob = mil.mlp_program(g, u, d, 128, mode, h_exp=6)
    assert "func main<ios18>" in text and "tensor_buffer_to_tensor<ios17>" in text
    assert "tanh(x = gh)" in text and "sigmoid(x = g)" not in text
    assert "strides = tensor<int64, [4]>([8192, 8192, 128, 1])" in text
    if mode == "fp16":
        offsets = []
        cursor = 64
        for _ in range(3):
            offsets.append(cursor)
            _, _, size, pos = struct.unpack_from("<IIQQ", blob, cursor)
            cursor = (pos + size + 63) // 64 * 64
        for off, expected in zip(offsets, [1, 64, 1 / 64], strict=True):
            _, _, size, pos = struct.unpack_from("<IIQQ", blob, off)
            assert np.all(
                np.frombuffer(blob[pos : pos + size], dtype=np.float16) == expected
            )
    else:
        assert "constexpr_blockwise_shift_scale" in text
        assert "dequantize(input" not in text


def test_bad_model_is_rejected_before_native_load(monkeypatch, tmp_path):
    load = Mock(side_effect=AssertionError("must not load private API"))
    monkeypatch.setattr("yunshu_engine.ane.native._library", load)
    session, reason = NativeSession.try_create("", b"", [(0, 128)], (64, 128), tmp_path)
    assert session is None and "dimensions" in reason
    load.assert_not_called()


def test_unavailable_and_build_failure_allow_fallback(monkeypatch, tmp_path):
    lib = Mock()
    lib.ys_ane_available.return_value = 0
    lib.ys_ane_error.return_value = b"changed selector"
    monkeypatch.setattr("yunshu_engine.ane.native._library", lambda _: lib)
    assert NativeSession.try_create("", b"", [(64, 128)], (64, 128), tmp_path) == (
        None,
        "changed selector",
    )
    monkeypatch.setattr(
        "yunshu_engine.ane.native._library", Mock(side_effect=OSError("no compiler"))
    )
    assert NativeSession.try_create("", b"", [(64, 128)], (64, 128), tmp_path) == (
        None,
        "no compiler",
    )


def test_failed_evaluation_retires_and_never_reads_output():
    lib = Mock()
    lib.ys_ane_run.return_value = 0
    lib.ys_ane_error.return_value = b"evaluation failed"
    s = NativeSession(lib, 1, [(64, 128), (64, 128)])
    assert s.run() is None and s.reason == "evaluation failed"
    assert s.run() is None
    lib.ys_ane_run.assert_called_once()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        s.copy_output()
    lib.ys_ane_data.assert_not_called()
    s.close()
    s.close()
    lib.ys_ane_destroy.assert_called_once_with(1)


def test_native_probe_parser_and_first_failure_contract():
    path = Path(__file__).resolve().parents[2] / "scripts/research/anenative_probe.py"
    spec = importlib.util.spec_from_file_location("anenative_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parser().parse_args(
        ["--cache", "/Volumes/P5Plus/cache", "--mode", "int8"]
    )
    assert args.rows == 128 and args.mode == "int8"


def test_run_repeats_validation():
    s = NativeSession(Mock(), 1, [])
    for invalid in (0, -1, 10001, 1.5):
        with pytest.raises(ValueError):
            s.run(invalid)


def test_micro_schema_and_economics():
    path = Path(__file__).resolve().parents[2] / "scripts/research/anenative_micro.py"
    spec = importlib.util.spec_from_file_location("anenative_micro", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parser().parse_args(
        ["--cache", "/Volumes/P5Plus/cache", "--out", "/Volumes/P5Plus/result.jsonl"]
    )
    assert args.rows == "128,512,1024,2048"
    assert mod.validate(np.ones((2, 2)), np.ones((2, 2))) == 0
    with pytest.raises(ValueError):
        mod.validate(np.full((2, 2), np.nan), np.ones((2, 2)))
    with pytest.raises(ValueError):
        mod.validate(np.ones((2, 2)) * 2, np.ones((2, 2)))
    result = mod.economics(10.0, 1.0, 1.0 / 17408)
    assert result["best_share"] == pytest.approx(9 / 11)
    assert result["saving_ms"] == pytest.approx(90 / 11)
    assert mod.economics(10.0, 11.0, 0.001)["best_share"] == 0


def test_concurrent_load_parser():
    path = Path(__file__).resolve().parents[2] / "scripts/research/anenative_load.py"
    spec = importlib.util.spec_from_file_location("anenative_load", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parser().parse_args(
        [
            "--cache",
            "/Volumes/P5Plus/cache",
            "--out",
            "/Volumes/P5Plus/result.json",
            "--ready",
            "/Volumes/P5Plus/ready.json",
            "--backend",
            "native",
        ]
    )
    assert args.channels == 2176 and args.rows == 1024
    assert args.layers == "40,41,42,43"


def test_receipt_analysis_requires_complete_and_fits_known_cost(tmp_path, monkeypatch):
    import json

    scripts = Path(__file__).resolve().parents[2] / "scripts/research"
    monkeypatch.syspath_prepend(str(scripts))
    import anenative_analyze as analysis

    receipt = tmp_path / "results.jsonl"
    receipt.write_text('{"kind":"meta"}\n')
    with pytest.raises(ValueError, match="incomplete"):
        analysis.summarize(receipt)
    records = [{"kind": "gpu_ffn", "rows": 128, "median_ms": 10.0}]
    for channel in (64, 1088):
        records.append(
            {
                "kind": "slice",
                "rows": 128,
                "mode": "fp16",
                "channels": channel,
                "arm": "cand",
                "ms": [1 + channel * 0.001],
                "copy_ms": [0.2],
                "rel_l2_fp32": 0.001,
                "cached": False,
            }
        )
    records.append({"complete": True})
    receipt.write_text("".join(json.dumps(row) + "\n" for row in records))
    result = analysis.summarize(receipt)
    assert result["fits"][0]["floor_ms"] == pytest.approx(1.0)
    assert result["fits"][0]["ms_per_1000_channels"] == pytest.approx(1.0)
    records[1]["cached"] = True
    receipt.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="cached"):
        analysis.summarize(receipt)


def test_minimal_scale_program_has_real_op_and_no_weight_records():
    text, blob = mil.scale_program(5120, 1024)
    assert "mul(x = xt, y = fp16(2))" in text
    assert "[5242880, 5242880, 1024, 1]" in text
    assert struct.unpack_from("<II", blob) == (0, 2)
    assert len(blob) == 64


def test_alias_retains_surface_after_session_close(monkeypatch):
    import ctypes
    import gc
    import sys
    from types import ModuleType

    class Alias:
        def __init__(self, view):
            self.view = view

    core = ModuleType("mlx.core")

    def from_dlpack(view, *, copy):
        assert copy is False
        return Alias(view)

    core.from_dlpack = from_dlpack
    pkg = ModuleType("mlx")
    pkg.core = core
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    storage = ctypes.create_string_buffer(128)
    lib = Mock()
    lib.ys_ane_busy.return_value = 0
    lib.ys_ane_stride.return_value = 64
    lib.ys_ane_data.return_value = ctypes.addressof(storage)
    lib.ys_ane_retain.return_value = 2
    session = NativeSession(lib, 1, [(2, 32), (2, 32)])
    alias, reason = session.try_alias(0)
    assert reason is None and alias.view.shape == (2, 32)
    session.close()
    lib.ys_ane_destroy.assert_called_once_with(1)
    del alias
    gc.collect()
    assert [call.args[0] for call in lib.ys_ane_destroy.call_args_list] == [1, 2]


def test_alias_busy_and_ticket_failure_are_unavailable(monkeypatch):
    import sys
    from types import ModuleType

    from yunshu_engine.ane.native import MlxTicket

    pkg, core = ModuleType("mlx"), ModuleType("mlx.core")
    pkg.core = core
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    lib = Mock()
    lib.ys_ane_busy.return_value = 1
    session = NativeSession(lib, 1, [(2, 32), (2, 32)])
    alias, reason = session.try_alias()
    assert alias is None and "in flight" in reason
    lib.ys_ane_retain.assert_not_called()
    lib.ys_ane_mlx_finish.return_value = 0
    lib.ys_ane_error.return_value = b"native timeout"
    ticket = MlxTicket(session, 3)
    assert ticket.finish() is False and session.reason == "native timeout"
    ticket.close()
    ticket.close()
    lib.ys_ane_mlx_destroy.assert_called_once_with(3)
    session.close()


def test_economics_does_not_extrapolate_or_claim_negative_copy_savings(
    tmp_path, monkeypatch
):
    scripts = Path(__file__).resolve().parents[2] / "scripts/research"
    monkeypatch.syspath_prepend(str(scripts))
    import anenative_analyze as analysis

    points = [(64, 1.064, 1.0), (1088, 2.088, 1.0)]
    result = analysis.measured_curve(10.0, points)
    assert result["channels"] == 1088
    assert result["saving_ms"] == pytest.approx(0.625)
    copied = analysis.measured_curve(10.0, points, serialize_copies=True)
    assert copied["channels"] == 0 and copied["saving_ms"] == 0
    with pytest.raises(ValueError):
        analysis.measured_curve(10.0, [(64, 1.0, 0.0), (64, 2.0, 0.0)])
