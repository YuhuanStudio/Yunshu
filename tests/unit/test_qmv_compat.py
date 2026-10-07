"""CPU regression tests for MLX's promoted bias sum transcription."""

from yunshu_engine.kernels.qmv_compat import stock_load_vector


def test_promotes_each_operand_without_changing_weight_loads():
    old = "sum += x[i] + x[i + 1] +\n x[i + 2]; x_thread[i] = x[i] / 16.0f;"
    new = stock_load_vector(old, True)
    assert new == (
        "sum += float(x[i]) + float(x[i + 1]) +\n float(x[i + 2]); "
        "x_thread[i] = x[i] / 16.0f;"
    )
    assert stock_load_vector(old, False) == old
    assert stock_load_vector(new, True) == new


def test_version_boundary_includes_source_dev_build(monkeypatch):
    import importlib.util
    from importlib import metadata
    from pathlib import Path

    path = Path(__file__).parents[2] / "python/yunshu_engine/kernels/qmv_compat.py"
    for installed, expected in [
        ("0.32.3", False),
        ("0.32.4.dev20261007+f8aaf49d", True),
        ("0.32.4", True),
    ]:
        monkeypatch.setattr(metadata, "version", lambda name: installed)
        spec = importlib.util.spec_from_file_location("qmv_contract_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.FLOAT_SUMS is expected


def test_install_syncs_upstream_base_header_once(monkeypatch):
    from mlx_vlm.models import quantized_verifier as qv
    from mlx_vlm.speculative.ops import linear as ops

    from yunshu_engine.kernels import verify_select

    monkeypatch.setattr(verify_select, "FLOAT_SUMS", True)
    monkeypatch.setattr(
        verify_select, "stock_load_vector", lambda h: stock_load_vector(h, True)
    )
    monkeypatch.setattr(
        qv, "_target_verify_qlinear_header", lambda *a, **kw: "sum += x[i] + x[i + 1];"
    )
    monkeypatch.setattr(qv, "optimized_affine_linear", lambda *a: None)
    monkeypatch.setattr(qv, "optimized_affine_linears", lambda *a: None)
    # Register both aliases with monkeypatch so installation is fully undone.
    for name in (
        "_target_verify_optimized_affine_linear",
        "_target_verify_quantized_linears",
    ):
        monkeypatch.setattr(ops, name, getattr(ops, name))
    assert verify_select.install()
    header = qv._target_verify_qlinear_header
    assert header(4, 64) == "sum += float(x[i]) + float(x[i + 1]);"
    assert header(5, 64) == "sum += float(x[i]) + float(x[i + 1]);"
    assert verify_select.install()
    assert qv._target_verify_qlinear_header is header
