"""CPU tests for kernels.attn_prefill: header harvest, eligibility and dispatch decisions (no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from yunshu_engine.kernels import attn_prefill as ap  # noqa: E402


def test_eligible_shapes():
    ok = ap.eligible((1, 24, 2048, 256), (1, 4, 8192, 256), mx.bfloat16, 0.0625)
    assert ok
    assert not ap.eligible((1, 24, 512, 256), (1, 4, 8192, 256), mx.bfloat16, 0.0625)
    assert not ap.eligible((1, 24, 2048, 128), (1, 4, 8192, 128), mx.bfloat16, 0.0625)
    assert not ap.eligible((2, 24, 2048, 256), (2, 4, 8192, 256), mx.bfloat16, 0.0625)
    assert not ap.eligible((1, 24, 2048, 256), (1, 4, 8192, 256), mx.float16, 0.0625)
    assert not ap.eligible((1, 24, 2048, 256), (1, 4, 8192, 256), mx.bfloat16, 0.1)
    assert not ap.eligible((1, 24, 4096, 256), (1, 4, 2048, 256), mx.bfloat16, 0.0625)
    assert not ap.eligible((1, 25, 2048, 256), (1, 4, 8192, 256), mx.bfloat16, 0.0625)


def test_skip_reasons():
    q = mx.zeros((1, 24, 2048, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 4, 4096, 256), dtype=mx.bfloat16)

    class KVCache:
        pass

    assert ap.skip_reason(q, k, k, KVCache(), 0.0625, "causal", None) is None
    assert ap.skip_reason(q, k, k, KVCache(), 0.0625, None, None) == "mask:NoneType"
    assert ap.skip_reason(q, k, k, KVCache(), 0.0625, "causal", q) == "sinks"
    assert ap.skip_reason(q, k, k, object(), 0.0625, "causal", None) == "cache:object"


def test_header_swaps_exchange_layout_and_fixes_constants():
    pytest.importorskip("mlx_vlm")
    h = ap.header(True, False)
    assert "(ik * kEPF + i) * 32 + short(simd_lane_id)]" in h
    assert "base + ik * kEPF + i]" not in h
    assert "constant bool align_Q = true;" in h
    assert "constant bool align_K = false;" in h
    assert "constant bool do_causal = true;" in h
    assert "function_constant" not in h.split("yunshu_attn_dsplit")[0]


def test_identity_script_dry_run(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "ident",
        Path(__file__).parents[2] / "scripts/research/prefillspeed_attn_identity.py",
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.main(["--out", str(tmp_path / "o"), "--dry-run"]) == 0
    assert '"cases"' in capsys.readouterr().out
    assert any(r % 32 for r, _, _, _ in m.cases()) and any(
        k % 32 for _, k, _, _ in m.cases()
    )


def test_nax_lane_tile_rows():
    from yunshu_engine.kernels import nax_prefill

    assert nax_prefill.tile_rows(2048, 17408) == 64
    assert nax_prefill.tile_rows(8192, 17408) == 64
    assert nax_prefill.tile_rows(2048, 5120) == 128
    assert nax_prefill.tile_rows(2048, 6144) == 128


def test_short_attention_is_only_enabled_inside_prompt_scope(monkeypatch):
    from yunshu_engine.kernels import gdn_prefill

    monkeypatch.setitem(gdn_prefill._STATE, "enabled", True)
    monkeypatch.setitem(gdn_prefill._STATE, "split", True)
    args = ((1, 24, 1, 256), (1, 4, 32769, 256), mx.bfloat16, 0.0625)
    assert not ap.eligible(*args)
    with gdn_prefill.prefill_scope():
        assert ap.eligible(*args)
    assert not ap.eligible(*args)


@pytest.mark.parametrize(
    "tail,rows,short,warps",
    [
        (True, 1, True, 2),
        (True, 16, True, 2),
        (True, 17, False, 4),
        (False, 16, False, 4),
    ],
)
def test_short_tail_schedule_matches_kernel_geometry(
    monkeypatch, tail, rows, short, warps
):
    from types import SimpleNamespace

    calls = []
    launches = []

    def kernel(*args):
        calls.append(args)

        def launch(**kwargs):
            launches.append(kwargs)
            return ["output"]

        return launch

    monkeypatch.setattr(ap, "kernel", kernel)
    monkeypatch.setattr(ap.gdn_prefill, "in_tail", lambda: tail)
    q = SimpleNamespace(shape=(1, 24, rows, 256))
    k = SimpleNamespace(shape=(1, 4, 32768 + rows, 256))
    assert ap.attend(q, k, k) == "output"
    tile = 16 if short else 32
    assert calls == [(rows % tile == 0, (32768 + rows) % 32 == 0, 24, 4, short)]
    assert launches[0]["grid"] == ((rows + tile - 1) // tile * 32, 24 * warps, 1)
    assert launches[0]["threadgroup"] == (32, warps, 1)


def test_short_kernel_uses_same_bk_and_d_split_with_one_query_simdgroup(monkeypatch):
    captured = {}

    def compile_kernel(**kwargs):
        captured.update(kwargs)
        return "compiled"

    ap.kernel.cache_clear()
    monkeypatch.setattr(ap, "header", lambda aq, ak: "unchanged header")
    monkeypatch.setattr(mx.fast, "metal_kernel", compile_kernel)
    try:
        assert ap.kernel(True, False, 24, 2, True) == "compiled"
        assert captured["header"] == "unchanged header"
        source = captured["source"]
        assert "threadgroup float s_xchg[1][2][32 * 16]" in source
        assert "yunshu_attn_dsplit<bfloat, 16, 32, 256, 1, 2, float, float>" in source
        assert "params.qL_rem = params.qL % 16" in source
    finally:
        ap.kernel.cache_clear()
