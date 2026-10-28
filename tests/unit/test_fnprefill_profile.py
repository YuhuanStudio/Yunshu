"""CPU checks of the Flash-Next prefill profile aggregation."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

base = Path(__file__).resolve().parents[2] / "scripts/research"
sys.path.insert(0, str(base))
spec = importlib.util.spec_from_file_location(
    "fnprefill_profile", base / "fnprefill_profile.py"
)
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def chunk(spans, wall):
    return dict(method="prefill-chunk", rows=1, wall_s=wall, spans=spans)


def span(s, flops=0.0, moved=0.0, count=1):
    return dict(exclusive_s=s, count=count, flops=flops, bytes=moved)


def test_categories():
    assert fp.category("gemm:M2048:K2560:N248320:b6@x") == "lm_head"
    assert fp.category("gemm:M2048:K2560:N512:b8@moe.remainder") == "router (gate gemm)"
    assert (
        fp.category("gemm:M2048:K2560:N640:b4@moe.shared_expert")
        == "shared expert (projections)"
    )
    assert fp.category("gemm:M2048:K2560:N6144:b4@gdn.remainder") == "GDN projections"
    assert fp.category("moe.experts") == "expert gather_qmm"
    assert fp.category("ple.ssd_read") == "PLE SSD read"
    assert fp.category("qsa.indexer.build_mask") == "QSA indexer"


def test_aggregate_scales_per_chunk_and_rates():
    rows = [
        chunk({"moe.experts": span(0.2, 4e12, 40e9), "gdn.core": span(0.1)}, 0.4),
        chunk({"moe.experts": span(0.2, 4e12, 40e9), "gdn.core": span(0.1)}, 0.4),
    ]
    out = fp.aggregate(rows)
    top = out["categories"][0]
    assert top["category"] == "expert gather_qmm"
    assert top["ms_per_chunk"] == 200.0 and top["share"] == pytest.approx(
        0.6667, abs=1e-3
    )
    assert top["tflops"] == 20.0 and top["gbps"] == 200.0
    assert out["unattributed_ms_per_chunk"] == 100.0


def test_no_chunks_fails_closed():
    with pytest.raises(RuntimeError):
        fp.aggregate([])
    assert fp.parse_log("x\n" + fp.PREFIX + json.dumps({"a": 1}) + "\n") == [{"a": 1}]
