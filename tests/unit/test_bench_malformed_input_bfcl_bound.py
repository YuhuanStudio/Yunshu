"""bench malformed-input → clean 422 + bfcl max_samples upper bound.

Final batch of the endpoint-closure effort. The bench + cached_contents PATCH audit
found no HIGH/IDOR — both are hardened (can_benchmark/can_infer gating, ownership, SSRF
defense, benchmark serialization lock). Two LOW in bench:
  - a gemm_sizes sublist of length != 3 raised ValueError at the `for m, n, k in ...` unpack
    → caught by the generic handler → 500 instead of 422.
  - bfcl-eval max_samples had no upper bound.
"""

from __future__ import annotations

import pydantic
import pytest

from yunshu_gateway.routers.bench import BFCLEvalRequest, RooflineModelRequest


def test_gemm_sizes_must_be_triples():
    assert RooflineModelRequest(gemm_sizes=[[1, 2, 3]]).gemm_sizes == [[1, 2, 3]]
    # wrong arity → rejected at the schema (422), not a 500 at unpack time
    for bad in ([[1, 2]], [[1, 2, 3, 4]], [[1, 2, 3], [4, 5]]):
        with pytest.raises(pydantic.ValidationError):
            RooflineModelRequest(gemm_sizes=bad)
    # non-positive values rejected
    with pytest.raises(pydantic.ValidationError):
        RooflineModelRequest(gemm_sizes=[[1, -2, 3]])
    with pytest.raises(pydantic.ValidationError):
        RooflineModelRequest(gemm_sizes=[[0, 2, 3]])


def test_bfcl_max_samples_upper_bound():
    with pytest.raises(pydantic.ValidationError):
        BFCLEvalRequest(model="m", max_samples=100001)
    # boundary + default ok
    assert BFCLEvalRequest(model="m", max_samples=100000).max_samples == 100000
    assert BFCLEvalRequest(model="m").max_samples == 10
    assert (
        BFCLEvalRequest(model="m", max_samples=0).max_samples == 0
    )  # 0 = all, still valid
