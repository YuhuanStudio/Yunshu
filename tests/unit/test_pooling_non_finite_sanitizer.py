"""/v1/pooling sanitizes non-finite (NaN/Inf) vector components.

pool() returns RAW un-normalized hidden states; a degenerate/overflowing state can yield
NaN/Inf components. FastAPI JSONResponse renders with allow_nan=False, so a single
non-finite component raised an uncaught ValueError at the return (a 500). The other three
scoring endpoints sanitized their scalar scores but /v1/pooling shipped vectors
un-checked. Now coerce non-finite components to 0.0 (mirrors /v1/embeddings).
"""
from __future__ import annotations

import inspect
import json


def test_pooling_has_finite_guard_like_siblings():
    from yunshu_gateway.routers import scoring
    src = inspect.getsource(scoring.create_pooling)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the pooling loop now sanitizes vector components before encoding
    assert "math.isfinite(x) for x in vec" in code
    assert "x if math.isfinite(x) else 0.0 for x in vec" in code


def test_jsonresponse_rejects_nan_proving_the_bug():
    """Prove the failure mode: a NaN component makes JSONResponse.render raise, so the
    guard (which runs before the response is built) is load-bearing, not cosmetic."""
    # JSONResponse uses allow_nan=False — a raw NaN vector would raise here (the 500)
    import math

    from fastapi.responses import JSONResponse

    raised = False
    try:
        _ = JSONResponse({"data": [{"data": [float("nan"), 1.0]}]}).body  # render
    except ValueError:
        raised = True
    assert raised, "expected JSONResponse to reject NaN (allow_nan=False)"

    # the sanitization the route applies turns it into valid JSON
    vec = [float("nan"), float("inf"), 1.0]
    clean = [x if math.isfinite(x) else 0.0 for x in vec]
    assert clean == [0.0, 0.0, 1.0]
    json.dumps(clean, allow_nan=False)  # must not raise
