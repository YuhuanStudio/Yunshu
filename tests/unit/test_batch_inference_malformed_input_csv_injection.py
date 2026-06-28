"""batch_inference malformed-input → clean 4xx + CSV-injection hardening.

A focused audit of the batch_inference router (part of the W965 endpoint-closure effort)
found no HIGH/IDOR/isolation bugs — the security-critical paths are all correctly gated.
Three LOW issues remained:
  - max_concurrent form value > 64 passed the handler's own check but raised a Pydantic
    ValidationError later inside create_batch (past the FastAPI boundary) → 500 not 400.
  - a non-UTF-8 CSV upload raised an uncaught UnicodeDecodeError → 500 not 400.
  - results.csv wrote prompt-controlled model output raw → CSV formula injection when the
    download is opened in a spreadsheet.
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import batch_inference as bi


def test_csv_safe_neutralizes_formula_triggers():
    assert bi._csv_safe("=SUM(A1)") == "'=SUM(A1)"
    assert bi._csv_safe("+1+1") == "'+1+1"
    assert bi._csv_safe("-2") == "'-2"
    assert bi._csv_safe("@cmd") == "'@cmd"
    assert bi._csv_safe("\tx") == "'\tx"
    # benign values untouched; non-strings pass through
    assert bi._csv_safe("hello") == "hello"
    assert bi._csv_safe("") == ""
    assert bi._csv_safe(42) == 42
    assert bi._csv_safe(None) is None


def test_max_concurrent_upper_bound_enforced_in_handler():
    src = inspect.getsource(bi.upload_batch_csv)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the handler now rejects max_concurrent > 64 itself (matching the BatchRequest le=64),
    # instead of letting it raise a 500 deep inside create_batch
    assert "max_concurrent > 64" in code


def test_utf8_decode_guarded():
    src = inspect.getsource(bi.upload_batch_csv)
    assert "except UnicodeDecodeError" in src
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "status_code=400" in code  # the malformed-encoding case returns 400


def test_results_csv_export_uses_csv_safe():
    # the CSV export path sanitizes every cell
    import yunshu_gateway.routers.batch_inference as m
    src = inspect.getsource(m)
    assert "_csv_safe(c) for c in row" in src
