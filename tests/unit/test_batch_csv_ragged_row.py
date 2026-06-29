"""(MED): a ragged/short CSV row (fewer fields than the header) crashed the whole
batch upload with a 500.

csv.DictReader fills a short row's missing columns with None (default restval). The upload
parser then did row.get(col, "").strip() → None.strip() → AttributeError, which the
(ValueError, TypeError) guards do NOT catch → uncaught 500 aborting the entire upload. Fix:
DictReader(restval="") so missing cells are empty strings (.strip() safe, cells default),
and custom_id falls back to a generated id when missing/empty.
"""

from __future__ import annotations

import csv
import inspect
import io

from yunshu_gateway.routers import batch_inference as bi


def _parse_row_like_production(row, max_tokens=128):
    """Faithful replica of the upload parser's per-cell defaulting (the .strip() paths)."""
    custom_id = row.get("custom_id") or "GENERATED"
    rmt_raw = row.get("max_tokens", "")
    try:
        rmt = (
            int(rmt_raw) if rmt_raw.strip() else max_tokens
        )  # None.strip() would crash here
    except (ValueError, TypeError):
        rmt = max_tokens
    rt_raw = row.get("temperature", "")
    try:
        rt = float(rt_raw) if rt_raw.strip() else 0.7
    except (ValueError, TypeError):
        rt = 0.7
    return custom_id, rmt, rt


def test_old_behavior_yielded_none_for_short_row():
    # documents the bug: without restval, missing cols are None → .strip() AttributeError
    text = "custom_id,prompt,max_tokens,temperature\nid1,hello\n"
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["max_tokens"] is None
    assert rows[0]["temperature"] is None


def test_ragged_row_defaults_cleanly_with_restval():
    text = "custom_id,prompt,max_tokens,temperature\nid1,hello\n"
    rows = list(csv.DictReader(io.StringIO(text), restval=""))
    # missing cells are now empty strings — .strip() is safe
    assert rows[0]["max_tokens"] == "" and rows[0]["temperature"] == ""
    cid, rmt, rt = _parse_row_like_production(rows[0])
    assert cid == "id1"  # present custom_id kept
    assert rmt == 128 and rt == 0.7  # missing numeric cells default, no crash


def test_missing_custom_id_column_gets_generated_id():
    text = "prompt,max_tokens\nhello,64\n"
    rows = list(csv.DictReader(io.StringIO(text), restval=""))
    cid, rmt, rt = _parse_row_like_production(rows[0])
    assert cid == "GENERATED"  # missing/empty custom_id → fallback, not None
    assert rmt == 64


def test_production_uses_restval_and_custom_id_fallback():
    src = inspect.getsource(bi.upload_batch_csv)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert 'restval=""' in code
    assert 'row.get("custom_id") or' in code
