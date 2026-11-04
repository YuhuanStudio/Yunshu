"""An engine's explicit refusal of a request shape is a recorded 'unsupported' cell, not a crash (wire-1108 B2:
TensorFold 1.2.1 answered the image cell with HTTP 400 'this server accepts text parts only; image, audio and
video inputs are unsupported' and tfbench died, so the rest of the chunk never ran). Any other HTTP error
still fails the cell."""

from __future__ import annotations

import io
import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import fullcmp_audit as fa  # noqa: E402
import tfbench as tb  # noqa: E402

# the refusal as TensorFold 1.2.1 logged it (wire-1108-work B2 server log); the JSON envelope is the OpenAI error shape
TF_REFUSAL = "this server accepts text parts only; image, audio and video inputs are unsupported"
TF_BODY = json.dumps({"error": {"message": TF_REFUSAL, "type": "invalid_request_error"}})


def _http_error(code, body):
    return urllib.error.HTTPError("http://x/v1/chat/completions", code, "err", {}, io.BytesIO(body.encode()))


def test_the_real_tf_refusal_is_an_engine_refusal_and_other_errors_are_not():
    assert tb.is_engine_refusal(400, TF_BODY)
    assert tb.is_engine_refusal(400, TF_REFUSAL)
    assert not tb.is_engine_refusal(500, TF_BODY)  # a server error is never a refusal
    assert not tb.is_engine_refusal(400, '{"error": {"message": "invalid json at line 3"}}')  # a plain bad request
    assert not tb.is_engine_refusal(429, "unsupported rate")  # rate limit / timeout
    assert not tb.is_engine_refusal(408, TF_BODY)


def _decode(tmp_path, send, media="img1", phases="cold,warm"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "cell-r0.jsonl"
    a = tb.parse_args(
        [
            "--engine", "tf-1.2.1", "--part", "decode", "--rep", "0", "--out", str(out),
            "--only-ctx", "1024", "--only-kind", "prose", "--decode-tokens", "64", "--phases", phases,
            "--media", media,
        ]
    )
    s = types.SimpleNamespace(url="http://x", model="m", engine="tf-1.2.1", log=tmp_path / "n.log")
    with out.open("w") as fh:
        tb.part_decode(s, fh, a)
    return [json.loads(x) for x in out.read_text().splitlines()], out


def test_an_image_refusal_becomes_an_unsupported_row_and_the_cell_stops(tmp_path, monkeypatch):
    calls = []

    def send(url, body, **kw):
        calls.append(body)
        raise _http_error(400, TF_BODY)

    monkeypatch.setattr(tb, "send", send)
    rows, out = _decode(tmp_path, send)
    assert len(calls) == 1  # the warm phase needs the cold reply: the cell stopped at the refusal
    un = [r for r in rows if r.get("part") == "unsupported"]
    assert len(un) == 1 and un[0]["http_status"] == 400 and TF_REFUSAL in un[0]["reason"]
    assert un[0]["media"] == "img1" and un[0]["phase"] == "cold"
    assert not [r for r in rows if r.get("part") == "decode"]
    assert tb.cell_unsupported(str(out)) is True


def test_any_other_http_error_still_fails_the_cell(tmp_path, monkeypatch):
    def send(url, body, **kw):
        raise _http_error(500, "internal error")

    monkeypatch.setattr(tb, "send", send)
    with pytest.raises(urllib.error.HTTPError):
        _decode(tmp_path, send)

    def bad_request(url, body, **kw):
        raise _http_error(400, '{"error": {"message": "max_tokens must be positive"}}')

    monkeypatch.setattr(tb, "send", bad_request)
    with pytest.raises(urllib.error.HTTPError):
        _decode(tmp_path / "b", bad_request)


def test_a_cell_without_an_unsupported_row_is_not_unsupported(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text(json.dumps({"part": "session"}) + "\n" + json.dumps({"part": "decode"}) + "\n")
    assert tb.cell_unsupported(str(f)) is False


def test_the_audit_lists_an_unsupported_cell_and_leaves_it_out_of_the_measurement_guards(tmp_path):
    """The refused cell: not a pass, not a crash. It leaves run.files (no 'no decode rows' failures anywhere)
    and shows in manifest_completeness as UNSUPPORTED with the engine's own message."""
    root = tmp_path / "raw"
    (root / "tf-1.2.1").mkdir(parents=True)
    cell = {"cell": "B2.image.tf-1.2.1.1K.c1.prose.o512.single.th-off.img1.greedy", "engine": "tf-1.2.1", "group": "g"}
    (root / "tf-1.2.1" / f"{cell['cell']}-r0.jsonl").write_text(
        json.dumps({"part": "session", "engine": "tf-1.2.1"})
        + "\n"
        + json.dumps({"part": "unsupported", "http_status": 400, "reason": TF_REFUSAL})
        + "\n"
    )
    manifest = {"cells": [cell], "reps": 1, "pins": {"tf-1.2.1": {}}, "groups": {"g": {"order": ["tf-1.2.1"]}}}
    run = fa.load_run(root, manifest)
    assert run.files == [] and len(run.unsupported) == 1
    res = fa.check_manifest_completeness(run)
    assert res.violations == [], res.violations
    assert any("UNSUPPORTED" in n and TF_REFUSAL in n for n in res.notes), res.notes


def test_tf_1_2_1_image_cells_are_unsupported_with_the_engines_own_message():
    import fullcmp_manifest as fm

    rows, _meta = fm.build_rows()
    fm.apply_cuts(rows, out=io.StringIO())
    img = [r for r in rows if r["engine"] == "tf-1.2.1" and r.get("media") not in (None, "none")]
    assert img and all(r["status"] == "unsupported" for r in img)
    assert all("accepts text parts only" in r["reason"] for r in img)
    # an engine whose census says nothing certain stays costed: the runtime refusal path covers it
    other = [r for r in rows if r["engine"] == "splash-1.5.0" and r.get("media") not in (None, "none")]
    assert other and all(r["status"] != "unsupported" for r in other)
