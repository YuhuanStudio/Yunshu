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
TF_REFUSAL = (
    "this server accepts text parts only; image, audio and video inputs are unsupported"
)
TF_BODY = json.dumps(
    {"error": {"message": TF_REFUSAL, "type": "invalid_request_error"}}
)


@pytest.fixture(autouse=True)
def _prompt_corpus(tmp_path_factory, monkeypatch):
    """The decode cell reads its prompt file from tfbench.WORK/prompts (the benchmark machine's corpus
    on the build volume): a stand-in file keeps the test about the refusal handling."""
    work = tmp_path_factory.mktemp("tfwork")
    (work / "prompts").mkdir()
    (work / "prompts" / "prose-1024.txt").write_text("a stand-in prompt. " * 40)
    monkeypatch.setattr(tb, "WORK", work)
    # no tokenizer: the default model path lives under /Volumes/P5Plus, which the ci-local sandbox denies,
    # and a stand-in prompt needs no length fitting (decode_prompt returns it unfitted for tok=None)
    monkeypatch.setattr(tb.pl, "load_tokenizer", lambda path=None: None)


def _http_error(code, body):
    return urllib.error.HTTPError(
        "http://x/v1/chat/completions", code, "err", {}, io.BytesIO(body.encode())
    )


def test_the_real_tf_refusal_is_an_engine_refusal_and_other_errors_are_not():
    assert tb.is_engine_refusal(400, TF_BODY)
    assert tb.is_engine_refusal(400, TF_REFUSAL)
    assert not tb.is_engine_refusal(
        500, '{"error": {"message": "internal error"}}'
    )  # a plain server error is not
    assert not tb.is_engine_refusal(
        400, '{"error": {"message": "invalid json at line 3"}}'
    )  # a plain bad request
    assert not tb.is_engine_refusal(429, "unsupported rate")  # rate limit / timeout
    assert not tb.is_engine_refusal(408, TF_BODY)


def _decode(tmp_path, send, media="img1", phases="cold,warm"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "cell-r0.jsonl"
    a = tb.parse_args(
        [
            "--engine",
            "tf-1.2.1",
            "--part",
            "decode",
            "--rep",
            "0",
            "--out",
            str(out),
            "--only-ctx",
            "1024",
            "--only-kind",
            "prose",
            "--decode-tokens",
            "64",
            "--phases",
            phases,
            "--media",
            media,
        ]
    )
    s = types.SimpleNamespace(
        url="http://x", model="m", engine="tf-1.2.1", log=tmp_path / "n.log"
    )
    with out.open("w") as fh:
        tb.part_decode(s, fh, a)
    return [json.loads(x) for x in out.read_text().splitlines()], out


def test_an_image_refusal_becomes_an_unsupported_row_and_the_cell_stops(
    tmp_path, monkeypatch
):
    calls = []

    def send(url, body, **kw):
        calls.append(body)
        raise _http_error(400, TF_BODY)

    monkeypatch.setattr(tb, "send", send)
    rows, out = _decode(tmp_path, send)
    assert (
        len(calls) == 1
    )  # the warm phase needs the cold reply: the cell stopped at the refusal
    un = [r for r in rows if r.get("part") == "unsupported"]
    assert (
        len(un) == 1 and un[0]["http_status"] == 400 and TF_REFUSAL in un[0]["reason"]
    )
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
    f.write_text(
        json.dumps({"part": "session"}) + "\n" + json.dumps({"part": "decode"}) + "\n"
    )
    assert tb.cell_unsupported(str(f)) is False


def test_the_audit_lists_an_unsupported_cell_and_leaves_it_out_of_the_measurement_guards(
    tmp_path,
):
    """The refused cell: not a pass, not a crash. It leaves run.files (no 'no decode rows' failures anywhere)
    and shows in manifest_completeness as UNSUPPORTED with the engine's own message."""
    root = tmp_path / "raw"
    (root / "tf-1.2.1").mkdir(parents=True)
    cell = {
        "cell": "B2.image.tf-1.2.1.1K.c1.prose.o512.single.th-off.img1.greedy",
        "engine": "tf-1.2.1",
        "group": "g",
    }
    (root / "tf-1.2.1" / f"{cell['cell']}-r0.jsonl").write_text(
        json.dumps({"part": "session", "engine": "tf-1.2.1"})
        + "\n"
        + json.dumps({"part": "unsupported", "http_status": 400, "reason": TF_REFUSAL})
        + "\n"
    )
    manifest = {
        "cells": [cell],
        "reps": 1,
        "pins": {"tf-1.2.1": {}},
        "groups": {"g": {"order": ["tf-1.2.1"]}},
    }
    run = fa.load_run(root, manifest)
    assert run.files == [] and len(run.unsupported) == 1
    res = fa.check_manifest_completeness(run)
    assert res.violations == [], res.violations
    assert any("UNSUPPORTED" in n and TF_REFUSAL in n for n in res.notes), res.notes


def test_tf_1_2_1_image_cells_are_unsupported_with_the_engines_own_message():
    import fullcmp_manifest as fm

    rows, _meta = fm.build_rows()
    fm.apply_cuts(rows, out=io.StringIO())
    img = [
        r
        for r in rows
        if r["engine"] in ("tf-1.2.1", "tf-1.1.8")
        and r.get("media") not in (None, "none")
    ]
    assert img and all(r["status"] == "unsupported" for r in img)
    assert all("accepts text parts only" in r["reason"] for r in img)
    # an engine whose census says nothing certain stays costed: the runtime refusal path covers it
    other = [
        r
        for r in rows
        if r["engine"] == "splash-1.5.0" and r.get("media") not in (None, "none")
    ]
    assert other and all(r["status"] != "unsupported" for r in other)


def test_an_image_cell_4xx_about_the_image_is_a_refusal_but_a_plain_bad_request_is_not():
    # llama-server without an mmproj answers an image part with a 400 that talks about the image
    llama = '{"error": {"message": "image input is not supported - hint: provide the mmproj file"}}'
    assert tb.is_engine_refusal(400, llama, "img1")
    odd = '{"error": {"message": "unknown content part type image_url"}}'
    assert tb.is_engine_refusal(400, odd, "img4")
    assert not tb.is_engine_refusal(
        400, odd, "none"
    )  # a text cell never counts on the image wording
    assert not tb.is_engine_refusal(
        400, '{"error": {"message": "temperature out of range"}}', "img1"
    )
    assert not tb.is_engine_refusal(500, odd, "img1")


# --- the restart cell's evidence verdict reads both servers' logs (wire-1108 B2 p3, tf-1.2.1) -------------------

RESTART_LOG = (
    Path(__file__).parent / "fixtures" / "tf_1_2_1_restart_cell.server.log"
).read_text()


def _restart_srv(tmp_path, monkeypatch):
    """The relaunch truncated the live log: it holds the second server only; the first server's bytes (its startup
    and the cold request's done line) were saved at the restart (_LOG_PREFIX). The fixture is the cell's real slice."""
    second = RESTART_LOG.index(
        "info: prompt cache: 16.0 GiB", RESTART_LOG.index("finish=length")
    )
    first_part, second_part = RESTART_LOG[:second], RESTART_LOG[second:]
    monkeypatch.setattr(tb, "_LOG_PREFIX", [first_part.encode()])
    srv = object.__new__(tb.Srv)
    srv.engine = "tf-1.2.1"
    srv.requested_spec_mode = "dflash"
    srv.log = tmp_path / "live.log"
    srv.log.write_text(second_part)
    srv.ready_log_offset = len(
        second_part.split("[tensorfold] serving")[0]
    )  # the relaunched server's startup
    srv.first_ready_log_offset = len(first_part.split("[tensorfold] 127.0.0.1")[0])
    srv.probe_text = lambda: ""
    return srv


def _restart_rows():
    done = {  # the two measured requests as tfbench recorded them (cold on server 1, restart on server 2)
        "cold": dict(pt=1024, ct=512, cached=0, sha="5d2fa71b70f9ac2a"),
        "restart": dict(pt=1024, ct=512, cached=0, sha="5d2fa71b70f9ac2a"),
    }
    return [dict(part="decode", phase=p, finish="length", **v) for p, v in done.items()]


def test_restart_cell_evidence_reads_the_pre_restart_log_for_the_cold_row(
    tmp_path, monkeypatch
):
    """Failing-before (wire-1108 B2 p3): the final check read only the post-restart log, the cold row's done line was
    gone and the verdict was 'engaged=unverified (evidence: startup_only)'. It now reads the saved first-server bytes
    then the live log, so both rows correlate with the server process that served them."""
    monkeypatch.setenv("TFB_FULLCMP", "1")
    srv = _restart_srv(tmp_path, monkeypatch)
    log, ready = srv.evidence_log()
    assert (
        log.count("[tensorfold] done chatcmpl") >= 2
    )  # the cold row's line and the restart row's line
    srv.verify_spec_mode(final=True, rows=_restart_rows())
    assert srv.engaged_spec_mode == "dflash", (
        srv.engaged_spec_mode,
        srv.engaged_evidence_kind,
    )


def test_without_the_saved_first_log_the_same_cell_is_startup_only(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TFB_FULLCMP", "1")
    srv = _restart_srv(tmp_path, monkeypatch)
    monkeypatch.setattr(
        tb, "_LOG_PREFIX", []
    )  # the old behaviour: only the live (post-restart) log
    with pytest.raises(RuntimeError, match="startup_only"):
        srv.verify_spec_mode(final=True, rows=_restart_rows())


# --- restart cells: the disk tier decides (manifest status and audit label) ------------------------------------


def _restart_cf(engine, cached):
    cell = {
        "cell": f"B2.restart.{engine}.1K.c1.code.o512.restart.th-off.none.greedy",
        "engine": engine,
        "group": "g",
    }
    rows = [
        {"part": "decode", "phase": "cold", "cached": 0},
        {"part": "decode", "phase": "restart", "cached": cached},
    ]
    return fa.CellFile(
        cell, 0, Path("x"), {"part": "session", "fresh_process": True}, rows
    )


def _cold_cache(*cfs):
    return fa.check_cold_cache(
        fa.Run(manifest={}, files=list(cfs), missing=[], extra=[], parse_errors=[])
    )


def test_restart_miss_on_an_engine_without_a_disk_tier_is_a_labelled_outcome_not_a_failure():
    res = _cold_cache(
        _restart_cf("tf-1.2.1", 0),
        _restart_cf("tf-1.1.8", 0),
        _restart_cf("llama-cpp-b12061", 0),
    )
    assert res.violations == [], res.violations
    assert sum("UNSUPPORTED" in n and "restart" in n for n in res.notes) == 3
    assert any("hits 0, misses 1" in n for n in res.notes)  # TF's own log line is cited


def test_restart_miss_on_an_engine_with_a_disk_tier_fails_and_a_hit_on_a_no_tier_engine_contradicts_the_evidence():
    tiers = (
        "yunshu",
        "omlx-0.7.3",
        "splash-1.5.0",
        "mlx-serve-26.11.1",
        "kapitsa-0.2.26",
        "mlx-vlm-server",
    )
    res = _cold_cache(*(_restart_cf(e, 0) for e in tiers))
    assert len(res.violations) == len(tiers) and all(
        "must hit the disk tier" in v for v in res.violations
    )
    res = _cold_cache(_restart_cf("tf-1.2.1", 768))
    assert res.violations and "evidence says no disk tier" in res.violations[0]
    assert (
        _cold_cache(_restart_cf("yunshu", 960)).violations == []
    )  # a real disk-tier hit passes


def test_manifest_restart_cells_follow_the_disk_tier_evidence():
    import fullcmp_manifest as fm

    rows, _meta = fm.build_rows()
    fm.apply_cuts(rows, out=io.StringIO())
    restart = {r["engine"]: r for r in rows if r.get("turns") == "restart"}
    for eng in ("tf-1.2.1", "tf-1.1.8", "llama-cpp-b12061", "mlx-lm-server"):
        assert restart[eng]["status"] == "unsupported", eng
    assert "hits 0, misses 1" in restart["tf-1.2.1"]["reason"]
    assert "no automatic disk tier" in restart["llama-cpp-b12061"]["reason"]
    # an engine with a disk tier the launch turns on is measured: yunshu (settings.py:211, default on), oMLX
    # (--paged-ssd-cache-dir), Splash (--persistent-cache), mlx-serve (--prefix-cache-disk), Kapitsa (--disk-cache auto,
    # default), mlx-vlm (APC_ENABLED=1, disk persistence default)
    for eng in (
        "yunshu",
        "yunshu-pb2",
        "omlx-0.7.3",
        "omlx-0.7.3rc1",
        "splash-1.5.0",
        "mlx-serve-26.11.1",
        "kapitsa-0.2.26",
        "mlx-vlm-server",
    ):
        assert restart[eng]["status"] != "unsupported", eng


def test_mlx_serve_launch_enables_its_documented_ssd_prefix_tier():
    """Lead ruling: an engine's documented disk tier is on for every cell (docs/cli.md:90 of mlx-serve: --prefix-cache-disk,
    default off), so the restart cell measures it."""
    import bench_engines as be

    launch = be.build_launch("mlx-serve-26.11.1", 18990, "/tmp/h", {})
    cmd = list(launch.cmd)
    assert cmd[cmd.index("--prefix-cache-disk") + 1] == "20GB"
    assert launch.flags["prefix_cache_disk"].startswith("20GB")


def test_mlx_vlm_launch_turns_prefix_caching_on_and_splash_its_persistent_cache():
    """Each engine's launch enables the documented cache features a user turns on (lead rule, 2026-11-04)."""
    import bench_engines as be

    vlm = be.build_launch("mlx-vlm-server", 18990, "/tmp/h", {})
    assert (
        vlm.env["APC_ENABLED"] == "1"
    )  # server/runtime_config.py:76, :360: prefix caching is off by default
    splash = be.build_launch("splash-1.5.0", 18990, "/tmp/h", {})
    assert "--persistent-cache" in splash.cmd  # server/serve_options.py:555-562


# --- an error event inside a 200 stream (wire-1108rest, mlx-vlm kind/json-schema) ------------------------------------

MLX_VLM_STREAM_ERROR = (
    "Structured response_format is not supported with speculative decoding."
)


def _sse_server(frames):
    """A local server answering every chat completion with HTTP 200 and the given SSE data frames."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for f in frames:
                self.wfile.write(f"data: {f}\n\n".encode())
            self.wfile.flush()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _decode_kind(url, tmp_path, kind="json-schema"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "cell-r0.jsonl"
    a = tb.parse_args(
        [
            "--engine",
            "mlx-vlm-server",
            "--part",
            "decode",
            "--rep",
            "0",
            "--out",
            str(out),
            "--only-ctx",
            "1024",
            "--only-kind",
            kind,
            "--decode-tokens",
            "64",
            "--phases",
            "cold",
        ]
    )
    s = types.SimpleNamespace(
        url=url, model="m", engine="mlx-vlm-server", log=tmp_path / "n.log"
    )
    with out.open("w") as fh:
        tb.part_decode(s, fh, a)
    return [json.loads(x) for x in out.read_text().splitlines()], out


def test_a_stream_error_event_saying_unsupported_is_a_recorded_refusal(tmp_path):
    """Failing-before: HTTP 200 with an error frame raised 'SSE error ...' and the chunk aborted. The real mlx-vlm
    message now becomes the 'unsupported' row, labelled, and the cell closes."""
    srv = _sse_server([json.dumps({"error": MLX_VLM_STREAM_ERROR})])
    try:
        rows, out = _decode_kind(f"http://127.0.0.1:{srv.server_port}", tmp_path)
    finally:
        srv.shutdown()
    un = [r for r in rows if r.get("part") == "unsupported"]
    assert (
        len(un) == 1
        and un[0]["reason"] == MLX_VLM_STREAM_ERROR
        and un[0]["stream_error"] is True
    )
    assert (
        un[0]["label"]
        == "unsupported (json-schema with speculative decoding, engine message)"
    )
    assert un[0]["http_status"] is None and un[0]["kind"] == "json-schema"
    assert tb.cell_unsupported(str(out)) is True
    assert not [r for r in rows if r.get("part") == "decode"]


def test_an_error_object_with_a_message_is_read_and_any_other_stream_error_still_fails(
    tmp_path,
):
    srv = _sse_server(
        [
            json.dumps(
                {
                    "error": {
                        "message": MLX_VLM_STREAM_ERROR,
                        "type": "invalid_request_error",
                    }
                }
            )
        ]
    )
    try:
        rows, _ = _decode_kind(f"http://127.0.0.1:{srv.server_port}", tmp_path / "a")
    finally:
        srv.shutdown()
    assert [r["reason"] for r in rows if r.get("part") == "unsupported"] == [
        MLX_VLM_STREAM_ERROR
    ]
    srv = _sse_server([json.dumps({"error": "CUDA out of memory"})])
    try:
        with pytest.raises(tb.StreamError, match="out of memory"):
            _decode_kind(f"http://127.0.0.1:{srv.server_port}", tmp_path / "b")
    finally:
        srv.shutdown()


def test_the_audit_lists_the_stream_refusal_with_its_label(tmp_path):
    root = tmp_path / "raw"
    (root / "mlx-vlm-server").mkdir(parents=True)
    cell = {
        "cell": "B2.kind.mlx-vlm-server.8K.c1.json-schema",
        "engine": "mlx-vlm-server",
        "group": "g",
    }
    (root / "mlx-vlm-server" / f"{cell['cell']}-r0.jsonl").write_text(
        json.dumps({"part": "session"})
        + "\n"
        + json.dumps(
            {
                "part": "unsupported",
                "http_status": None,
                "stream_error": True,
                "reason": MLX_VLM_STREAM_ERROR,
                "label": "unsupported (json-schema with speculative decoding, engine message)",
            }
        )
        + "\n"
    )
    manifest = {
        "cells": [cell],
        "reps": 1,
        "pins": {"mlx-vlm-server": {}},
        "groups": {"g": {"order": ["mlx-vlm-server"]}},
    }
    res = fa.check_manifest_completeness(fa.load_run(root, manifest))
    assert res.violations == []
    assert any(
        "unsupported (json-schema with speculative decoding, engine message)" in n
        and MLX_VLM_STREAM_ERROR in n
        for n in res.notes
    ), res.notes


# --- llama-server without an mmproj: HTTP 500 with the engine's own refusal (wire-1108rest llama.cpp image cell) -------

LLAMA_IMAGE_BODY = (
    '{"error":{"code":500,"message":"image input is not supported - hint: if this is unexpected, you may need to '
    'provide the mmproj"}}'
)


def test_a_5xx_whose_body_says_the_input_is_not_supported_is_a_refusal_other_5xx_are_not():
    assert tb.is_engine_refusal(500, LLAMA_IMAGE_BODY, "img1")
    assert tb.is_engine_refusal(
        500, LLAMA_IMAGE_BODY
    )  # the body alone decides: no media needed
    assert not tb.is_engine_refusal(
        500, '{"error": {"message": "internal error"}}', "img1"
    )
    assert not tb.is_engine_refusal(
        500, '{"error": {"message": "CUDA out of memory"}}', "img1"
    )
    assert not tb.is_engine_refusal(
        503, '{"error": {"message": "unsupported operation in kernel"}}', "img1"
    )  # crash text
    assert not tb.is_engine_refusal(502, "bad gateway", "img1")


def test_the_real_llama_image_500_becomes_an_unsupported_row(tmp_path, monkeypatch):
    def send(url, body, **kw):
        raise _http_error(500, LLAMA_IMAGE_BODY)

    monkeypatch.setattr(tb, "send", send)
    rows, out = _decode(tmp_path, send)
    un = [r for r in rows if r.get("part") == "unsupported"]
    assert (
        len(un) == 1
        and un[0]["http_status"] == 500
        and "provide the mmproj" in un[0]["reason"]
    )
    assert tb.cell_unsupported(str(out)) is True


def test_a_real_5xx_failure_still_fails_the_cell(tmp_path, monkeypatch):
    def send(url, body, **kw):
        raise _http_error(500, '{"error": {"message": "internal error"}}')

    monkeypatch.setattr(tb, "send", send)
    with pytest.raises(urllib.error.HTTPError):
        _decode(tmp_path, send)


def test_llama_cpp_image_cells_are_unsupported_in_this_run_with_the_mmproj_follow_up():
    import fullcmp_manifest as fm

    rows, _meta = fm.build_rows()
    fm.apply_cuts(rows, out=io.StringIO())
    img = [
        r
        for r in rows
        if r["engine"] == "llama-cpp-b12061" and r.get("media") not in (None, "none")
    ]
    assert img and all(r["status"] == "unsupported" for r in img)
    for r in img:
        assert r["reason"].startswith(
            "unsupported in this run (mmproj not provisioned;"
        )
        assert (
            "mmproj-F16.gguf" in r["reason"]
            and "--mmproj" in r["reason"]
            and "follow-up" in r["reason"]
        )


# --- prompt_identity on image cells (wire-1108 image cells, copied values) -----------------------------------------

IMG_FILE_SHA = "657679481d9e3cb65ab054cb53167689c39eb7ca688d9aba4c4f0488b5ab15c4"
IMG_REQ_SHA = "154561527bc387ca8aa80b353cf4a44e645b40326e92c86afad956b25e50a569"


def _img_cf(engine, pt, req_sha=IMG_REQ_SHA, **extra):
    cell = {
        "cell": f"B2.image.{engine}",
        "engine": engine,
        "group": "B2.1K.code.c1.img1",
    }
    row = {
        "part": "decode",
        "phase": "cold",
        "pt": pt,
        "media": "img1",
        "image_parts": 1,
        "image_tokens": None,
        "prompt_ids_sha": req_sha,
        "prompt_file_sha256": IMG_FILE_SHA,
        "prompt_file": "/x/code-1024.txt",
        **extra,
    }
    return fa.CellFile(cell, 0, Path("x"), {"part": "session"}, [row])


def _prompt_identity(monkeypatch, *cfs):
    monkeypatch.setattr(
        fa, "load_prompt_file_table", lambda: {"code-1024.txt": IMG_FILE_SHA}
    )
    return fa.check_prompt_identity(
        fa.Run(manifest={}, files=list(cfs), missing=[], extra=[], parse_errors=[])
    )


def test_image_cells_with_different_image_token_counts_agree_on_the_prompt(monkeypatch):
    """Failing-before (wire-1108 images): yunshu, Kapitsa, oMLX, Splash, mlx-vlm report pt 1090 and mlx-serve 1091 for the
    same request (each vision encoder emits its own image token count): 'engines disagree on prompt (pt, sha)'. The
    identity is the request as sent, the prompt file and the image."""
    res = _prompt_identity(
        monkeypatch,
        _img_cf("yunshu", 1090),
        _img_cf("kapitsa-0.2.26", 1090),
        _img_cf("mlx-serve-26.11.1", 1091),
    )
    assert res.violations == [], res.violations


def test_image_cells_still_fail_when_the_request_the_file_or_the_image_differs(
    monkeypatch,
):
    res = _prompt_identity(
        monkeypatch,
        _img_cf("yunshu", 1090),
        _img_cf("splash-1.5.0", 1090, req_sha="b" * 64),
    )
    assert (
        res.violations
        and "disagree on the image prompt" in res.violations[0]
        and "request" in res.violations[0]
    )
    res = _prompt_identity(
        monkeypatch,
        _img_cf("yunshu", 1090, image_sha256="a" * 64),
        _img_cf("omlx-0.7.3", 1090, image_sha256="c" * 64),
    )
    assert any("image" in v for v in res.violations)
    res = _prompt_identity(
        monkeypatch,
        _img_cf("yunshu", 1090, text_tokens=1066),
        _img_cf("omlx-0.7.3", 1091, text_tokens=1067),
    )
    assert any("text_tokens" in v for v in res.violations)
    assert (
        _prompt_identity(
            monkeypatch,
            _img_cf("yunshu", 1090, image_sha256="a" * 64, text_tokens=1066),
            _img_cf("omlx-0.7.3", 1095, image_sha256="a" * 64, text_tokens=1066),
        ).violations
        == []
    )


def test_a_text_cell_still_compares_the_token_count(monkeypatch):
    def text(engine, pt):
        cf = _img_cf(engine, pt)
        cf.rows[0].pop("media")
        cf.rows[0]["image_parts"] = 0
        return cf

    # the same request (same sha) rendered by another chat template: a per-engine token-count note, not a violation
    res = _prompt_identity(
        monkeypatch, text("yunshu", 1024), text("kapitsa-0.2.26", 1025)
    )
    assert res.violations == [] and any("template delta" in n for n in res.notes)
    # a different request is still a violation
    other = text("kapitsa-0.2.26", 1024)
    other.rows[0]["prompt_ids_sha"] = "f" * 64
    res = _prompt_identity(monkeypatch, text("yunshu", 1024), other)
    assert res.violations and "different requests" in res.violations[0]


def test_the_image_rows_record_the_image_sha_and_text_tokens():
    import hashlib

    assert (
        hashlib.sha256(__import__("base64").b64decode(tb.IMAGE_PNG_B64)).hexdigest()
        == tb.IMAGE_PNG_SHA256
    )
    assert len(tb.IMAGE_PNG_SHA256) == 64


# --- cold_cache on conc / mixed allows the chat template's shared header (wire-1108c8 TensorFold: cached=3) ----------


def _mixed_cf(engine, cached_list, phase="mixed", cell=None):
    cell = cell or f"B2.mixed.{engine}.8K.c4"
    rows = [
        {"part": phase, "phase": phase, "cached": c, "pt": 1024, "trial": 0}
        for c in cached_list
    ]
    return fa.CellFile(
        {"cell": cell, "engine": engine, "group": "g"},
        0,
        Path("x"),
        {"part": "session", "fresh_process": True},
        rows,
    )


def test_the_template_prefix_is_computed_from_the_models_chat_template():
    try:
        import snapshot_prompts as sp

        n = sp.template_prefix_tokens()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local tokenizer: {exc}")
    assert (
        n == 3
    )  # '<|im_start|>', 'user', '\n': the header every Qwen chat prompt starts with


def test_a_mixed_request_reporting_the_template_header_as_cached_passes_and_one_above_it_fails(
    monkeypatch,
):
    """Failing-before (wire-1108c8 TensorFold mixed 8K c4): cached=3 on fresh prompts, the shared template header before the
    salted content, failed 'must be server-reported 0'."""
    monkeypatch.setattr(fa, "template_prefix_allowance", lambda: 3)
    ok = fa.check_cold_cache(
        fa.Run(
            manifest={},
            files=[_mixed_cf("tf-1.2.1", [3, 3, 0, 3])],
            missing=[],
            extra=[],
            parse_errors=[],
        )
    )
    assert ok.violations == [], ok.violations
    bad = fa.check_cold_cache(
        fa.Run(
            manifest={},
            files=[_mixed_cf("tf-1.2.1", [3, 4])],
            missing=[],
            extra=[],
            parse_errors=[],
        )
    )
    assert (
        bad.violations
        and "cached=4" in bad.violations[0]
        and "3-token header" in bad.violations[0]
    )
    big = fa.check_cold_cache(
        fa.Run(
            manifest={},
            files=[_mixed_cf("yunshu", [8192])],
            missing=[],
            extra=[],
            parse_errors=[],
        )
    )
    assert big.violations  # a real prefix-cache hit still fails


def test_without_a_tokenizer_the_guard_stays_strict(monkeypatch):
    monkeypatch.setattr(fa, "template_prefix_allowance", lambda: 0)
    res = fa.check_cold_cache(
        fa.Run(
            manifest={},
            files=[_mixed_cf("tf-1.2.1", [3])],
            missing=[],
            extra=[],
            parse_errors=[],
        )
    )
    assert res.violations and "must be server-reported 0" in res.violations[0]


def test_the_allowance_function_falls_back_to_zero_when_the_tokenizer_cannot_load(
    monkeypatch,
):
    import snapshot_prompts as sp

    def boom(*a, **k):
        raise OSError("no tokenizer")

    monkeypatch.setattr(sp, "template_prefix_tokens", boom)
    assert fa.template_prefix_allowance() == 0
