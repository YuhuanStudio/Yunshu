"""CPU tests of the modality route checks (scripts/research/route_checks_media.py): the pure helpers,
and every check run against a canned *good* server (httpx MockTransport), so the check's own
parsing, request shapes and SDK calls are proven before an M3 job. A canned-bad answer must fail."""

from __future__ import annotations

import base64
import io
import json
import math
import struct
import sys
import wave
from pathlib import Path

import httpx
import openai
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

import route_checks as rc  # noqa: E402
import route_checks_media as m  # noqa: E402


def make_wav(seconds=3.0, rate=24000, amp=8000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(amp * math.sin(i / 20)))
                for i in range(int(seconds * rate))
            )
        )
    return buf.getvalue()


def make_png(size=256, color=(200, 30, 30)):
    from PIL import Image

    im = Image.new("RGB", (size, size), color)
    for i in range(size):
        im.putpixel((i, i), (0, 0, 255))
        im.putpixel((i, size - 1 - i), (0, 255, 0))
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def test_helpers():
    rate, secs, peak = m.wav_info(make_wav(2.0))
    assert rate == 24000 and abs(secs - 2.0) < 0.01 and peak > 7000
    with pytest.raises(rc.Fail):
        m.wav_info(b"ID3....")
    assert m.png_size(make_png(128)) == (128, 128) and m.png_not_blank(make_png())
    with pytest.raises(rc.Fail):
        m.png_size(b"nope" * 8)
    assert abs(m.cosine([1, 0], [1, 1]) - 1 / math.sqrt(2)) < 1e-9
    assert m.loose_match("Hello world. This is a test.", "hello world this is the test")
    assert not m.loose_match("Hello world. This is a test.", "completely different")
    png = m.render_text_png("Yunshu 42")
    assert m.png_size(png)[0] > 100
    ev = m.sse_events('data: {"a": 1}\n\ndata: [DONE]\n\n')
    assert ev == [{"a": 1}, "[DONE]"]


def vec(i, dim=512):
    v = [math.sin(i * 7 + k) for k in range(dim)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


TEXT_VEC = {
    m.CAT: vec(1),
    m.KITTEN: _unit([0.9 * a + 0.1 * b for a, b in zip(vec(1), vec(2), strict=True)]),
    m.STOCK: vec(5),
    **{t: vec(7 + i) for i, t in enumerate(m.MIXED) if t not in (m.CAT, m.KITTEN, m.STOCK)},
}


def good_server(request: httpx.Request) -> httpx.Response:
    p, body = request.url.path, request.content
    ctype = request.headers.get("content-type", "")
    j = json.loads(body) if "json" in ctype and body else {}

    def js(o, code=200):
        return httpx.Response(code, json=o)

    def err(code, msg):
        return httpx.Response(
            code, json={"error": {"message": msg, "type": "invalid_request_error"}}
        )

    if p == "/v1/audio/voices":
        return js({"voices": ["alloy"]})
    if p == "/v1/audio/speech":
        if not j.get("input"):
            return err(400, "input: field is required")
        if j.get("response_format") == "mp3":
            return httpx.Response(
                200,
                content=b"ID3\x04" + b"\0" * 50,
                headers={"content-type": "audio/mpeg"},
            )
        return httpx.Response(
            200, content=make_wav(3.2), headers={"content-type": "audio/wav"}
        )
    if p == "/v1/audio/speech/stream":
        pcm = make_wav(3.0)[44:]
        sse = (
            "data: "
            + json.dumps({"type": "header", "wav_header": "AA==", "sample_rate": 24000})
            + "\n\n"
            + "data: "
            + json.dumps(
                {"type": "audio", "audio": base64.b64encode(pcm).decode(), "segment": 0}
            )
            + "\n\n"
            + 'data: {"type": "done"}\n\ndata: [DONE]\n\n'
        )
        return httpx.Response(
            200, content=sse, headers={"content-type": "text/event-stream"}
        )
    if p in ("/v1/audio/transcriptions", "/v1/audio/translations"):
        if b"not audio" in body:
            return err(400, "could not decode audio")
        if p.endswith("translations"):
            return err(501, "translation needs a Whisper model")
        txt = m.SPEECH_TEXT
        if b"srt" in body:
            return httpx.Response(200, text="1\n00:00:00,000 --> 00:00:03,000\n" + txt)
        if b"vtt" in body:
            return httpx.Response(200, text="WEBVTT\n\n00:00.000 --> 00:03.000\n" + txt)
        if b'name="response_format"\r\n\r\ntext' in body:
            return httpx.Response(200, text=txt)
        if b"verbose_json" in body:
            return js({"text": txt, "language": "en", "duration": 3.2, "segments": []})
        return js({"text": txt})
    if p == "/v1/ocr":
        if b"not an image" in body:
            return err(400, "cannot read image")
        return js(
            {
                "text": "Yunshu reads 42 words",
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 8,
                    "total_tokens": 58,
                },
            }
        )
    if p == "/v1/images/generations/stream":
        return httpx.Response(
            200,
            content="data: "
            + json.dumps(
                {
                    "step": 2,
                    "progress": 1.0,
                    "image": base64.b64encode(make_png()).decode(),
                    "is_final": True,
                }
            )
            + "\n\ndata: [DONE]\n\n",
        )
    if p in ("/v1/images/generations", "/v1/images/edits", "/v1/images/variations"):
        if j and j.get("size") == "100x100":
            return err(400, "dimensions must be multiples of 64")
        color = (10, 200, 10) if p.endswith("edits") else (200, 30, 30)
        return js(
            {
                "created": 1,
                "data": [
                    {"b64_json": base64.b64encode(make_png(color=color)).decode()}
                ],
            }
        )
    if p == "/v1/embeddings":
        inp = j.get("input")
        if inp == "":
            return err(400, "input must not be empty")
        items = inp if isinstance(inp, list) else [inp]
        dim = j.get("dimensions")
        out = []
        for i, t in enumerate(items):
            v = TEXT_VEC[t]
            if dim:
                v = v[:dim]
            if j.get("encoding_format") == "base64":
                e = base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode()
            else:
                e = v
            out.append({"object": "embedding", "index": i, "embedding": e})
        return js(
            {
                "object": "list",
                "data": out,
                "model": "m",
                "usage": {
                    "prompt_tokens": 5 * len(items),
                    "total_tokens": 5 * len(items),
                },
            }
        )
    if p == "/v1/pooling":
        return js(
            {
                "object": "list",
                "data": [
                    {"object": "pooling", "index": i, "data": TEXT_VEC[t]}
                    for i, t in enumerate(j["input"])
                ],
            }
        )
    if p == "/v1/score":
        return js(
            {
                "object": "list",
                "data": [
                    {
                        "object": "score",
                        "index": i,
                        "score": m.cosine(TEXT_VEC[j["text_1"]], TEXT_VEC[t]),
                    }
                    for i, t in enumerate(j["text_2"])
                ],
            }
        )
    if p == "/v1/rerank":
        sc = sorted(
            (
                (m.cosine(TEXT_VEC[m.CAT], TEXT_VEC[d]) if d in TEXT_VEC else -0.5, i)
                for i, d in enumerate(j["documents"])
            ),
            reverse=True,
        )
        return js(
            {
                "object": "list",
                "results": [{"index": i, "relevance_score": s} for s, i in sc][
                    : j.get("top_n", 99)
                ],
            }
        )
    if p == "/v1/classify":
        want = next(w for t, _, w in m.CLASSIFY_CASES if t == j["input"])
        order = [want] + [x for x in j["labels"] if x != want]
        return js(
            {
                "model": "m",
                "results": [
                    {"label": x, "score": sc, "index": j["labels"].index(x)}
                    for x, sc in zip(order, (0.8, 0.15, 0.05), strict=False)
                ],
            }
        )
    if p == "/api/embed":
        return js({"embeddings": [TEXT_VEC[t] for t in j["input"]]})
    if p == "/api/embeddings":
        return js({"embedding": TEXT_VEC[j["prompt"]]})
    return err(404, f"no route {p}")


def ctx_for(needs, handler=good_server, shared=None, ref=True):
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://srv")
    return rc.Ctx(
        url="http://srv", token="", model="m", kind=needs, http=http,
        oa=openai.OpenAI(base_url="http://srv/v1", api_key="x", http_client=http, max_retries=0),
        an=None, shared=shared if shared is not None else {},
        fixtures={"embed_reference": lambda ts: [TEXT_VEC[t] for t in ts]} if ref else {},
    )  # fmt: skip


@pytest.mark.parametrize("needs", ["tts", "asr", "ocr", "image", "embed"])
def test_every_media_check_passes_against_a_good_server(needs):
    shared = {}
    if needs == "asr":
        shared["tts_wav"] = make_wav(3.2)
    for chk in rc.REGISTRY.values():
        if chk.needs == needs:
            chk.fn(ctx_for(needs, shared=shared))


def test_tts_output_is_kept_for_asr():
    ctx = ctx_for("tts")
    rc.REGISTRY["tts_served"].fn(ctx)
    assert m.wav_info(ctx.shared["tts_wav"])[1] > 3


def test_bad_answers_fail_the_checks():
    def silent(req):
        if req.url.path == "/v1/audio/speech" and b"mp3" not in req.content:
            return httpx.Response(
                200, content=make_wav(3, amp=0), headers={"content-type": "audio/wav"}
            )
        return good_server(req)

    with pytest.raises(rc.Fail, match="silent"):
        rc.REGISTRY["tts_served"].fn(ctx_for("tts", silent))

    def wrong_order(req):
        r = good_server(req)
        if req.url.path == "/v1/rerank":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 0, "relevance_score": 0.9},
                        {"index": 1, "relevance_score": 0.1},
                    ]
                },
            )
        return r

    with pytest.raises(rc.Fail, match="rerank"):
        rc.REGISTRY["embed_served"].fn(ctx_for("embed", wrong_order))

    def wrong_size(req):
        if req.url.path == "/v1/images/generations":
            return httpx.Response(
                200,
                json={"data": [{"b64_json": base64.b64encode(make_png(128)).decode()}]},
            )
        return good_server(req)

    with pytest.raises(rc.Fail, match="size"):
        rc.REGISTRY["image_served"].fn(ctx_for("image", wrong_size))

    def bad_text(req):
        if req.url.path == "/v1/ocr":
            return httpx.Response(
                200, json={"text": "garbage", "usage": {"prompt_tokens": 1}}
            )
        return good_server(req)

    with pytest.raises(rc.Fail, match="ocr text"):
        rc.REGISTRY["ocr_served"].fn(ctx_for("ocr", bad_text))

    def bad_asr(req):
        if req.url.path == "/v1/audio/transcriptions":
            return httpx.Response(200, json={"text": "something else entirely"})
        return good_server(req)

    with pytest.raises(rc.Fail, match="transcript"):
        rc.REGISTRY["asr_served"].fn(ctx_for("asr", bad_asr, {"tts_wav": make_wav()}))


def test_embed_check_fails_when_served_vectors_disagree_with_the_reference():
    """The broken-weights case: plausible unit vectors, wrong space. Similarity order alone passed it."""

    def scrambled(req):
        r = good_server(req)
        if req.url.path == "/v1/embeddings" and req.content != b"":
            j = json.loads(req.content)
            if j["input"] != "" and j.get("encoding_format") != "base64" and not j.get("dimensions"):
                items = j["input"] if isinstance(j["input"], list) else [j["input"]]
                data = [
                    {"object": "embedding", "index": i, "embedding": vec(50 + len(t))}
                    for i, t in enumerate(items)
                ]
                return httpx.Response(
                    200,
                    json={"object": "list", "data": data, "model": "m",
                          "usage": {"prompt_tokens": 5, "total_tokens": 5}},
                )  # fmt: skip
        return r

    with pytest.raises(rc.Fail, match="differ from the reference"):
        rc.REGISTRY["embed_served"].fn(ctx_for("embed", scrambled))


def test_embed_check_needs_a_reference():
    with pytest.raises(rc.Fail, match="no reference"):
        rc.REGISTRY["embed_served"].fn(ctx_for("embed", ref=False))


def test_classify_cases_are_clear_and_each_label_set_has_the_answer():
    assert len(m.CLASSIFY_CASES) >= 3
    for _, labels, want in m.CLASSIFY_CASES:
        assert want in labels and len(set(labels)) == len(labels)
