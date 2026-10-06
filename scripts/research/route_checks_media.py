"""Served-path checks of the route registry for the modalities (imported at the bottom of
route_checks.py): speech, transcription, OCR, image generation, embeddings. Each runs against its
own real server (`needs` = tts / asr / ocr / image / embed); TTS output is kept in `ctx.shared` and
round-tripped through ASR. The helpers at the top are pure and unit-tested on CPU."""

from __future__ import annotations

import base64
import io
import json
import math
import re
import struct
import wave

from route_checks import Ctx, Fail, check, err_ok, expect

SPEECH_TEXT = "Hello world. This is a test of the speech system."
ZH_TEXT = "今天天气很好，我们一起去公园散步吧。"
ZH_WORDS = (
    "weather",
    "park",
    "walk",
    "today",
    "nice",
    "good",
    "beautiful",
    "sunny",
    "stroll",
)

# ── pure helpers ─────────────────────────────────────────────────────────────────────────


def wav_info(data: bytes) -> tuple[int, float, int]:
    """(sample_rate, seconds, peak amplitude 0..32767) of a PCM16 WAV; raises Fail if invalid."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise Fail(f"not a WAV (starts {data[:12]!r})")
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate, n, ch, sw = (
                w.getframerate(),
                w.getnframes(),
                w.getnchannels(),
                w.getsampwidth(),
            )
            raw = w.readframes(n)
    except (wave.Error, EOFError) as e:
        raise Fail(f"unreadable WAV: {e}") from None
    if sw != 2:
        raise Fail(f"sample width {sw}, want 16-bit")
    samples = struct.unpack(f"<{len(raw) // 2}h", raw[: len(raw) // 2 * 2])
    return rate, n / (rate * ch), max((abs(s) for s in samples), default=0)


def pcm_seconds(pcm: bytes, rate: int) -> float:
    return len(pcm) / 2 / rate


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise Fail(f"not a PNG (starts {data[:8]!r})")
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def png_not_blank(data: bytes) -> bool:
    """More than one colour in the image (needs PIL; True when PIL is missing)."""
    try:
        from PIL import Image
    except ImportError:
        return True
    im = Image.open(io.BytesIO(data)).convert("RGB").resize((32, 32))
    return len(set(im.getdata())) > 4


def cosine(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def loose_match(expected: str, got: str, need: float = 0.6) -> bool:
    """At least `need` of the expected words appear in what came back."""
    want = words(expected)
    have = set(words(got))
    return bool(want) and sum(w in have for w in want) / len(want) >= need


def render_text_png(text: str, size: int = 56) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        font = ImageFont.load_default(size=size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    box = probe.textbbox((0, 0), text, font=font)
    im = Image.new("RGB", (box[2] + 60, box[3] + 60), "white")
    ImageDraw.Draw(im).text((30, 30), text, fill="black", font=font)
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def sse_events(text: str) -> list:
    out = []
    for block in text.split("\n\n"):
        for ln in block.splitlines():
            if ln.startswith("data:"):
                d = ln[5:].strip()
                out.append(d if d == "[DONE]" else json.loads(d))
    return out


# ── TTS ──────────────────────────────────────────────────────────────────────────────────


def _speech_body(c: Ctx, **kw):
    return {
        "model": c.model,
        "input": SPEECH_TEXT,
        "voice": "alloy",
        "instruct": "A calm adult female voice.",
        "response_format": "wav",
        **kw,
    }


@check(
    "tts_served",
    "POST /v1/audio/speech",
    "POST /v1/audio/speech/stream",
    "GET /v1/audio/voices",
    needs="tts",
    served=True,
)
def _tts(c: Ctx):
    v = c.req("GET", "/v1/audio/voices")
    expect(v.status_code == 200 and v.json(), f"voices {v.status_code} {v.text[:100]}")
    r = c.req("POST", "/v1/audio/speech", json=_speech_body(c), timeout=600)
    expect(r.status_code == 200, f"speech {r.status_code} {r.text[:200]}")
    expect(
        "audio" in r.headers.get("content-type", ""),
        f"content-type {r.headers.get('content-type')}",
    )
    rate, secs, peak = wav_info(r.content)
    expect(rate >= 16000, f"sample rate {rate}")
    expect(1.0 <= secs <= 20.0, f"duration {secs:.1f}s for a 9-word sentence")
    expect(peak > 1000, f"audio is silent (peak {peak})")
    c.shared["tts_wav"] = r.content
    zh = c.req(
        "POST",
        "/v1/audio/speech",
        json={**_speech_body(c), "input": ZH_TEXT, "response_format": "wav"},
        timeout=600,
    )
    expect(zh.status_code == 200, f"chinese speech {zh.status_code} {zh.text[:200]}")
    _, zsecs, zpeak = wav_info(zh.content)
    expect(zsecs >= 1.0 and zpeak > 1000, f"chinese speech {zsecs:.1f}s peak {zpeak}")
    c.shared["tts_wav_zh"] = zh.content
    c.notes["tts"] = f"{secs:.1f}s @ {rate} Hz, peak {peak}"
    # the OpenAI SDK path (typed client, binary body)
    s = c.oa.audio.speech.create(
        model=c.model,
        voice="alloy",
        input="Good morning.",
        response_format="wav",
        extra_body={"instruct": "A calm adult female voice."},
    )
    expect(wav_info(s.content)[1] > 0.3, "SDK speech too short")
    # default format of the SDK is mp3: must be a real mp3 or an honest error, never mislabelled WAV
    m = c.req(
        "POST",
        "/v1/audio/speech",
        json=_speech_body(c, response_format="mp3", input="Hi there."),
        timeout=600,
    )
    if m.status_code == 200:
        head = m.content[:3]
        if head == b"ID3" or m.content[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            expect(
                "mpeg" in m.headers.get("content-type", ""), "mp3 bytes labelled other"
            )
            c.notes["tts_mp3"] = "mp3"
        else:
            # no ffmpeg on the machine: honest WAV, labelled as WAV and flagged as a fallback
            wav_info(m.content)
            expect(
                "wav" in m.headers.get("content-type", "")
                and m.headers.get("x-yunshu-audio-format-fallback") == "wav",
                f"WAV bytes for an mp3 request, headers {dict(m.headers)}",
            )
            c.notes["tts_mp3"] = "wav fallback (no ffmpeg), labelled"
    else:
        err_ok(m, "openai")
        c.notes["tts_mp3"] = f"{m.status_code}: {m.text[:100]}"
    # streaming: header event, audio chunks, done; the PCM length matches the one-shot answer
    st = c.req("POST", "/v1/audio/speech/stream", json=_speech_body(c), timeout=600)
    expect(st.status_code == 200, f"stream {st.status_code} {st.text[:200]}")
    ev = sse_events(st.text)
    expect(ev and ev[-1] == "[DONE]", "no [DONE]")
    hdr = [e for e in ev if isinstance(e, dict) and e.get("type") == "header"]
    audio = [e for e in ev if isinstance(e, dict) and e.get("type") == "audio"]
    expect(
        len(hdr) == 1 and audio,
        f"events {[e.get('type') if isinstance(e, dict) else e for e in ev][:6]}",
    )
    expect(
        not [
            e
            for e in ev
            if isinstance(e, dict) and (e.get("type") == "error" or "error" in e)
        ],
        "error event in the stream",
    )
    pcm = b"".join(base64.b64decode(e["audio"]) for e in audio)
    ssecs = pcm_seconds(pcm, hdr[0]["sample_rate"])
    expect(1.0 <= ssecs <= 20.0, f"streamed duration {ssecs:.1f}s")
    expect(
        max(
            abs(x)
            for x in struct.unpack(f"<{len(pcm) // 2}h", pcm[: len(pcm) // 2 * 2])
        )
        > 1000,
        "streamed audio silent",
    )
    c.notes["tts_stream"] = f"{ssecs:.1f}s in {len(audio)} chunks"
    bad = c.req(
        "POST",
        "/v1/audio/speech",
        json={"model": c.model, "input": "", "voice": "alloy"},
    )
    err_ok(bad, "openai")
    expect(bad.status_code == 400, f"empty input -> {bad.status_code}")


# ── ASR ──────────────────────────────────────────────────────────────────────────────────


@check(
    "asr_served",
    "POST /v1/audio/transcriptions",
    needs="asr",
    served=True,
)
def _asr(c: Ctx):
    wav = c.shared.get("tts_wav")
    expect(wav, "needs the TTS output (the tts server runs first in the same job)")
    t = c.oa.audio.transcriptions.create(
        model=c.model, file=("speech.wav", wav, "audio/wav")
    )
    expect(
        loose_match(SPEECH_TEXT, t.text), f"transcript {t.text!r} vs {SPEECH_TEXT!r}"
    )
    c.notes["asr_text"] = t.text
    v = c.oa.audio.transcriptions.create(
        model=c.model,
        file=("speech.wav", wav, "audio/wav"),
        response_format="verbose_json",
    )
    expect(v.text and v.duration and v.duration > 1.0, f"verbose_json {v}")
    for fmt in ("text", "srt", "vtt"):
        r = c.req(
            "POST",
            "/v1/audio/transcriptions",
            data={"model": c.model, "response_format": fmt},
            files={"file": ("s.wav", wav, "audio/wav")},
            timeout=600,
        )
        expect(
            r.status_code == 200 and r.text.strip(),
            f"{fmt}: {r.status_code} {r.text[:100]}",
        )
        if fmt == "vtt":
            expect(r.text.startswith("WEBVTT"), f"vtt starts {r.text[:20]!r}")
    bad = c.req(
        "POST",
        "/v1/audio/transcriptions",
        data={"model": c.model},
        files={"file": ("s.wav", b"not audio", "audio/wav")},
        timeout=120,
    )
    err_ok(bad, "openai")
    expect(bad.status_code in (400, 415, 422), f"garbage audio -> {bad.status_code}")


# ── OCR ──────────────────────────────────────────────────────────────────────────────────


@check("ocr_served", "POST /v1/ocr", needs="ocr", served=True)
def _ocr(c: Ctx):
    text = "Yunshu reads 42 words"
    png = render_text_png(text)
    r = c.req(
        "POST",
        "/v1/ocr",
        data={"model": c.model},
        files={"file": ("t.png", png, "image/png")},
        timeout=600,
    )
    expect(r.status_code == 200, f"ocr {r.status_code} {r.text[:200]}")
    j = r.json()
    expect(
        loose_match(text, j.get("text", ""), 0.8),
        f"ocr text {j.get('text')!r} vs {text!r}",
    )
    expect(j.get("usage", {}).get("prompt_tokens", 0) > 0, f"usage {j.get('usage')}")
    c.notes["ocr_text"] = j.get("text", "")[:80]
    bad = c.req(
        "POST",
        "/v1/ocr",
        data={"model": c.model},
        files={"file": ("t.png", b"not an image", "image/png")},
        timeout=120,
    )
    expect(
        bad.status_code in (400, 415, 422, 500) and bad.status_code != 500,
        f"garbage image -> {bad.status_code} {bad.text[:100]}",
    )
    err_ok(bad, "openai")


# ── image generation ─────────────────────────────────────────────────────────────────────

IMG = {"size": "256x256"}
FAST = {"num_inference_steps": 2, "seed": 7}


@check(
    "image_served",
    "POST /v1/images/generations",
    "POST /v1/images/generations/stream",
    "POST /v1/images/variations",
    "POST /v1/images/edits",
    needs="image",
    served=True,
)
def _image(c: Ctx):
    g = c.oa.images.generate(
        model=c.model,
        prompt="a red apple on a table",
        n=1,
        response_format="b64_json",
        extra_body=FAST,
        **IMG,
    )
    png = base64.b64decode(g.data[0].b64_json)
    expect(png_size(png) == (256, 256), f"size {png_size(png)}")
    expect(png_not_blank(png), "blank image")
    g2 = c.oa.images.generate(
        model=c.model,
        prompt="a red apple on a table",
        n=1,
        response_format="b64_json",
        extra_body=FAST,
        **IMG,
    )
    expect(base64.b64decode(g2.data[0].b64_json) == png, "same seed, different image")
    st = c.req(
        "POST",
        "/v1/images/generations/stream",
        json={"model": c.model, "prompt": "a blue cube", **IMG, **FAST},
        timeout=900,
    )
    expect(st.status_code == 200, f"stream {st.status_code} {st.text[:200]}")
    ev = [e for e in sse_events(st.text) if isinstance(e, dict)]
    fin = [e for e in ev if e.get("is_final")]
    expect(fin and not [e for e in ev if "error" in e], f"stream events {ev[-2:]}")
    expect(
        png_size(base64.b64decode(fin[-1]["image"])) == (256, 256), "stream final size"
    )
    e = c.oa.images.edit(
        model=c.model,
        image=("in.png", png, "image/png"),
        prompt="make it green",
        response_format="b64_json",
        extra_body=FAST,
        **IMG,
    )
    ep = base64.b64decode(e.data[0].b64_json)
    expect(png_size(ep) == (256, 256) and ep != png, "edit result")
    v = c.oa.images.create_variation(
        model=c.model,
        image=("in.png", png, "image/png"),
        n=1,
        response_format="b64_json",
        extra_body=FAST,
        **IMG,
    )
    vp = base64.b64decode(v.data[0].b64_json)
    expect(png_size(vp) == (256, 256), "variation size")
    bad = c.req(
        "POST",
        "/v1/images/generations",
        json={"model": c.model, "prompt": "x", "size": "100x100"},
    )
    err_ok(bad, "openai")
    expect(bad.status_code == 400, f"bad size -> {bad.status_code}")


# ── embeddings ───────────────────────────────────────────────────────────────────────────

CAT = "A cat sat on the warm windowsill."
KITTEN = "A small kitten rests on the sunny window ledge."
FIN = "financial news about interest rates and markets"
STOCK = "Quarterly earnings beat expectations as the stock market rallied."


@check(
    "embed_served",
    "POST /v1/embeddings",
    "POST /v1/rerank",
    "POST /v1/score",
    "POST /v1/pooling",
    "POST /v1/classify",
    "POST /api/embed",
    "POST /api/embeddings",
    needs="embed",
    served=True,
)
def _embed(c: Ctx):
    one = [
        c.oa.embeddings.create(model=c.model, input=t).data[0].embedding
        for t in (CAT, KITTEN, STOCK)
    ]
    dim = len(one[0])
    expect(dim >= 256, f"dimension {dim}")
    expect(
        all(abs(math.sqrt(sum(x * x for x in v)) - 1) < 1e-3 for v in one),
        "not unit length",
    )
    expect(
        cosine(one[0], one[1]) > cosine(one[0], one[2]) + 0.05,
        f"similar {cosine(one[0], one[1]):.3f} vs unrelated {cosine(one[0], one[2]):.3f}",
    )
    batch = c.oa.embeddings.create(model=c.model, input=[CAT, KITTEN, STOCK])
    expect([d.index for d in batch.data] == [0, 1, 2], "batch order")
    expect(
        all(
            cosine(b.embedding, s) > 0.999 for b, s in zip(batch.data, one, strict=True)
        ),
        "batch differs from the singles",
    )
    expect(batch.usage.prompt_tokens > 0, "usage")
    c.notes["embed_dim"] = dim
    d = (
        c.oa.embeddings.create(model=c.model, input=CAT, dimensions=256)
        .data[0]
        .embedding
    )
    expect(len(d) == 256 and cosine(d, one[0][:256]) > 0.999, "dimensions=256")
    b64 = (
        c.oa.embeddings.create(model=c.model, input=CAT, encoding_format="base64")
        .data[0]
        .embedding
    )
    raw = base64.b64decode(b64) if isinstance(b64, str) else b""
    vec = struct.unpack(f"<{len(raw) // 4}f", raw) if raw else ()
    expect(len(vec) == dim and cosine(vec, one[0]) > 0.999, "base64 differs from float")
    bad = c.req("POST", "/v1/embeddings", json={"model": c.model, "input": ""})
    err_ok(bad, "openai")
    expect(bad.status_code == 400, f"empty -> {bad.status_code}")
    # pooling: vectors of the model's width
    p = c.req("POST", "/v1/pooling", json={"model": c.model, "input": [CAT, STOCK]})
    expect(
        p.status_code == 200
        and len(p.json()["data"]) == 2
        and len(p.json()["data"][0]["data"]) == dim,
        f"pooling {p.status_code} {p.text[:150]}",
    )
    # score: similar pair above the unrelated pair
    s = c.req(
        "POST",
        "/v1/score",
        json={"model": c.model, "text_1": CAT, "text_2": [KITTEN, STOCK]},
    )
    expect(s.status_code == 200, f"score {s.status_code} {s.text[:150]}")
    sc = [x["score"] for x in s.json()["data"]]
    expect(sc[0] > sc[1], f"score {sc}")
    # rerank: the relevant document first
    docs = [STOCK, KITTEN, "Rain is expected tomorrow."]
    r = c.req(
        "POST",
        "/v1/rerank",
        json={"model": c.model, "query": CAT, "documents": docs, "top_n": 2},
    )
    expect(r.status_code == 200, f"rerank {r.status_code} {r.text[:150]}")
    res = r.json()["results"]
    expect(
        res[0]["index"] == 1
        and len(res) == 2
        and res[0]["relevance_score"] >= res[1]["relevance_score"],
        f"rerank {res}",
    )
    # classify: the right label on top, probabilities sum to 1
    k = c.req(
        "POST",
        "/v1/classify",
        json={
            "model": c.model,
            "input": "The central bank raised interest rates and bond yields climbed.",
            "labels": [FIN, "a football match report", "a cooking recipe"],
        },
    )
    expect(k.status_code == 200, f"classify {k.status_code} {k.text[:150]}")
    kr = k.json()["results"]
    expect(
        kr[0]["label"] == FIN and abs(sum(x["score"] for x in kr) - 1) < 1e-3,
        f"classify {kr}",
    )
    # Ollama
    o = c.req(
        "POST", "/api/embed", json={"model": c.model, "input": [CAT, KITTEN, STOCK]}
    )
    expect(o.status_code == 200, f"/api/embed {o.status_code} {o.text[:150]}")
    oe = o.json()["embeddings"]
    expect(
        len(oe) == 3 and len(oe[0]) == dim and cosine(oe[0], one[0]) > 0.999,
        "/api/embed vs /v1/embeddings",
    )
    expect(cosine(oe[0], oe[1]) > cosine(oe[0], oe[2]), "/api/embed similarity order")
    ol = c.req("POST", "/api/embeddings", json={"model": c.model, "prompt": CAT})
    expect(
        ol.status_code == 200 and cosine(ol.json()["embedding"], one[0]) > 0.999,
        f"/api/embeddings {ol.status_code}",
    )


@check(
    "asr_translations_error", "POST /v1/audio/translations", needs="asr", served=False
)
def _asr_translations(c: Ctx):
    """Error path: only Whisper-family models translate, Qwen3-ASR answers a documented 501."""
    wav = c.shared.get("tts_wav")
    expect(wav, "needs the TTS output")
    tr = c.req(
        "POST",
        "/v1/audio/translations",
        data={"model": c.model},
        files={"file": ("s.wav", wav, "audio/wav")},
        timeout=600,
    )
    if tr.status_code == 200:
        expect(tr.json().get("text"), "translation without text")
        c.notes["asr_translations"] = "served"
    else:
        err_ok(tr, "openai")
        expect(
            tr.status_code == 501, f"translations -> {tr.status_code} {tr.text[:100]}"
        )  # non-Whisper: documented 501
        c.notes["asr_translations"] = f"501: {tr.text[:100]}"


def cjk_ratio(text: str) -> float:
    """Share of CJK / kana / hangul characters among the letters of `text` (0 for no letters)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ord(ch) > 0x2E80) / len(letters)


@check(
    "whisper_translations",
    "POST /v1/audio/translations",
    needs="translate",
    served=True,
)
def _whisper_translations(c: Ctx):
    """Chinese speech (made by Qwen3-TTS) in, English text out, judged loosely: non-empty, mostly
    Latin letters, and at least one word of what was said (weather / park / walk ...)."""
    wav = c.shared.get("tts_wav_zh")
    expect(wav, "needs the Chinese TTS output of the tts check")
    r = c.req(
        "POST",
        "/v1/audio/translations",
        data={"model": c.model},
        files={"file": ("zh.wav", wav, "audio/wav")},
        timeout=900,
    )
    expect(r.status_code == 200, f"translations {r.status_code} {r.text[:200]}")
    text = (r.json().get("text") or "").strip()
    c.notes["translation_json"] = text[:160]
    expect(text, "translation without text")
    expect(cjk_ratio(text) < 0.2, f"not English: {text!r}")
    expect(
        any(w in text.lower() for w in ZH_WORDS),
        f"translation misses the content: {text!r}",
    )
    # the SDK's typed client, and the text response format
    t = c.oa.audio.translations.create(model=c.model, file=("zh.wav", wav, "audio/wav"))
    expect(t.text.strip() and cjk_ratio(t.text) < 0.2, f"sdk translation {t.text!r}")
    p = c.req(
        "POST",
        "/v1/audio/translations",
        data={"model": c.model, "response_format": "text"},
        files={"file": ("zh.wav", wav, "audio/wav")},
        timeout=900,
    )
    expect(
        p.status_code == 200 and p.text.strip(),
        f"text format {p.status_code} {p.text[:100]}",
    )
