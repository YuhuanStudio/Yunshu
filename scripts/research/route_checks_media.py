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
LONG = (
    "The committee met for three days to review the harbour expansion plan, weighing dredging costs, "
    "the effect on migratory birds, ferry timetables, flood defences, and a proposed cycle path along "
    "the old rail embankment. After hearing from residents, engineers and the fishing cooperative, "
    "members agreed to a phased schedule, starting with the breakwater and leaving the marina for later."
)
MIXED = [CAT, "Hi", STOCK, LONG, KITTEN]  # one batch of very different lengths
REF_COS = 0.999  # served vs official-recipe reference, per text

# (input, labels, the label that must win): realistic labels, clearly separable topics
CLASSIFY_CASES = [
    (
        "The central bank raised interest rates and bond yields climbed.",
        [FIN, "a football match report", "a cooking recipe"],
        FIN,
    ),
    (
        "The striker scored twice in the second half to win the derby 3-1.",
        ["a football match report", FIN, "a cooking recipe"],
        "a football match report",
    ),
    (
        "Simmer the tomatoes with garlic and basil for twenty minutes, then stir in the cream.",
        [FIN, "a football match report", "a cooking recipe"],
        "a cooking recipe",
    ),
    (
        "The new GPU compiler fuses small kernels and cuts memory traffic by a third.",
        ["a cooking recipe", "software and hardware engineering", "a travel diary"],
        "software and hardware engineering",
    ),
]


def reference_gaps(got, ref, floor=REF_COS):
    """Indexes (with cosine) of served vectors that differ from the reference by less than `floor`."""
    return [
        (i, round(cosine(g, r), 4))
        for i, (g, r) in enumerate(zip(got, ref, strict=True))
        if cosine(g, r) < floor
    ]


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
    ref_fn = c.fixtures.get("embed_reference")
    expect(ref_fn, "no reference embedder in this run (fixtures['embed_reference'])")
    ref = ref_fn(MIXED)
    singles = [
        c.oa.embeddings.create(model=c.model, input=t).data[0].embedding for t in MIXED
    ]
    gaps = reference_gaps(singles, ref)
    expect(not gaps, f"single embeddings differ from the reference (idx, cos): {gaps}")
    mixed = c.oa.embeddings.create(model=c.model, input=MIXED)
    gaps = reference_gaps([d.embedding for d in mixed.data], ref)
    expect(not gaps, f"mixed-length batch differs from the reference: {gaps}")
    c.notes["embed_ref_min_cos"] = round(
        min(cosine(g, r) for g, r in zip(singles, ref, strict=True)), 5
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
    alias = c.req(
        "POST",
        "/v1/score",
        json={
            "model": c.model,
            "queries": "a cat",
            "documents": ["a feline", "a car"],
            "instruction": "ignored for bi-encoder",
        },
    )
    expect(
        alias.status_code == 200 and len(alias.json().get("data", [])) == 2,
        f"score aliases: {alias.status_code} {alias.text[:160]}",
    )

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
    # classify: the right label on top in every case, probabilities sum to 1
    for text, labels, want in CLASSIFY_CASES:
        k = c.req(
            "POST",
            "/v1/classify",
            json={"model": c.model, "input": text, "labels": labels},
        )
        expect(k.status_code == 200, f"classify {k.status_code} {k.text[:150]}")
        kr = k.json()["results"]
        expect(
            kr[0]["label"] == want and abs(sum(x["score"] for x in kr) - 1) < 1e-3,
            f"classify {text[:40]!r} wanted {want!r}: {kr}",
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


# ── EmbeddingGemma 2: text, image, audio, video, interleaved ─────────────────────────────

_MIME = {".png": "image/png", ".wav": "audio/wav"}


def _data_uri(path: str) -> str:
    import os

    with open(path, "rb") as f:
        raw = f.read()
    mime = _MIME.get(os.path.splitext(path)[1], "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def egemma2_wire_item(item):
    """A reference item (media as file paths) -> the request item (media as data URIs)."""
    if isinstance(item, str):
        return item
    out = {}
    for k, v in item.items():
        if k in ("image", "audio"):
            out[k] = [_data_uri(p) for p in v] if isinstance(v, list) else _data_uri(v)
        elif k == "video":
            out[k] = [[_data_uri(p) for p in fr] for fr in v]
        else:
            out[k] = v
    return out


def egemma2_chat_messages(m):
    """The vLLM-style `messages` form of the interleaved case (one embedding)."""
    img = lambda p: {"type": "image_url", "image_url": {"url": _data_uri(p)}}  # noqa: E731
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Photos: "},
                img(m["red"]),
                {"type": "text", "text": " and "},
                img(m["blue"]),
                {"type": "text", "text": " side by side."},
            ],
        }
    ]


@check(
    "embed_gemma2_served",
    "POST /v1/embeddings",
    "POST /v1/score",
    "POST /api/embed",
    needs="embed2",
    served=True,
)
def _embed_gemma2(c: Ctx):
    import os
    import tempfile

    from egemma2_cases import cases, make_media

    ref_fn = c.fixtures.get("embed_reference")
    expect(ref_fn, "no reference embedder in this run (fixtures['embed_reference'])")
    media = make_media(
        c.fixtures.get("egemma2_media") or tempfile.mkdtemp(prefix="eg2-")
    )
    cs = cases(media)
    names = list(cs)
    ref = dict(zip(names, ref_fn([cs[n] for n in names]), strict=True))

    def post(**body):
        r = c.req(
            "POST", "/v1/embeddings", json={"model": c.model, **body}, timeout=600
        )
        expect(r.status_code == 200, f"embeddings {r.status_code} {r.text[:200]}")
        return r.json()

    def served(name):
        item = egemma2_wire_item(cs[name])
        d = post(input=[item] if isinstance(item, dict) else item)
        return d["data"][0]["embedding"], d["usage"]["prompt_tokens"]

    cos = {}
    toks = {}
    for n in names:
        v, toks[n] = served(n)
        cos[n] = cosine(v, ref[n])
    bad = {n: round(x, 4) for n, x in cos.items() if x < REF_COS}
    expect(not bad, f"served vectors differ from the official reference (cos): {bad}")
    for n, lo in (
        ("text_long", 1500),
        ("image_red", 100),
        ("audio_speech", 50),
        ("video", 100),
    ):
        expect(toks[n] > lo, f"usage.prompt_tokens for {n}: {toks[n]} (want > {lo})")
    c.notes["egemma2_min_cos"] = round(min(cos.values()), 5)
    c.notes["egemma2_cos"] = {n: round(x, 4) for n, x in cos.items()}
    expect(len(ref["text_cat"]) == 768, "native dimension is 768")

    # one batch of every case (mixed lengths and modalities) equals the singles
    wire = [egemma2_wire_item(cs[n]) for n in names]
    wire = [w if isinstance(w, dict) else {"text": w} for w in wire]
    b = post(input=wire)
    expect([d["index"] for d in b["data"]] == list(range(len(names))), "batch order")
    gaps = {
        n: round(cosine(d["embedding"], ref[n]), 4)
        for n, d in zip(names, b["data"], strict=True)
    }
    gaps = {n: x for n, x in gaps.items() if x < REF_COS}
    expect(not gaps, f"batch differs from the reference: {gaps}")

    # task prompts: SearchQuery / Document are the official prefixes; the vectors differ
    q, d_ = "What causes the northern lights?", "Sun particles hit the atmosphere."
    rq = ref_fn([q])[0]
    plain = post(input=q)["data"][0]["embedding"]
    expect(cosine(plain, rq) > REF_COS, "plain text is the raw-text reference")
    tq = post(input=q, task="SearchQuery")["data"][0]["embedding"]
    expect(
        cosine(tq, plain) < 0.9999,
        "task=SearchQuery changed nothing (prefix not applied)",
    )
    task_ref = c.fixtures.get("embed_reference_task")
    if task_ref:
        for task, text in (("SearchQuery", q), ("Document", d_)):
            got = post(input=text, task=task)["data"][0]["embedding"]
            expect(
                cosine(got, task_ref(text, task)) > REF_COS,
                f"task={task} differs from the reference",
            )
    bad_task = c.req(
        "POST", "/v1/embeddings", json={"model": c.model, "input": q, "task": "Nope"}
    )
    err_ok(bad_task, "openai")
    expect(bad_task.status_code == 400, f"unknown task -> {bad_task.status_code}")

    # Matryoshka
    d256 = post(input=CAT, dimensions=256)["data"][0]["embedding"]
    full = ref["text_cat"]
    tr = [x for x in full[:256]]
    expect(
        len(d256) == 256 and cosine(d256, tr) > REF_COS,
        "dimensions=256 vs truncated reference",
    )
    expect(
        abs(math.sqrt(sum(x * x for x in d256)) - 1) < 1e-3,
        "dimensions=256 not renormalised",
    )

    # chat-style messages (vLLM form) = the interleaved item
    mm = post(messages=egemma2_chat_messages(media))["data"][0]["embedding"]
    expect(
        cosine(mm, ref["interleaved"]) > REF_COS,
        "messages form differs from the interleaved item",
    )

    # cross-modal: the spoken sentence is closer to its text than to another sentence
    sp = served("audio_speech")[0]
    from egemma2_cases import SPEECH

    t_same = post(input=SPEECH)["data"][0]["embedding"]
    t_other = post(input=STOCK)["data"][0]["embedding"]
    expect(
        cosine(sp, t_same) > cosine(sp, t_other) + 0.05,
        "audio not closer to its transcript",
    )
    rd = served("image_red")[0]
    expect(
        cosine(
            rd, post(input="a red circle on a white background")["data"][0]["embedding"]
        )
        > cosine(rd, post(input="a blue rectangle")["data"][0]["embedding"]),
        "image not closer to its caption",
    )

    # errors are 400, never 500: marker count, wrong media, unreadable audio, over the context
    for body in (
        {"input": [{"text": "x <|image|> y"}]},
        {"input": [{"audio": "data:audio/wav;base64,AAAA"}]},
        {"input": [{"image": "data:image/png;base64,AAAA"}]},
        {"input": "word " * 12000},
    ):
        r = c.req(
            "POST", "/v1/embeddings", json={"model": c.model, **body}, timeout=600
        )
        err_ok(r, "openai")
        expect(
            r.status_code in (400, 422),
            f"bad input {str(body)[:60]} -> {r.status_code} {r.text[:100]}",
        )

    # score and Ollama go through the same engine
    s = c.req(
        "POST",
        "/v1/score",
        json={"model": c.model, "text_1": CAT, "text_2": [KITTEN, STOCK]},
    )
    expect(s.status_code == 200, f"score {s.status_code} {s.text[:150]}")
    sc = [x["score"] for x in s.json()["data"]]
    expect(sc[0] > sc[1], f"score {sc}")
    o = c.req("POST", "/api/embed", json={"model": c.model, "input": [CAT, STOCK]})
    expect(o.status_code == 200, f"/api/embed {o.status_code} {o.text[:150]}")
    oe = o.json()["embeddings"]
    expect(
        cosine(oe[0], ref["text_cat"]) > REF_COS,
        "/api/embed differs from the reference",
    )
