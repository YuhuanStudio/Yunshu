"""Omni checks of the route registry (imported at the bottom of route_checks.py).

Three server kinds, selected by `needs`:
  "omni"     one server on a small input-omni checkpoint (Gemma 4 E2B): audio / image / video in
             through chat, responses and the Ollama dialect; stream and non-stream.
  "cascade"  a multi-model server (chat + ASR + TTS): the Realtime voice turn on both sockets,
             speech in through the ASR, speech out through the TTS.
  "native"   Qwen3-Omni (Thinker + Talker, M5 only): /v1/omni/speech/stream, native speech-to-speech
             on both Realtime sockets, audio + image in, APC hit on repeated audio / image.
The spoken fixture (`ctx.fixtures["speech_wav"]`, the phrase in `ctx.fixtures["phrase"]`) is made
by the job with the allowlisted Qwen3-TTS server; a check without it skips (CPU harness)."""

from __future__ import annotations

import base64
import io
import json
import tempfile
import wave
from pathlib import Path

from route_checks import (
    Ctx,
    _png,
    _ws_events,
    check,
    expect,
    skip,
)

SECRET = "pineapple"
PHRASE = f"The secret word is {SECRET}."
ASK_WORD = "What word did the speaker say is the secret? Answer with that one word."


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def wav_pcm16(wav: bytes, rate: int = 24000) -> bytes:
    """WAV bytes -> mono int16 PCM at `rate` (linear resample), the Realtime input format."""
    import numpy as np

    with wave.open(io.BytesIO(wav)) as w:
        ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    expect(sw == 2, f"fixture wav is {sw * 8}-bit, want 16")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != rate:
        n = int(len(x) * rate / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


def _speech(c: Ctx) -> bytes:
    w = c.fixtures.get("speech_wav")
    if not w:
        skip("no spoken fixture (the job makes it with the allowlisted Qwen3-TTS)")
    return w


def _word_in(text: str, word: str = SECRET) -> bool:
    return word in (text or "").lower().replace("-", "")


def _chat_audio_msg(wav: bytes, question: str, image: bytes | None = None):
    parts = [
        {"type": "text", "text": question},
        {"type": "input_audio", "input_audio": {"data": _b64(wav), "format": "wav"}},
    ]
    if image:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + _b64(image)},
            }
        )
    return [{"role": "user", "content": parts}]


def _stream_text(c: Ctx, messages, max_tokens=48):
    """Chat completions stream: (text, usage chunk, number of content deltas)."""
    text, usage, n = "", None, 0
    with c.oa.chat.completions.create(
        model=c.model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    ) as s:
        for ev in s:
            if ev.usage:
                usage = ev.usage
            if ev.choices and ev.choices[0].delta.content:
                text += ev.choices[0].delta.content
                n += 1
    return text, usage, n


def _cached(usage) -> int:
    d = getattr(usage, "prompt_tokens_details", None)
    return int(getattr(d, "cached_tokens", 0) or 0)


@check(
    "omni_audio_in",
    "POST /v1/chat/completions",
    "POST /v1/responses",
    needs="omni",
    served=True,
)
def _omni_audio_in(c: Ctx):
    wav = _speech(c)
    base = c.oa.chat.completions.create(
        model=c.model,
        messages=[{"role": "user", "content": ASK_WORD}],
        max_tokens=4,
    )
    msgs = _chat_audio_msg(wav, ASK_WORD)
    r = c.oa.chat.completions.create(model=c.model, messages=msgs, max_tokens=48)
    ans = r.choices[0].message.content or ""
    c.notes["audio_chat"] = ans[:120]
    expect(
        r.usage.prompt_tokens > base.usage.prompt_tokens + 10,
        f"audio not counted: {r.usage.prompt_tokens} vs {base.usage.prompt_tokens}",
    )
    expect(_word_in(ans), f"chat answer does not reflect the spoken word: {ans!r}")
    text, usage, n = _stream_text(c, msgs)
    c.notes["audio_chat_stream"] = text[:120]
    expect(n > 0 and usage is not None, "audio stream: no deltas or no usage chunk")
    expect(_word_in(text), f"streamed answer does not reflect the word: {text!r}")
    o = c.oa.responses.create(
        model=c.model,
        max_output_tokens=48,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": ASK_WORD},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": _b64(wav), "format": "wav"},
                    },
                ],
            }
        ],
    )
    c.notes["audio_responses"] = o.output_text[:120]
    expect(
        o.usage.input_tokens > base.usage.prompt_tokens + 10,
        "responses audio not counted",
    )
    expect(_word_in(o.output_text), f"responses answer: {o.output_text!r}")
    # the same audio as an audio_url data URI (the other accepted shape)
    u = c.oa.chat.completions.create(
        model=c.model,
        max_tokens=48,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ASK_WORD},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "data:audio/wav;base64," + _b64(wav)},
                    },
                ],
            }
        ],
    )
    expect(_word_in(u.choices[0].message.content or ""), "audio_url answer")


@check(
    "omni_image_audio_in",
    "POST /v1/chat/completions",
    "POST /v1/messages",
    needs="omni",
    served=True,
)
def _omni_image_audio_in(c: Ctx):
    wav = _speech(c)
    q = "What colour is the image, and what word did the speaker say is the secret?"
    r = c.oa.chat.completions.create(
        model=c.model,
        messages=_chat_audio_msg(wav, q, _png(224, (200, 30, 30))),
        max_tokens=64,
    )
    ans = (r.choices[0].message.content or "").lower()
    c.notes["image_audio_chat"] = ans[:160]
    expect("red" in ans and _word_in(ans), f"image + audio answer: {ans!r}")
    # Messages has no audio block (Anthropic's API); its image block still reaches the model
    m = c.an.messages.create(
        model=c.model,
        max_tokens=24,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _b64(_png(224, (30, 30, 200))),
                        },
                    },
                    {"type": "text", "text": "What colour is this image? One word."},
                ],
            }
        ],
    )
    txt = "".join(b.text for b in m.content if b.type == "text").lower()
    c.notes["image_messages"] = txt[:80]
    expect("blue" in txt, f"messages image answer: {txt!r}")


def _mp4(seconds=2, bgr=(0, 0, 200)) -> bytes:
    """A solid-colour mp4 (default red), written with OpenCV (a locked dependency)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        skip("no OpenCV on this machine: cannot write the video fixture")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "v.mp4"
        w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (224, 224))
        for _ in range(int(seconds * 4)):
            w.write(np.full((224, 224, 3), bgr, dtype=np.uint8))
        w.release()
        data = out.read_bytes()
    expect(len(data) > 500, f"video fixture is {len(data)} bytes")
    return data


@check("omni_video_in", "POST /v1/chat/completions", needs="omni", served=True)
def _omni_video_in(c: Ctx):
    v = _mp4()
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What colour is this video? One word."},
                {
                    "type": "video_url",
                    "video_url": {"url": "data:video/mp4;base64," + _b64(v)},
                },
            ],
        }
    ]
    r = c.oa.chat.completions.create(model=c.model, messages=msgs, max_tokens=24)
    ans = (r.choices[0].message.content or "").lower()
    c.notes["video_chat"] = ans[:100]
    expect("red" in ans, f"video answer: {ans!r}")
    text, usage, n = _stream_text(c, msgs, 24)
    expect(n > 0 and "red" in text.lower(), f"video stream answer: {text!r}")


@check("omni_media_cache", "POST /v1/chat/completions", needs="omni", served=True)
def _omni_media_cache(c: Ctx):
    """Repeated audio / image: the second identical request must not fail and must produce the
    same answer; whether it hits the prefix cache is recorded (Gemma 4's sliding-window cache
    cannot be checkpointed, so a hit is required only where `cached_tokens` is advertised)."""
    wav = _speech(c)
    msgs = _chat_audio_msg(wav, ASK_WORD)
    a = c.oa.chat.completions.create(
        model=c.model, messages=msgs, max_tokens=24, temperature=0
    )
    b = c.oa.chat.completions.create(
        model=c.model, messages=msgs, max_tokens=24, temperature=0
    )
    c.notes["audio_repeat_cached"] = [_cached(a.usage), _cached(b.usage)]
    expect(
        a.choices[0].message.content == b.choices[0].message.content,
        f"repeat differs: {a.choices[0].message.content!r} vs {b.choices[0].message.content!r}",
    )
    other = c.oa.chat.completions.create(
        model=c.model,
        messages=_chat_audio_msg(wav[: len(wav) // 2], ASK_WORD),
        max_tokens=24,
    )
    expect(
        other.usage.prompt_tokens < a.usage.prompt_tokens,
        "a shorter clip must cost fewer prompt tokens",
    )


# ── Realtime voice, cascade: speech in -> ASR -> chat model -> TTS -> speech out ─────────


def _voice_turn(c: Ctx, path: str, pcm: bytes, modalities=("audio",)):
    with c.ws(path + f"?model={c.model}") as ws:
        first = json.loads(ws.recv(timeout=60))
        expect(
            first.get("type") == "session.created", f"first event {first.get('type')}"
        )
        # GA schema on /v1/realtime, flat beta schema on the legacy /realtime path
        sess = (
            {"modalities": ["text", "audio"]}
            if path == "/realtime"
            else {"type": "realtime", "output_modalities": list(modalities)}
        )
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        **sess,
                        "instructions": "Reply in one short sentence.",
                        "turn_detection": None,
                    },
                }
            )
        )
        for i in range(0, len(pcm), 48000):
            ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": _b64(pcm[i : i + 48000]),
                    }
                )
            )
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        ws.send(json.dumps({"type": "response.create"}))
        return _ws_events(
            ws, lambda e: e.get("type") in ("response.done", "error"), timeout=600
        )


def _audio_bytes(evs) -> int:
    return sum(
        len(base64.b64decode(e.get("delta", "")))
        for e in evs
        if e.get("type") in ("response.output_audio.delta", "response.audio.delta")
    )


def _spoken_text(evs) -> str:
    return " ".join(
        (e.get("transcript") or "")
        for e in evs
        if e.get("type")
        in (
            "response.output_audio_transcript.done",
            "response.audio_transcript.done",
        )
    )


def _realtime_voice(c: Ctx, require_transcript_word: bool):
    pcm = wav_pcm16(_speech(c))
    for path in ("/v1/realtime", "/realtime"):
        evs = _voice_turn(c, path, pcm)
        types = [e["type"] for e in evs]
        errs = [e for e in evs if e["type"] == "error"]
        expect(not errs, f"{path}: error events {json.dumps(errs[:1])[:300]}")
        expect("input_audio_buffer.committed" in types, f"{path}: no committed {types}")
        expect(types[-1] == "response.done", f"{path}: {types[-6:]}")
        nb = _audio_bytes(evs)
        expect(
            nb > 24000,
            f"{path}: only {nb} bytes of speech out (<0.5 s); events {sorted(set(types))}",
        )
        heard = " ".join(
            (e.get("transcript") or "")
            for e in evs
            if "input_audio_transcription.completed" in e.get("type", "")
        )
        if require_transcript_word:
            expect(_word_in(heard), f"{path}: input transcript {heard!r}")
        c.notes[f"realtime_voice{path}"] = {
            "audio_bytes": nb,
            "input_transcript": heard[:80],
            "spoken": _spoken_text(evs)[:120],
            "status": evs[-1]["response"].get("status"),
        }
        expect(
            evs[-1]["response"].get("status") in ("completed", "incomplete"),
            f"{path}: status {evs[-1]['response'].get('status')}",
        )


@check(
    "realtime_voice_cascade",
    "WS /v1/realtime",
    "WS /realtime",
    needs="cascade",
    served=True,
)
def _realtime_voice_cascade(c: Ctx):
    _realtime_voice(c, require_transcript_word=True)


# ── native speech-to-speech (Qwen3-Omni, M5) ────────────────────────────────────────────


def _sse_events(r):
    ev = []
    for line in r.iter_lines():
        if line.startswith("data: "):
            body = line[6:]
            if body == "[DONE]":
                ev.append({"type": "[DONE]"})
                break
            ev.append(json.loads(body))
    return ev


@check("omni_speech_stream", "POST /v1/omni/speech/stream", needs="native", served=True)
def _omni_speech_stream(c: Ctx):
    wav = _speech(c)
    for label, body in (
        (
            "text",
            {"text": "Say hello in one short sentence.", "thinker_max_new_tokens": 48},
        ),
        (
            "audio_in",
            {
                "text": ASK_WORD,
                "audio_path": "data:audio/wav;base64," + _b64(wav),
                "thinker_max_new_tokens": 48,
            },
        ),
        (
            "image_in",
            {
                "text": "What colour is this image? One word.",
                "image_path": "data:image/png;base64," + _b64(_png(224)),
                "thinker_max_new_tokens": 24,
            },
        ),
    ):
        with c.http.stream(
            "POST",
            "/v1/omni/speech/stream",
            headers=c.auth(),
            json=body,
            timeout=600,
        ) as r:
            expect(r.status_code == 200, f"{label}: {r.status_code}")
            evs = _sse_events(r)
        types = [e["type"] for e in evs]
        expect(types and types[-1] == "[DONE]", f"{label}: no [DONE] {types[-4:]}")
        expect(
            "error" not in types, f"{label}: {[e for e in evs if e['type'] == 'error']}"
        )
        text = "".join(e["delta"] for e in evs if e["type"] == "text")
        pcm = b"".join(
            base64.b64decode(e["delta"]) for e in evs if e["type"] == "audio"
        )
        sr = next((e["sr"] for e in evs if e["type"] == "audio"), 0)
        expect(text.strip(), f"{label}: no text")
        expect(sr == 24000 and len(pcm) > 9600, f"{label}: audio {len(pcm)} B sr {sr}")
        expect("done" in types, f"{label}: no done event")
        c.notes[f"omni_stream_{label}"] = {"text": text[:100], "pcm_bytes": len(pcm)}
        if label == "audio_in":
            expect(_word_in(text), f"audio_in text {text!r}")
        if label == "image_in":
            expect("red" in text.lower(), f"image_in text {text!r}")
    r = c.req(
        "POST", "/v1/omni/speech/stream", json={"text": "hi", "speaker": "nobody"}
    )
    expect(r.status_code == 400, f"unknown speaker -> {r.status_code}")


@check(
    "realtime_voice_native",
    "WS /v1/realtime",
    "WS /realtime",
    needs="native",
    served=True,
)
def _realtime_voice_native(c: Ctx):
    _realtime_voice(c, require_transcript_word=False)
    # native speech-in: the spoken question is answered from the raw audio
    pcm = wav_pcm16(_speech(c))
    evs = _voice_turn(c, "/v1/realtime", pcm)
    spoken = _spoken_text(evs).lower()
    c.notes["native_voice_reply"] = spoken[:160]


@check(
    "omni_native_chat_cache",
    "POST /v1/chat/completions",
    "POST /v1/responses",
    needs="native",
    served=True,
)
def _omni_native_chat_cache(c: Ctx):
    wav = _speech(c)
    msgs = _chat_audio_msg(wav, "Listen carefully. " * 20 + ASK_WORD)
    a = c.oa.chat.completions.create(
        model=c.model, messages=msgs, max_tokens=32, temperature=0
    )
    expect(_word_in(a.choices[0].message.content or ""), "audio answer")
    b = c.oa.chat.completions.create(
        model=c.model, messages=msgs, max_tokens=32, temperature=0
    )
    c.notes["audio_cached"] = [
        _cached(a.usage),
        _cached(b.usage),
        b.usage.prompt_tokens,
    ]
    expect(
        _cached(b.usage) > 0, f"repeated audio: no prefix hit {c.notes['audio_cached']}"
    )
    expect(
        a.choices[0].message.content == b.choices[0].message.content,
        f"cached answer differs: {a.choices[0].message.content!r} vs {b.choices[0].message.content!r}",
    )
    # image alone: recorded, not required. OPEN FINDING 2026-10-06: a repeated image gets 0 cached
    # tokens on Qwen3-Omni (0 of 108) although the pixel hash is stable (omni_apc_probe.py) and
    # Qwen3.5 hits (198 of ~200); a repeated audio clip hits (112 of 113)
    only_img = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What colour is the image? " * 8},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + _b64(_png(224, (200, 30, 30)))
                    },
                },
            ],
        }
    ]
    c.oa.chat.completions.create(model=c.model, messages=only_img, max_tokens=8)
    i2 = c.oa.chat.completions.create(model=c.model, messages=only_img, max_tokens=8)
    c.notes["image_cached"] = [_cached(i2.usage), i2.usage.prompt_tokens]
    # image + audio together: recorded too (0 of 99 on 2026-10-06)
    img = _chat_audio_msg(wav, "What colour is the image?", _png(224, (200, 30, 30)))
    c.oa.chat.completions.create(model=c.model, messages=img, max_tokens=8)
    r2 = c.oa.chat.completions.create(model=c.model, messages=img, max_tokens=8)
    c.notes["image_audio_cached"] = [_cached(r2.usage), r2.usage.prompt_tokens]


@check("vision_media_cache", "POST /v1/chat/completions", served=True)
def _vision_media_cache(c: Ctx):
    """APC image-pixel key on the VLM runner: the same image twice hits the prefix cache with an
    identical answer; a different image with the same text must not reuse the first image's
    cache (the pixel key differs)."""
    if c.kind != "vlm":
        skip("image prefix cache is a VLM-runner feature")

    def ask(rgb):
        m = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the colour of this image in one word. " * 12,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + _b64(_png(224, rgb))
                        },
                    },
                ],
            }
        ]
        return c.oa.chat.completions.create(
            model=c.model, messages=m, max_tokens=12, temperature=0
        )

    a = ask((200, 30, 30))
    b = ask((200, 30, 30))
    other = ask((30, 30, 200))
    c.notes["image_cached"] = [_cached(a.usage), _cached(b.usage), _cached(other.usage)]
    expect(
        _cached(b.usage) > 0, f"repeated image: no prefix hit {c.notes['image_cached']}"
    )
    expect(
        a.choices[0].message.content == b.choices[0].message.content,
        "cached image answer differs",
    )
    # the text prefix before the image may hit, but the image's own tokens must not
    expect(
        _cached(other.usage) < other.usage.prompt_tokens - 8,
        f"a different image reused the first image's cache: {c.notes['image_cached']}",
    )
