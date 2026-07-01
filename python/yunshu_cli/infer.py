"""Agent-facing inference commands — a thin, JSON-aware client over a running server.

Every server capability is a single non-interactive command (`yunshu complete/embed/
tokenize/rerank/transcribe/speak/ocr/image`), so an agent can drive the whole engine
through the CLI without constructing HTTP calls. All honor the global ``--json`` flag and
exit non-zero on failure. They talk to a running `yunshu serve` (start it first).
"""

from __future__ import annotations

import base64
import contextlib
from pathlib import Path

import typer

from ._output import auth_headers, console, emit, fail

_URL = typer.Option(
    "http://localhost:8000",
    "--url",
    "-u",
    envvar="YUNSHU_GATEWAY_URL",
    help="Server URL.",
)
_MODEL = typer.Option("local", "--model", "-m", help="Model name (single-model: any).")


def _post(url: str, path: str, *, json=None, files=None, data=None, timeout=300):
    """POST to the server; fail() cleanly on connection error / non-2xx."""
    import httpx

    try:
        resp = httpx.post(
            f"{url}{path}",
            json=json,
            files=files,
            data=data,
            headers=auth_headers(),
            timeout=timeout,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout):
        fail(f"Cannot connect to {url} — start the server with `yunshu serve`.", code=2)
    except Exception as e:  # noqa: BLE001
        fail(f"Request failed: {e}", code=1)
    if resp.status_code >= 300:
        detail = resp.text[:500]
        with contextlib.suppress(Exception):
            body = resp.json()
            detail = body.get("error", {}).get("message") or body.get("detail", detail)
        fail(
            f"Server error {resp.status_code}: {detail}",
            code=1,
            status=resp.status_code,
        )
    return resp


def _body(resp) -> dict:
    """Parse a JSON response body; fail() cleanly if it isn't JSON (empty/HTML/etc.)."""
    try:
        return resp.json()
    except Exception:  # noqa: BLE001
        fail(f"Server returned a non-JSON response: {resp.text[:200]!r}", code=1)


def _pick(fn):
    """Extract a value from a parsed body via `fn`; fail() cleanly on an unexpected shape
    (missing key / empty list / wrong type) instead of crashing with a traceback."""
    try:
        return fn()
    except (KeyError, IndexError, TypeError, ValueError) as e:
        fail(f"Unexpected response shape ({e}).", code=1)


def _binary(resp, kind: str) -> bytes:
    """Return response bytes for a file-producing command; fail() if the server actually
    sent JSON/text (i.e. an error body) rather than the expected media."""
    ct = resp.headers.get("content-type", "")
    if ct.startswith(("application/json", "text/")):
        fail(
            f"Expected {kind} bytes but the server returned {ct}: {resp.text[:300]}",
            code=1,
        )
    return resp.content


def _write(out: Path, content: bytes) -> None:
    """Write bytes to `out`; fail() cleanly on an unwritable path instead of a traceback."""
    try:
        out.write_bytes(content)
    except OSError as e:
        fail(f"Cannot write {out}: {e}", code=1)


def _get(url: str, path: str, *, timeout=30):
    """GET from the server; fail() cleanly on connection error / non-2xx."""
    import httpx

    try:
        resp = httpx.get(f"{url}{path}", headers=auth_headers(), timeout=timeout)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        fail(f"Cannot connect to {url} — start the server with `yunshu serve`.", code=2)
    except Exception as e:  # noqa: BLE001
        fail(f"Request failed: {e}", code=1)
    if resp.status_code >= 300:
        fail(
            f"Server error {resp.status_code}: {resp.text[:300]}",
            code=1,
            status=resp.status_code,
        )
    return resp


def _b64file(p: Path) -> str:
    """Base64-encode a file's bytes for JSON transport."""
    return base64.b64encode(p.read_bytes()).decode()


def _img_result(url: str, path: str, body: dict, out: Path) -> None:
    """POST an image op returning b64 image data; write it to `out` and emit JSON."""
    resp = _post(url, path, json=body, timeout=600)
    d = _body(resp)
    b64 = _pick(lambda: d["data"][0]["b64_json"])
    _write(out, _pick(lambda: base64.b64decode(b64)))
    emit(
        {"file": str(out), "bytes": out.stat().st_size},
        human=lambda: console.print(
            f"wrote [bold]{out}[/] ({out.stat().st_size} bytes)"
        ),
    )


def _audio_result(url: str, path: str, body: dict, out: Path) -> None:
    """POST an audio op returning raw audio bytes (response_format=wav); write to `out`."""
    resp = _post(url, path, json=body, timeout=600)
    _write(out, _binary(resp, "audio"))
    emit(
        {"file": str(out), "bytes": out.stat().st_size},
        human=lambda: console.print(
            f"wrote [bold]{out}[/] ({out.stat().st_size} bytes)"
        ),
    )


# ── text ──────────────────────────────────────────────────────────────────────


def complete(
    prompt: str = typer.Argument(..., help="The prompt / user message."),
    url: str = _URL,
    model: str = _MODEL,
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max output tokens."),
    temperature: float = typer.Option(0.7, "--temperature", "-t"),
    system: str | None = typer.Option(None, "--system", "-s", help="System prompt."),
):
    """Chat completion (POST /v1/chat/completions)."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    resp = _post(
        url,
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    d = _body(resp)
    choice = _pick(lambda: d["choices"][0])
    text = choice.get("message", {}).get("content") or ""
    emit(
        {
            "text": text,
            "model": d.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "usage": d.get("usage"),
        },
        human=lambda: console.print(text),
    )


def tokenize(
    text: str = typer.Argument(..., help="Text to tokenize."),
    url: str = _URL,
    model: str = _MODEL,
    ids: bool = typer.Option(
        False, "--ids", help="Return token IDs (/v1/tokenize) instead of just a count."
    ),
):
    """Count tokens, or with --ids return token IDs (POST /v1/token_count | /v1/tokenize)."""
    if ids:
        resp = _post(url, "/v1/tokenize", json={"model": model, "text": text})
        d = _body(resp)
        emit(d, human=lambda: console.print(str(d.get("tokens"))))
        return
    resp = _post(url, "/v1/token_count", json={"model": model, "prompt": text})
    d = _body(resp)
    count = d.get("token_count", d.get("count"))
    emit(d, human=lambda: console.print(f"tokens: [bold]{count}[/]"))


def detokenize(
    tokens: list[int] = typer.Argument(..., help="Token IDs to decode back to text."),
    url: str = _URL,
    model: str = _MODEL,
):
    """Decode token IDs back to text (POST /v1/detokenize)."""
    resp = _post(url, "/v1/detokenize", json={"model": model, "tokens": list(tokens)})
    d = _body(resp)
    emit(d, human=lambda: console.print(d.get("text", "")))


def embed(
    text: str = typer.Argument(..., help="Text to embed."),
    url: str = _URL,
    model: str = _MODEL,
):
    """Embed text (POST /v1/embeddings)."""
    resp = _post(url, "/v1/embeddings", json={"model": model, "input": text})
    d = _body(resp)
    vec = _pick(lambda: d["data"][0]["embedding"])
    emit(
        {
            "embedding": vec,
            "dim": len(vec),
            "model": d.get("model"),
            "usage": d.get("usage"),
        },
        human=lambda: console.print(
            f"dim [bold]{len(vec)}[/] — first 8: {[round(x, 4) for x in vec[:8]]}"
        ),
    )


def rerank(
    query: str = typer.Argument(..., help="The query."),
    documents: list[str] = typer.Argument(..., help="Documents to rank."),
    url: str = _URL,
    model: str = _MODEL,
    top_n: int = typer.Option(0, "--top-n", help="Return top N (0 = all)."),
):
    """Rerank documents against a query (POST /v1/rerank)."""
    body = {"model": model, "query": query, "documents": list(documents)}
    if top_n:
        body["top_n"] = top_n
    resp = _post(url, "/v1/rerank", json=body)
    d = _body(resp)

    def _human():
        for r in d.get("results", []):
            console.print(
                f"  {r['relevance_score']:.4f}  [{r['index']}] {documents[r['index']][:70]}"
            )

    emit(d, human=_human)


# ── audio / vision / image (file in/out) ────────────────────────────────────────


def transcribe(
    file: Path = typer.Argument(..., exists=True, help="Audio file to transcribe."),
    url: str = _URL,
    model: str = _MODEL,
    language: str | None = typer.Option(None, "--language", "-l"),
    translate: bool = typer.Option(False, "--translate", help="Translate to English."),
):
    """Transcribe (or --translate) audio (POST /v1/audio/transcriptions|translations)."""
    path = "/v1/audio/translations" if translate else "/v1/audio/transcriptions"
    data = {"model": model}
    if language and not translate:
        data["language"] = language
    with open(file, "rb") as f:
        resp = _post(
            url,
            path,
            files={"file": (file.name, f, "application/octet-stream")},
            data=data,
        )
    d = _body(resp)
    emit(d, human=lambda: console.print(d.get("text", "")))


def speak(
    text: str = typer.Argument(..., help="Text to synthesize."),
    out: Path = typer.Option(..., "--out", "-o", help="Output audio file (.wav/.mp3)."),
    url: str = _URL,
    model: str = _MODEL,
    voice: str = typer.Option("ethan", "--voice", help="Voice / speaker."),
):
    """Synthesize speech to a file (POST /v1/audio/speech)."""
    fmt = out.suffix.lstrip(".").lower() or "wav"
    resp = _post(
        url,
        "/v1/audio/speech",
        json={"model": model, "input": text, "voice": voice, "response_format": fmt},
    )
    _write(out, _binary(resp, "audio"))
    emit(
        {"file": str(out), "bytes": out.stat().st_size, "format": fmt},
        human=lambda: console.print(
            f"wrote [bold]{out}[/] ({len(resp.content)} bytes)"
        ),
    )


def ocr(
    file: Path = typer.Argument(..., exists=True, help="Image file to OCR."),
    url: str = _URL,
    model: str = _MODEL,
):
    """Extract text from an image (POST /v1/ocr)."""
    with open(file, "rb") as f:
        resp = _post(
            url,
            "/v1/ocr",
            files={"file": (file.name, f, "image/png")},
            data={"model": model},
        )
    d = _body(resp)
    emit(d, human=lambda: console.print(d.get("text", "")))


def image(
    prompt: str = typer.Argument(..., help="Image prompt."),
    out: Path = typer.Option(..., "--out", "-o", help="Output image file (.png)."),
    url: str = _URL,
    model: str = typer.Option("Z-Image-Turbo-MLX-4bit", "--model", "-m"),
    size: str = typer.Option("1024x1024", "--size"),
    steps: int = typer.Option(4, "--steps", help="Inference steps."),
):
    """Generate an image to a file (POST /v1/images/generations)."""
    resp = _post(
        url,
        "/v1/images/generations",
        json={
            "model": model,
            "prompt": prompt,
            "size": size,
            "num_inference_steps": steps,
            "response_format": "b64_json",
        },
    )
    d = _body(resp)
    b64 = _pick(lambda: d["data"][0]["b64_json"])
    _write(out, _pick(lambda: base64.b64decode(b64)))
    emit(
        {"file": str(out), "bytes": out.stat().st_size, "size": size},
        human=lambda: console.print(
            f"wrote [bold]{out}[/] ({out.stat().st_size} bytes)"
        ),
    )


def classify(
    text: str = typer.Argument(..., help="Text to classify."),
    labels: list[str] = typer.Argument(..., help="Candidate labels."),
    url: str = _URL,
    model: str = _MODEL,
):
    """Zero-shot classify text into one of the labels (POST /v1/classify)."""
    resp = _post(
        url,
        "/v1/classify",
        json={"model": model, "input": text, "labels": list(labels)},
    )
    d = _body(resp)
    results = d.get("results") or d.get("data") or []

    def _human():
        for r in sorted(results, key=lambda x: -x.get("score", 0)):
            console.print(f"  {r.get('score', 0):.4f}  {r.get('label')}")

    emit(d, human=_human)


def score(
    text1: str = typer.Argument(..., help="First text."),
    text2: str = typer.Argument(..., help="Second text."),
    url: str = _URL,
    model: str = _MODEL,
    scoring_type: str = typer.Option(
        "cosine", "--type", help="Similarity: cosine, dot, or euclidean."
    ),
):
    """Similarity score between two texts (POST /v1/score)."""
    resp = _post(
        url,
        "/v1/score",
        json={
            "model": model,
            "text_1": text1,
            "text_2": text2,
            "scoring_type": scoring_type,
        },
    )
    d = _body(resp)
    data = d.get("data", [])
    emit(
        d,
        human=lambda: console.print(
            ", ".join(f"{r.get('score', 0):.4f}" for r in data) or "(no score)"
        ),
    )


def cancel(
    request_id: str = typer.Argument(
        None, help="Generation request id to cancel (omit when using --all)."
    ),
    all_: bool = typer.Option(False, "--all", help="Cancel all active generations."),
    url: str = _URL,
):
    """Cancel an in-flight generation (POST /v1/cancel)."""
    if not request_id and not all_:
        fail("Provide a request id, or --all to cancel everything.", code=1)
    resp = _post(url, "/v1/cancel", json={"request_id": request_id, "cancel_all": all_})
    d = _body(resp)
    emit(d, human=lambda: console.print(str(d)))


def image_edit(
    image_file: Path = typer.Argument(..., exists=True, help="Source image to edit."),
    prompt: str = typer.Argument(..., help="Edit instruction."),
    out: Path = typer.Option(..., "--out", "-o", help="Output image file (.png)."),
    url: str = _URL,
    model: str = typer.Option("Z-Image-Turbo-MLX-4bit", "--model", "-m"),
    steps: int = typer.Option(4, "--steps", help="Inference steps."),
    strength: float = typer.Option(
        0.8, "--strength", help="Re-denoise strength (1.0=full, 0.0=keep source)."
    ),
):
    """Edit an image with a text prompt (POST /v1/images/edits)."""
    b64src = base64.b64encode(image_file.read_bytes()).decode()
    resp = _post(
        url,
        "/v1/images/edits",
        json={
            "image": b64src,
            "prompt": prompt,
            "model": model,
            "num_inference_steps": steps,
            "denoise_strength": strength,
            "response_format": "b64_json",
        },
        timeout=600,
    )
    d = _body(resp)
    b64 = _pick(lambda: d["data"][0]["b64_json"])
    _write(out, _pick(lambda: base64.b64decode(b64)))
    emit(
        {"file": str(out), "bytes": out.stat().st_size},
        human=lambda: console.print(
            f"wrote [bold]{out}[/] ({out.stat().st_size} bytes)"
        ),
    )


def video(
    prompt: str = typer.Argument(..., help="Video prompt."),
    out: Path = typer.Option(..., "--out", "-o", help="Output video file (.mp4)."),
    url: str = _URL,
    model: str = typer.Option("wan-2.2-t2v", "--model", "-m"),
    frames: int = typer.Option(81, "--frames", help="Number of frames."),
    steps: int = typer.Option(20, "--steps", help="Inference steps."),
):
    """Generate a video to a file (POST /v1/video/generations)."""
    resp = _post(
        url,
        "/v1/video/generations",
        json={
            "model": model,
            "prompt": prompt,
            "num_frames": frames,
            "num_inference_steps": steps,
            "response_format": "mp4",
        },
        timeout=1800,
    )
    d = _body(resp)
    b64 = _pick(lambda: d["data"][0]["b64_json"])
    _write(out, _pick(lambda: base64.b64decode(b64)))
    emit(
        {"file": str(out), "bytes": out.stat().st_size},
        human=lambda: console.print(
            f"wrote [bold]{out}[/] ({out.stat().st_size} bytes)"
        ),
    )


def image_variations(
    image_file: Path = typer.Argument(..., exists=True, help="Source image."),
    out: Path = typer.Option(..., "--out", "-o", help="Output image file."),
    url: str = _URL,
    model: str = typer.Option("Z-Image-Turbo-MLX-4bit", "--model", "-m"),
    steps: int = typer.Option(4, "--steps"),
):
    """Generate variations of an image, no prompt (POST /v1/images/variations)."""
    _img_result(
        url,
        "/v1/images/variations",
        {
            "image": _b64file(image_file),
            "model": model,
            "num_inference_steps": steps,
            "response_format": "b64_json",
        },
        out,
    )


def image_inpaint(
    image_file: Path = typer.Argument(..., exists=True, help="Source image."),
    prompt: str = typer.Argument(..., help="What to fill the masked region with."),
    out: Path = typer.Option(..., "--out", "-o", help="Output image file."),
    mask: Path = typer.Option(
        None, "--mask", help="Mask image (white=fill, black=keep)."
    ),
    url: str = _URL,
    model: str = typer.Option("Z-Image-Turbo-MLX-4bit", "--model", "-m"),
):
    """Inpaint a masked region (POST /v1/images/inpaint)."""
    body = {
        "image": _b64file(image_file),
        "prompt": prompt,
        "model": model,
        "response_format": "b64_json",
    }
    if mask:
        body["mask"] = _b64file(mask)
    _img_result(url, "/v1/images/inpaint", body, out)


def image_controlnet(
    image_file: Path = typer.Argument(..., exists=True, help="Control image."),
    condition_type: str = typer.Argument(..., help="Condition, e.g. canny/depth/pose."),
    prompt: str = typer.Argument(..., help="Generation prompt."),
    out: Path = typer.Option(..., "--out", "-o", help="Output image file."),
    url: str = _URL,
    model: str = typer.Option("Z-Image-Turbo-MLX-4bit", "--model", "-m"),
):
    """ControlNet-guided generation (POST /v1/images/controlnet)."""
    _img_result(
        url,
        "/v1/images/controlnet",
        {
            "prompt": prompt,
            "image": _b64file(image_file),
            "condition_type": condition_type,
            "model": model,
            "response_format": "b64_json",
        },
        out,
    )


def image_depth(
    depth_file: Path = typer.Argument(..., exists=True, help="Depth-map image."),
    prompt: str = typer.Argument(..., help="Generation prompt."),
    out: Path = typer.Option(..., "--out", "-o", help="Output image file."),
    url: str = _URL,
    model: str = typer.Option("Z-Image-Turbo-MLX-4bit", "--model", "-m"),
):
    """Depth-guided generation (POST /v1/images/depth-guided)."""
    _img_result(
        url,
        "/v1/images/depth-guided",
        {
            "prompt": prompt,
            "depth_image": _b64file(depth_file),
            "model": model,
            "response_format": "b64_json",
        },
        out,
    )


def audio_enhance(
    audio_file: Path = typer.Argument(..., exists=True, help="Input audio."),
    out: Path = typer.Option(..., "--out", "-o", help="Output audio (.wav)."),
    url: str = _URL,
    method: str = typer.Option(
        None, "--method", help="spectral_gating / deep_filter / minimal."
    ),
):
    """Enhance audio — denoise / dereverb (POST /v1/audio/speech-to-speech/enhance)."""
    body = {"audio": _b64file(audio_file), "response_format": "wav"}
    if method:
        body["method"] = method
    _audio_result(url, "/v1/audio/speech-to-speech/enhance", body, out)


def audio_separate(
    audio_file: Path = typer.Argument(..., exists=True, help="Input audio."),
    out: Path = typer.Option(..., "--out", "-o", help="Output audio (.wav)."),
    url: str = _URL,
    source: str = typer.Option(None, "--source", help="Text of the source to isolate."),
):
    """Isolate a source from audio (POST /v1/audio/speech-to-speech/separate)."""
    body = {"audio": _b64file(audio_file), "response_format": "wav"}
    if source:
        body["source_text"] = source
    _audio_result(url, "/v1/audio/speech-to-speech/separate", body, out)


def audio_transform(
    audio_file: Path = typer.Argument(..., exists=True, help="Input audio."),
    out: Path = typer.Option(..., "--out", "-o", help="Output audio (.wav)."),
    url: str = _URL,
    pitch: float = typer.Option(0.0, "--pitch", help="Pitch shift (semitones)."),
    formant: float = typer.Option(1.0, "--formant", help="Formant ratio."),
):
    """Transform a voice — pitch/formant (POST /v1/audio/speech-to-speech/transform)."""
    _audio_result(
        url,
        "/v1/audio/speech-to-speech/transform",
        {
            "audio": _b64file(audio_file),
            "response_format": "wav",
            "pitch_shift": pitch,
            "formant_ratio": formant,
        },
        out,
    )


def voice_pipeline(
    audio_file: Path = typer.Argument(..., exists=True, help="Input speech."),
    out: Path = typer.Option(..., "--out", "-o", help="Output speech file (.wav)."),
    url: str = _URL,
    model: str = typer.Option(
        ..., "--model", "-m", help="LLM model for the middle step (required)."
    ),
    voice: str = typer.Option(None, "--voice", help="Reply voice."),
):
    """Speech-to-speech via LLM: transcribe → LLM → TTS (POST /v1/audio/voice-pipeline)."""
    data = {"llm_model": model}
    if voice:
        data["voice"] = voice
    with open(audio_file, "rb") as f:
        resp = _post(
            url,
            "/v1/audio/voice-pipeline",
            files={"file": (audio_file.name, f, "application/octet-stream")},
            data=data,
            timeout=600,
        )
    _write(out, _binary(resp, "audio"))
    emit(
        {"file": str(out), "bytes": out.stat().st_size},
        human=lambda: console.print(
            f"wrote [bold]{out}[/] ({out.stat().st_size} bytes)"
        ),
    )


def voices(url: str = _URL):
    """List available TTS voices for `speak` (GET /v1/audio/voices)."""
    resp = _get(url, "/v1/audio/voices")
    d = _body(resp)
    ids = [v.get("id") for v in d.get("data", [])]
    emit({"voices": ids}, human=lambda: console.print(", ".join(ids) or "(none)"))


def register(app: typer.Typer) -> None:
    """Attach the inference commands as top-level `yunshu` commands."""
    for fn, name in (
        (complete, "complete"),
        (embed, "embed"),
        (tokenize, "tokenize"),
        (rerank, "rerank"),
        (detokenize, "detokenize"),
        (classify, "classify"),
        (score, "score"),
        (transcribe, "transcribe"),
        (speak, "speak"),
        (ocr, "ocr"),
        (image, "image"),
        (image_edit, "image-edit"),
        (image_variations, "image-variations"),
        (image_inpaint, "image-inpaint"),
        (image_controlnet, "image-controlnet"),
        (image_depth, "image-depth"),
        (video, "video"),
        (audio_enhance, "audio-enhance"),
        (audio_separate, "audio-separate"),
        (audio_transform, "audio-transform"),
        (voice_pipeline, "voice-pipeline"),
        (voices, "voices"),
        (cancel, "cancel"),
    ):
        app.command(name)(fn)
