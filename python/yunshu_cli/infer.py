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
    text: str = typer.Argument(..., help="Text to count tokens for."),
    url: str = _URL,
    model: str = _MODEL,
):
    """Count tokens (POST /v1/token_count)."""
    resp = _post(url, "/v1/token_count", json={"model": model, "prompt": text})
    d = _body(resp)
    count = d.get("token_count", d.get("count"))
    emit(d, human=lambda: console.print(f"tokens: [bold]{count}[/]"))


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


def register(app: typer.Typer) -> None:
    """Attach the inference commands as top-level `yunshu` commands."""
    for fn, name in (
        (complete, "complete"),
        (embed, "embed"),
        (tokenize, "tokenize"),
        (rerank, "rerank"),
        (transcribe, "transcribe"),
        (speak, "speak"),
        (ocr, "ocr"),
        (image, "image"),
    ):
        app.command(name)(fn)
