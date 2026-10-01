"""A scripted engine behind the real routers, for client wire-contract tests (CPU only).

``ScriptedEngine`` is a ``BatchedEngine`` (so every router takes its production batched branch)
whose generation methods replay a ``Script``: one token per piece, cumulative token counts on
every streamed output, ``max_tokens`` truncation, user ``stop`` sequences, an optional
terminal error after N pieces. The same script drives ``chat`` / ``generate`` (non-stream) and
``stream_chat`` / ``stream_generate`` / ``generate_stream`` (stream), so a dialect that counts
differently between its stream and non-stream path shows up as a mismatch on identical input.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput


@dataclass
class Script:
    # One entry per generated token: text, or (text, state) with state "reasoning" for thinking.
    pieces: list = field(default_factory=lambda: ["Hello", " there", "!"])
    prompt_tokens: int = 7
    cached_tokens: int = 0
    finish_reason: str = "stop"
    # Terminal error (finish_reason "error") after this many pieces; None for a clean run.
    error_after: int | None = None
    error: str = "scripted engine failure"
    # Pause (seconds) before each piece, so a cancel can land mid-generation.
    delay: float = 0.0


class _Tok:
    """Whitespace tokenizer: deterministic, and enough for prompt counting."""

    def encode(self, text, *a, **k):
        return text.split()

    def apply_chat_template(self, *a, **k):
        raise RuntimeError("no template in test")


def _piece(p):
    return (p, "normal") if isinstance(p, str) else (p[0], p[1])


class ScriptedEngine(BatchedEngine):
    def __init__(self, script: Script | None = None):
        super().__init__()
        self._model = object()
        self._loaded = True
        self._running = True
        self._tokenizer = _Tok()
        self.model_name = "scripted"
        self.script = script or Script()
        self.calls: list[dict] = []
        self.cancelled = 0  # streams that saw their cancel event set
        self.stopped_early = 0  # streams abandoned before their last output

    def is_running(self) -> bool:  # noqa: D401
        return True

    # ── shared replay ────────────────────────────────────────────────────────────
    @staticmethod
    def _continue_after(pieces, messages):
        """A real engine given a trailing assistant turn (a forced tool-call prefill) continues
        it instead of repeating it: drop the scripted text the prefill already covers."""
        last = (messages or [None])[-1]
        prefill = last.get("content") if isinstance(last, dict) else None
        if not (
            isinstance(prefill, str) and last.get("role") == "assistant" and prefill
        ):
            return pieces
        full = "".join(p[0] for p in pieces)
        if not full.startswith(prefill):
            return pieces
        skip = len(prefill)
        out = []
        for t, state in pieces:
            if skip >= len(t):
                skip -= len(t)
                continue
            out.append((t[skip:], state))
            skip = 0
        return out

    def _steps(self, max_tokens, stop, messages=None):
        sc = self.script
        pieces = self._continue_after([_piece(p) for p in sc.pieces], messages)
        if sc.error_after is not None:
            pieces = pieces[: sc.error_after]
        finish = sc.finish_reason
        if max_tokens is not None and 0 <= max_tokens < len(pieces):
            pieces = pieces[:max_tokens]
            finish = "length"
        out = []
        text = ""
        reasoning = 0
        stopped = False
        for i, (t, state) in enumerate(pieces, 1):
            if state == "reasoning":
                reasoning += 1
            text += t
            visible = text
            hit = next((s for s in (stop or []) if s and s in visible), None)
            if hit:
                cut = visible.index(hit)
                new = t[: max(0, len(t) - (len(visible) - cut))]
                out.append((new, state, i, reasoning, True))
                stopped = True
                break
            out.append((t, state, i, reasoning, False))
        return out, ("stop" if stopped else finish), stopped

    def _record(self, kind, kw):
        self.calls.append({"kind": kind, **kw})

    async def _stream(self, max_tokens, stop, messages=None, cancel_event=None, **kw):
        sc = self.script
        steps, finish, stopped = self._steps(max_tokens, stop, messages)
        acc = ""
        completed = False
        try:
            for n, (t, state, ct, rt, _last_stop) in enumerate(steps, 1):
                if sc.delay:
                    await asyncio.sleep(sc.delay)
                if cancel_event is not None and cancel_event.is_set():
                    self.cancelled += 1
                    return
                acc += t
                last = n == len(steps) and sc.error_after is None
                yield GenerationOutput(
                    text=acc,
                    new_text=t,
                    prompt_tokens=sc.prompt_tokens,
                    completion_tokens=ct,
                    reasoning_tokens=rt,
                    cached_tokens=sc.cached_tokens,
                    finished=last,
                    finish_reason=finish if last else None,
                    stopped_by_stop_sequence=stopped and last,
                    current_state=state,
                )
            if sc.error_after is not None:
                yield GenerationOutput(
                    text=acc,
                    new_text="",
                    prompt_tokens=sc.prompt_tokens,
                    completion_tokens=len(steps),
                    finished=True,
                    finish_reason="error",
                    error=sc.error,
                )
            elif not steps:
                yield GenerationOutput(
                    prompt_tokens=sc.prompt_tokens,
                    finished=True,
                    finish_reason=finish,
                    cached_tokens=sc.cached_tokens,
                )
            completed = True
        finally:
            if not completed:
                self.stopped_early += 1

    async def _whole(self, max_tokens, stop, **kw):
        last = None
        text = ""
        async for o in self._stream(max_tokens, stop, **kw):
            text = o.text
            last = o
        if last is not None and last.error:
            raise RuntimeError(last.error)
        assert last is not None
        last.text = text
        return last

    # ── BatchedEngine surface used by the routers ────────────────────────────────
    async def chat(self, messages, max_tokens=256, stop=None, **kw):
        self._record(
            "chat", {"messages": messages, "max_tokens": max_tokens, "stop": stop, **kw}
        )
        return await self._whole(max_tokens, stop, messages=messages, **kw)

    async def generate(self, prompt=None, max_tokens=256, stop=None, **kw):
        self._record(
            "generate", {"prompt": prompt, "max_tokens": max_tokens, "stop": stop, **kw}
        )
        return await self._whole(max_tokens, stop, **kw)

    async def stream_chat(self, messages, max_tokens=256, stop=None, **kw):
        self._record(
            "stream_chat",
            {"messages": messages, "max_tokens": max_tokens, "stop": stop, **kw},
        )
        async for o in self._stream(max_tokens, stop, messages=messages, **kw):
            yield o

    async def stream_generate(self, prompt=None, max_tokens=256, stop=None, **kw):
        self._record(
            "stream_generate",
            {"prompt": prompt, "max_tokens": max_tokens, "stop": stop, **kw},
        )
        async for o in self._stream(max_tokens, stop, **kw):
            yield o

    generate_stream = stream_generate


def install(monkeypatch, script: Script | None = None):
    """Serve ``ScriptedEngine`` as the single loaded engine of a fresh gateway app.

    Returns ``(client, engine)``; ``client`` is a Starlette ``TestClient`` (an ``httpx.Client``,
    so the official SDKs take it as ``http_client``). Ollama's routes call the chat route over
    loopback HTTP; here that loopback goes to the same app in-process.
    """
    import httpx
    from fastapi.testclient import TestClient

    from yunshu_gateway import engine as engine_mod
    from yunshu_gateway.main import create_app
    from yunshu_gateway.routers import ollama as ollama_router

    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    eng = ScriptedEngine(script)
    monkeypatch.setattr(engine_mod, "_engine", eng)
    monkeypatch.setattr(engine_mod, "_model_manager", None)
    app = create_app()
    monkeypatch.setattr(
        ollama_router,
        "_client",
        lambda request: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            timeout=None,
        ),
    )
    return TestClient(app), eng
