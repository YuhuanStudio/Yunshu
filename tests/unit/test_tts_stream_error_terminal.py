"""(MED): a mid-synthesis failure in /v1/audio/speech streaming silently truncated
the SSE stream with NO terminal error/done event.

audio_engine.synthesize_stream enqueued a bare None on exception; its drain loop did
`if chunk is None: break` and ended WITHOUT yielding an is_final marker, so the route's
is_final/[DONE] block never ran → the client got a truncated audio stream indistinguishable
from a successful short clip (worst when the LAST segment errored → no done/[DONE] at all).

Fix: the engine enqueues an ERROR-tagged terminal chunk (is_final=True, error=...) instead of
None; the route detects chunk.get("error") and emits a proper error event + [DONE] (checked
before is_final, since the error chunk carries both). The realtime consumer is unaffected
(empty-audio + is_final → clean turn end via its finally).
"""

from __future__ import annotations

import inspect

from yunshu_engine import audio_engine
from yunshu_gateway.routers import audio as audio_router


def test_engine_enqueues_error_chunk_not_bare_none():
    src = inspect.getsource(audio_engine.TTSEngine.synthesize_stream)
    # strip comments so we match real code, not the explanatory prose
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the except path must build an error-tagged terminal chunk and NOT enqueue a bare None
    assert '"error": str(e)' in code and '"is_final": True' in code
    assert "put_nowait(None)" not in code, "bare-None error sentinel must be gone"


def test_route_surfaces_error_chunk_with_done_before_is_final():
    src = inspect.getsource(audio_router.stream_speech)
    err = src.index('chunk.get("error")')
    final = src.index('chunk.get("is_final")', 0)
    # the error check must come BEFORE the is_final check (the error chunk carries both)
    assert err < final, (
        "chunk.get('error') must be handled before chunk.get('is_final')"
    )
    # and it must emit a terminal error event + [DONE] and stop (within the error handler,
    # before the next is_final-gated block — search a window large enough to clear the comment)
    region = src[err : err + 900]
    assert "'type': 'error'" in region or '"type": "error"' in region
    assert "[DONE]" in region
    assert "return" in region
    # the yields must precede the is_final handling that follows this block
    assert src.index("[DONE]", err) < src.index('if chunk.get("is_final")', err)
