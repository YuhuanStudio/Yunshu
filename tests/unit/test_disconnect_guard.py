"""the non-streaming disconnect guard (chat.py's run_with_disconnect_guard) was
never propagated to completions / responses / anthropic. Those paths registered a
cancel_event and passed it to the engine but never polled is_disconnected(), so a client
that dropped mid-request ran to max_tokens / the 300s timeout, head-of-line-blocking every
subsequent request on the serial max_workers=1 executor. Each non-streaming engine await is
now wrapped so a disconnect SETS the cancel_event and the decode loop stops.
"""

from __future__ import annotations

import inspect


def test_completions_nonstream_uses_disconnect_guard():
    from yunshu_gateway.routers import completions

    src = inspect.getsource(completions)
    assert "run_with_disconnect_guard(\n                    request, _gen_one" in src
    # None return == disconnected → stop the generation loop
    assert "if _r is None:" in src


def test_responses_nonstream_uses_disconnect_guard():
    from yunshu_gateway.routers import responses

    src = inspect.getsource(responses)
    # both batched (engine.chat) and non-batched (engine.generate) awaits are wrapped
    assert "run_with_disconnect_guard(\n                        request,\n                        engine.chat(" in src
    assert "run_with_disconnect_guard(\n                        request,\n                        engine.generate(" in src
    # the disconnect (HTTPException 499) is re-raised, not swallowed by the broad except
    assert "except HTTPException:\n        raise" in src


def test_anthropic_nonstream_uses_disconnect_guard():
    from yunshu_gateway.routers import anthropic

    src = inspect.getsource(anthropic)
    # both helpers thread `request` and guard the engine coroutine
    assert src.count("run_with_disconnect_guard(\n                request, _chat_coro") >= 1
    assert src.count("run_with_disconnect_guard(\n                request, _gen_coro") >= 1
    # the helpers accept request and the call sites pass it
    assert "request=request," in src
    assert src.count("except HTTPException:\n        raise") >= 2
