"""(last deferred real-but-narrow item, from the chat-template hunt): the Responses
API's non-batched (legacy Engine) path re-implemented chat-template application but OMITTED
the engine's developer/function role remap and _normalize_messages_for_chat_template — so a
developer-role or tool-use Responses request on the legacy Engine was passed raw to the
template (the BatchedEngine default path applies all of this). Both non-batched sites
(non-stream + streaming) now apply the engine's role-remap (developer→system,
function→tool) + normalize before the family adapter. (Assistant-prefill
continue_final_message remains a documented gap of the deprecated path.)
"""

from __future__ import annotations

import inspect

from yunshu_engine.batched_engine import BatchedEngine
from yunshu_gateway.routers import (
    responses as R,  # noqa: N812  # intentional short module alias
)


def test_normalize_helper_is_reusable():
    # the engine's normalizer (used by the fix) is an importable static method
    out = BatchedEngine._normalize_messages_for_chat_template(
        [{"role": "user", "content": "hi"}]
    )
    assert isinstance(out, list) and out[0]["role"] == "user"


def test_both_nonbatched_sites_remap_and_normalize():
    src = inspect.getsource(R)
    # developer→system / function→tool remap appears at BOTH non-batched sites
    # (ruff may split the ternary across lines; check the condition itself)
    assert src.count('_m.get("role") in ("developer", "function")') == 2
    # and the engine normalizer is invoked at both
    assert src.count("_BE._normalize_messages_for_chat_template(_msgs)") == 2
    # the adapter now consumes the normalized messages, not the raw ones
    assert "_adapted = adapt_messages" in src
