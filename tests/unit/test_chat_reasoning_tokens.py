"""Regression tests for non-streaming reasoning-token accounting in chat.py.

Guards two fixes :
  1. completion_tokens must NOT have reasoning_tokens added on top (the engine's
     n_tok already includes reasoning) — neither in the user-facing usage nor in
     the trace record.
  2. When the response carries reasoning_content but the engine's reasoning
     parser returned 0, the reasoning_tokens detail is reconciled from the
     extracted thinking text (capped at completion_tokens).
"""

import asyncio
import json
from unittest.mock import patch

from yunshu_gateway.routers.chat import (
    ChatCompletionRequest,
    ChatMessage,
    create_chat_completion,
)


class _FakeTokenizer:
    def encode(self, text, *a, **k):
        # 1 token per whitespace-separated word — deterministic and simple.
        return text.split()

    def apply_chat_template(self, *a, **k):
        raise RuntimeError("no template in test")  # forces the heuristic path


class _FakeState:
    def __init__(self, text, prompt_tokens, completion_tokens, reasoning_tokens):
        self.generated_text = text
        self.prompt_token_count = prompt_tokens
        self.completion_token_count = completion_tokens
        self.finish_reason = "stop"
        self.reasoning_tokens = reasoning_tokens
        self.cached_tokens = 0
        self.logprobs = None


class _FakeEngine:
    """Non-batched engine (so chat.py takes the generate() path)."""

    is_loaded = True

    def __init__(self, state):
        self._tokenizer = _FakeTokenizer()
        self._state = state

    def resolve_model_id(self, model):
        return True

    async def generate(self, *a, **k):
        return self._state


def _make_request(model="test", max_tokens=64):
    mock_request = type("R", (), {})()
    mock_request.state = type("S", (), {})()
    mock_request.state.rbac_key = None
    mock_request.state.request_id = "test"
    mock_request.app = type("A", (), {})()
    mock_request.app.state = type("AS", (), {})()
    return mock_request


def _run(engine, req, mock_request):
    with (
        patch("yunshu_gateway.routers.chat.get_engine", return_value=engine),
        patch("yunshu_gateway.routers.chat.get_model_manager", return_value=None),
        patch("yunshu_gateway.routers.chat.validate_context_window", return_value=None),
    ):
        result = asyncio.new_event_loop().run_until_complete(
            create_chat_completion(req, mock_request)
        )
    return json.loads(result.body)


def test_reasoning_tokens_reconciled_when_engine_returns_zero():
    # Engine emitted a <think> block (3 words) but reported reasoning_tokens=0.
    # completion_token_count=20 already includes those tokens.
    state = _FakeState(
        "<think>one two three</think>The answer is 42.",
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_tokens=0,
    )
    req = ChatCompletionRequest(
        model="test", messages=[ChatMessage(role="user", content="hi")], max_tokens=64
    )
    body = _run(_FakeEngine(state), req, _make_request())

    usage = body["usage"]
    # No double-count: completion stays the engine count, total = prompt + completion.
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    # Reconciled detail: 3 reasoning words -> 3 tokens, and <= completion.
    rt = usage["completion_tokens_details"]["reasoning_tokens"]
    assert rt == 3
    assert rt <= usage["completion_tokens"]
    assert body["choices"][0]["message"].get("reasoning_content")


def test_no_reasoning_detail_when_no_thinking():
    state = _FakeState(
        "Just a plain answer.",
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=0,
    )
    req = ChatCompletionRequest(
        model="test", messages=[ChatMessage(role="user", content="hi")], max_tokens=64
    )
    body = _run(_FakeEngine(state), req, _make_request())

    usage = body["usage"]
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert "completion_tokens_details" not in usage


def test_reconciled_reasoning_capped_at_completion():
    # Thinking text has more words than completion_tokens; must cap at completion.
    state = _FakeState(
        "<think>a b c d e f g h</think>x",
        prompt_tokens=4,
        completion_tokens=3,
        reasoning_tokens=0,
    )
    req = ChatCompletionRequest(
        model="test", messages=[ChatMessage(role="user", content="hi")], max_tokens=64
    )
    body = _run(_FakeEngine(state), req, _make_request())

    usage = body["usage"]
    rt = usage["completion_tokens_details"]["reasoning_tokens"]
    assert rt == 3  # capped at completion_tokens, not 8
    assert rt <= usage["completion_tokens"]
