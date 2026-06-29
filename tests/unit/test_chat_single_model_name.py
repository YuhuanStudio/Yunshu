"""Single-model mode serves the loaded model under ANY requested name.

The quickstart (and most local-server clients) send model="local". In
single-model mode (YUNSHU_MODEL set, no model manager) the global engine is the
only model, so the chat endpoint must serve it regardless of the requested name
— matching Ollama / LM Studio / llama.cpp. Previously a name that didn't echo
the on-disk model id 404'd, breaking every copy-pasted quickstart request.
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
        return text.split()

    def apply_chat_template(self, *a, **k):
        raise RuntimeError("no template in test")


class _FakeState:
    def __init__(self):
        self.generated_text = "Hello!"
        self.prompt_token_count = 3
        self.completion_token_count = 1
        self.finish_reason = "stop"
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.logprobs = None


class _NameMismatchEngine:
    """Loaded single-model engine whose id does NOT match the requested name
    (resolve_model_id → False), e.g. the on-disk path vs. model='local'."""

    is_loaded = True

    def __init__(self):
        self._tokenizer = _FakeTokenizer()

    def resolve_model_id(self, model):
        return False  # the requested name never matches the on-disk id

    async def generate(self, *a, **k):
        return _FakeState()


def _mock_request():
    r = type("R", (), {})()
    r.state = type("S", (), {})()
    r.state.rbac_key = None
    r.state.request_id = "test"
    r.app = type("A", (), {})()
    r.app.state = type("AS", (), {})()
    return r


def _run(model_name):
    req = ChatCompletionRequest(
        model=model_name, messages=[ChatMessage(role="user", content="hi")]
    )
    with (
        patch(
            "yunshu_gateway.routers.chat.get_engine",
            return_value=_NameMismatchEngine(),
        ),
        patch("yunshu_gateway.routers.chat.get_model_manager", return_value=None),
        patch("yunshu_gateway.routers.chat.validate_context_window", return_value=None),
    ):
        result = asyncio.new_event_loop().run_until_complete(
            create_chat_completion(req, _mock_request())
        )
    return json.loads(result.body)


def test_single_model_serves_model_local():
    body = _run("local")  # the quickstart's name — must NOT 404
    assert body["choices"][0]["message"]["content"] == "Hello!"


def test_single_model_serves_arbitrary_name():
    body = _run("gpt-4o")  # any name resolves to the one loaded model
    assert body["choices"][0]["message"]["content"] == "Hello!"
