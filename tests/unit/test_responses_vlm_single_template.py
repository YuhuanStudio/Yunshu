"""/v1/responses on a VLMEngine hands the engine the message list, not a templated string.

VLMEngine templates its own messages; a string prompt becomes ONE user message. The router
used to render the chat template itself for every non-BatchedEngine, so on the 27B (Codex)
the whole ChatML transcript ended up inside a user turn, templated twice.
"""

import os

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway.routers.responses import _engine_templates_itself


class _Tok:
    chat_template = "x"

    def apply_chat_template(self, *a, **k):
        return "<|im_start|>RENDERED"  # what the router used to hand the engine


@pytest.fixture
def _vlm():
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.vlm_engine import VLMEngine
    from yunshu_gateway.engine import set_engine

    engine = VLMEngine("/models/Qwen3.8-27B")
    engine._model = object()
    engine._running = True
    engine._tokenizer = _Tok()
    yield engine, set_engine
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def test_only_engines_that_template_themselves_are_skipped(_vlm):
    from yunshu_engine.batched_engine import BatchedEngine

    assert _engine_templates_itself(_vlm[0])
    assert not _engine_templates_itself(BatchedEngine())
    assert not _engine_templates_itself(object())


@pytest.mark.parametrize("stream", [False, True])
def test_responses_pass_messages_to_a_vlm_engine(_vlm, monkeypatch, stream):
    from yunshu_engine.request import RequestOutput
    from yunshu_gateway.main import create_app

    engine, set_engine = _vlm
    seen = {}

    async def _generate(prompt=None, **kw):
        seen["prompt"] = prompt
        return {"text": "ok", "finish_reason": "stop"}

    async def _stream(prompt=None, **kw):
        seen["prompt"] = prompt
        yield RequestOutput(
            request_id="r",
            new_text="ok",
            output_text="ok",
            finished=True,
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=1,
        )

    monkeypatch.setattr(engine, "generate", _generate)
    monkeypatch.setattr(engine, "generate_stream", _stream)
    set_engine(engine)
    client = TestClient(create_app(), raise_server_exceptions=False)
    client.post(
        "/v1/responses",
        json={
            "model": "any",
            "stream": stream,
            "instructions": "SYS",
            "input": [
                {"type": "message", "role": "developer", "content": "DEV"},
                {"type": "message", "role": "user", "content": "hello"},
            ],
        },
    )
    prompt = seen.get("prompt")
    assert isinstance(prompt, list), f"engine got {type(prompt)}: {prompt!r}"
    assert [m["role"] for m in prompt][-1] == "user"
    assert any("SYS" in str(m["content"]) for m in prompt)
