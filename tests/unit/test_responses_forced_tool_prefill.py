"""Named Responses tools must leave JSON arguments for the model to generate."""

import json

import pytest
from fastapi.testclient import TestClient

from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput
from yunshu_engine.tool_format import INJECTED_JSON, JSON_MESSAGE
from yunshu_gateway.engine import set_engine


@pytest.mark.parametrize("choice", ["required", {"type": "function", "name": "shell"}])
def test_forced_response_prefills_only_opening_marker(monkeypatch, choice):
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    engine = BatchedEngine()
    # Generation is replaced below; this fixture tests prompt construction only.
    monkeypatch.setattr(engine, "validate_forced_tools", lambda *a: None)
    engine._model = object()
    engine._loaded = True
    engine._running = True
    engine.model_name = "forced-response-test"
    engine._yunshu_tool_formats = (INJECTED_JSON, JSON_MESSAGE)
    arguments = {"command": "printf rapidmlx", "timeout": 1000}
    seen = []

    async def generate(messages, **kwargs):
        # A prefill into {name, arguments:{ led Qwen to immediately close {}.
        # Match the proven Chat/Anthropic opening-marker contract instead.
        assert messages[-1] == {"role": "assistant", "content": "<tool_call>\n"}
        assert "shell" in messages[0]["content"]
        seen.append(messages)
        text = json.dumps({"name": "shell", "arguments": arguments}) + "</tool_call>"
        return GenerationOutput(
            text=text,
            new_text=text,
            prompt_tokens=20,
            completion_tokens=30,
            finished=True,
            finish_reason="stop",
        )

    monkeypatch.setattr(engine, "chat", generate)
    from yunshu_gateway.main import create_app

    set_engine(engine)
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": engine.model_name,
                    "input": "Call shell with command printf rapidmlx and timeout 1000.",
                    "max_output_tokens": 256,
                    "tool_choice": choice,
                    "tools": [
                        {
                            "type": "function",
                            "name": "shell",
                            "description": "Run a command",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                    "timeout": {"type": "integer"},
                                },
                                "required": ["command", "timeout"],
                            },
                        }
                    ],
                },
            )
        assert response.status_code == 200, response.text
        assert seen
        calls = [
            item
            for item in response.json()["output"]
            if item["type"] == "function_call"
        ]
        assert len(calls) == 1 and calls[0]["name"] == "shell"
        assert json.loads(calls[0]["arguments"]) == arguments
    finally:
        set_engine(None)
