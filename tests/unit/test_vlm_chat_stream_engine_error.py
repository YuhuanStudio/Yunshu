"""An engine failure mid-stream on the VLM chat path is an error event, not a clean "stop"."""

import asyncio
import json

from yunshu_engine.request import RequestOutput
from yunshu_gateway.routers import chat
from yunshu_gateway.routers.chat import ChatCompletionRequest, _stream_vlm_response


def test_engine_error_is_an_error_event_not_stop(monkeypatch):
    monkeypatch.setattr(chat, "_record_metrics", lambda *args: None)

    class Engine:
        _tokenizer = None

        async def generate_stream(self, **kwargs):
            yield RequestOutput(request_id="t", new_text="par", prompt_tokens=1)
            yield RequestOutput(
                request_id="t",
                new_text="",
                finish_reason="error",
                finished=True,
                error="runner step failed",
            )

    req = ChatCompletionRequest(
        model="test", messages=[{"role": "user", "content": "q"}], stream=True
    )

    async def collect():
        return [c async for c in _stream_vlm_response(Engine(), [], req, "test", None)]

    text = b"".join(asyncio.run(collect())).decode()
    records = [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert any(
        "error" in r and "runner step failed" in r["error"]["message"] for r in records
    )
    assert not any(
        c.get("finish_reason") == "stop" for r in records for c in r.get("choices", [])
    )
