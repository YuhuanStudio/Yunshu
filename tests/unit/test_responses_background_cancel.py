"""A background streamed response cancelled mid-generation stays `cancelled`: the stream ending
afterwards must not overwrite the status the cancel endpoint stored (found by the real-server check)."""

import asyncio
import json
import os


async def _drive():
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    engine = BatchedEngine()
    engine._model = object()
    engine._loaded = True
    engine._running = True
    engine.model_name = "m"
    release = asyncio.Event()

    async def slow(*args, **kwargs):
        yield GenerationOutput(
            text="He",
            new_text="He",
            prompt_tokens=3,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
        )
        await release.wait()  # the generation keeps running after the cancel
        yield GenerationOutput(
            text="Hello",
            new_text="llo",
            prompt_tokens=3,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
        )

    engine.stream_chat = slow
    set_engine(engine)
    from starlette.requests import Request

    from yunshu_gateway.routers import responses as R  # noqa: N812

    app = create_app()

    def make_request():
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/responses",
                "headers": [],
                "query_string": b"",
                "app": app,
                "state": {},
                "client": ("127.0.0.1", 1),
                "server": ("t", 80),
                "scheme": "http",
            }
        )

    try:
        req = R.ResponsesRequest(
            model="m", input="hi", background=True, stream=True, max_output_tokens=16
        )
        resp = await R.create_response(req, make_request())
        names, rid, cancelled = [], None, False
        async for chunk in resp.body_iterator:
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            for block in text.split("\n\n"):
                for line in block.splitlines():
                    if line.startswith("event:"):
                        names.append(line[6:].strip())
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        d = json.loads(line[5:])
                        rid = rid or (d.get("response") or {}).get("id")
                        if (
                            d.get("type") == "response.output_text.delta"
                            and not cancelled
                        ):
                            cancelled = True
                            got = await R.cancel_response(rid, make_request())
                            assert json.loads(got.body)["status"] == "cancelled"
                            release.set()
        await asyncio.sleep(0.5)  # the detached runner may finish after the follower
        final = R._public_stored(R._get_stored_response(rid))
        return final, names
    finally:
        set_engine(None)
        os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def test_cancelled_background_stream_stays_cancelled():
    final, names = asyncio.run(_drive())
    assert final["status"] == "cancelled", final["status"]
    assert names  # the stream was live when it was cancelled
