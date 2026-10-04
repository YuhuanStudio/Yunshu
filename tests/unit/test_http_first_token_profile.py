"""First-visible tracing uses visible SSE content and never raw role/empty deltas."""

import json

import pytest
from scripts.research.http_first_token_profile import Trace, install


def test_trace_wrap_keeps_result_and_records_only_before_visible(tmp_path):
    class Example:
        def run(self, value):
            return value + 1

    trace = Trace(tmp_path / "trace.jsonl")
    trace.wrap(Example, "run")
    trace.begin()
    assert Example().run(3) == 4
    trace.first_visible = 1.0
    assert Example().run(9) == 10
    row = trace.finish()
    assert len(row["events"]) == 1
    assert row["events"][0]["duration_ms"] >= 0
    assert row["first_visible_ms"] == 1.0
    assert not row["synchronized_layers"]


@pytest.mark.asyncio
async def test_http_first_visible_excludes_role_and_empty_sse(monkeypatch, tmp_path):
    from fastapi import FastAPI

    from yunshu_engine.vlm_engine import VLMEngine

    for name in (
        "load",
        "_runner_input",
        "_tokenize_with_cache",
        "_format_prompt",
        "_resolve_prompt_cache_plan",
    ):
        monkeypatch.setattr(VLMEngine, name, getattr(VLMEngine, name))

    async def original(app, scope, receive, send):
        for delta in ({"role": "assistant"}, {"content": ""}, {"content": "Hi"}):
            await send(
                dict(
                    type="http.response.body",
                    body=(
                        "data: "
                        + json.dumps(dict(choices=[dict(delta=delta)]))
                        + "\n\n"
                    ).encode(),
                    more_body=True,
                )
            )
        await send(
            dict(type="http.response.body", body=b"data: [DONE]\n\n", more_body=False)
        )

    monkeypatch.setattr(FastAPI, "__call__", original)
    path = tmp_path / "http.jsonl"
    trace = install(path)
    sent = []

    async def send(message):
        sent.append(message)

    await FastAPI()({"path": "/v1/chat/completions"}, None, send)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["success"]
    assert len(rows[0]["events"]) == 3
    assert len(sent) == 4
    assert trace.started is None
