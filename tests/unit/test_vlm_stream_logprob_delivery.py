"""Accepted target reports survive detokenizer and tool-parser buffering."""

import asyncio
import json
import queue

import pytest

from yunshu_engine.request import RequestOutput
from yunshu_engine.vlm_engine import VLMEngine
from yunshu_gateway.routers.chat import ChatCompletionRequest, _stream_vlm_response


def _lp(n):
    return {
        "token": f"t{n}",
        "logprob": -float(n),
        "top_logprobs": [
            {"token": f"t{n}", "logprob": -float(n)},
            {"token": "other", "logprob": -10.0},
        ],
    }


@pytest.mark.parametrize("route", ["tool", "reasoning", "text"])
def test_vlm_sse_reports_each_input_once_including_buffered_and_final_tokens(
    route, monkeypatch
):
    from yunshu_gateway.routers import chat

    monkeypatch.setattr(chat, "_record_metrics", lambda *args: None)
    texts = (
        ["", "<tool_", 'call>{"name":"f","arguments":{"x":1}}</tool_call>', ""]
        if route == "tool"
        else ["", "thinking" if route == "reasoning" else "hello", ""]
    )

    class Engine:
        _tokenizer = None

        async def generate_stream(self, **kwargs):
            for n, text in enumerate(texts):
                last = n == len(texts) - 1
                yield RequestOutput(
                    request_id="test",
                    new_text=text,
                    logprobs=[_lp(n)],
                    current_state="reasoning" if route == "reasoning" else "normal",
                    finish_reason="stop" if last else None,
                    finished=last,
                    prompt_tokens=1,
                    completion_tokens=n + 1,
                )

    tools = (
        [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                    },
                },
            }
        ]
        if route == "tool"
        else None
    )
    req = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "q"}],
        tools=tools,
        stream=True,
        logprobs=True,
        top_logprobs=1,
    )

    async def collect():
        return [
            chunk
            async for chunk in _stream_vlm_response(Engine(), [], req, "test", None)
        ]

    chunks = asyncio.run(collect())
    choices = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                record = json.loads(line[6:])
                assert "error" not in record
                choices.extend(record.get("choices", []))
    reports = [
        lp
        for choice in choices
        for lp in (choice.get("logprobs") or {}).get("content", [])
    ]
    assert [lp["token"] for lp in reports] == [f"t{n}" for n in range(len(texts))]
    assert [lp["logprob"] for lp in reports] == [-float(n) for n in range(len(texts))]
    assert all(len(lp["top_logprobs"]) == 1 for lp in reports)
    if route == "tool":
        deltas = [choice["delta"] for choice in choices]
        assert not any("<tool" in (d.get("content") or "") for d in deltas)
        calls = [call for d in deltas for call in d.get("tool_calls", [])]
        assert any(c["function"].get("name") == "f" for c in calls)
        assert json.loads(
            "".join(c["function"].get("arguments", "") for c in calls)
        ) == {"x": 1}


def test_runner_bridge_keeps_logprobs_when_detokenizer_has_no_text():
    engine = VLMEngine.__new__(VLMEngine)

    def events(ids, *, stats, **kwargs):
        for n, text in enumerate(["", "ok", ""]):
            stats.generated = n + 1
            yield text, n, "normal", "stop" if n == 2 else None, 0, _lp(n)

    engine._runner_events = events
    outputs = queue.Queue()
    engine._stream_vlm_runner_text([1], "test", outputs, logprobs=True)
    records = []
    while not outputs.empty():
        output = outputs.get_nowait()
        if output is not None:
            records.append(output)
    assert [o.logprobs for o in records] == [[_lp(n)] for n in range(3)]


@pytest.mark.parametrize("streaming", [False, True])
def test_masked_top_candidates_do_not_emit_nonfinite_json(streaming):
    from yunshu_gateway.routers.chat import _format_chat_logprobs, _format_logprobs

    raw = [_lp(1)]
    raw[0]["top_logprobs"] += [
        {"token": "masked", "logprob": float("-inf")},
        {"token": "invalid", "logprob": float("nan")},
    ]
    formatter = _format_chat_logprobs if streaming else _format_logprobs
    formatted = formatter(raw, tokenizer=None, top_logprobs=20)
    json.dumps(formatted, allow_nan=False)
    assert formatted["content"][0]["logprob"] == raw[0]["logprob"]
    assert formatted["content"][0]["top_logprobs"] == [
        {"token": "t1", "logprob": -1.0, "bytes": [116, 49]},
        {"token": "other", "logprob": -10.0, "bytes": [111, 116, 104, 101, 114]},
    ]
