"""``/tokenize`` and ``/apply-template`` render a conversation exactly as generation does, on the text
path (BatchedEngine) and the VLM path (VLMEngine): the family message adapter (a late system note after a
tool result stays in place as a ``[System note]`` user turn, token-counter markers are stripped), tool-call
history and role remaps all apply, so the previewed ids equal the ids the engine actually prefills.

Tiny random-weight CPU checkpoints with the real Qwen tokenizer; needs the vocab-only GGUF
(``python -m tests.unit.cpu_models``); skipped without it."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("transformers")
pytest.importorskip("gguf")

from tests.unit.cpu_models import (
    build_qwen3_dir,
    build_qwen35_dir,
    qwen_vocab_gguf,
    session_dir,
)
from tests.unit.cpu_serve import chat, serve

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        qwen_vocab_gguf() is None,
        reason="Qwen vocab GGUF absent: run `python -m tests.unit.cpu_models`",
    ),
]


def _tool_turn(i):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "Read", "arguments": json.dumps({"p": i})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"},
    ]


CONVERSATION = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Fix the bug in foo."},
    *_tool_turn(0),
    {"role": "system", "content": "hook note 0"},
    {
        "role": "system",
        "content": "budget\n<total_tokens>123 tokens left</total_tokens>",
    },
    *_tool_turn(1),
    {"role": "developer", "content": "late developer note"},
]


def _spy_encode(eng, monkeypatch):
    """Record every prompt the engine encodes for prefill (the encode point of each path)."""
    seen: list[list[int]] = []
    if type(eng).__name__ == "BatchedEngine":
        real = eng._encode_prompt

        def spy(tokenizer, text):
            ids = real(tokenizer, text)
            seen.append(list(ids))
            return ids

        monkeypatch.setattr(eng, "_encode_prompt", spy)
    else:
        prefix = eng._text_prompt_cache.prefix_tokens
        real = prefix.encode

        def spy(tokenizer, text, **kw):
            ids = real(tokenizer, text, **kw)
            seen.append(list(ids))
            return ids

        monkeypatch.setattr(prefix, "encode", spy)
    return seen


@pytest.fixture(scope="module", params=["text", "vlm"])
def served(request):
    if request.param == "text":
        model_dir = build_qwen3_dir(session_dir("qwen3"))
    else:
        model_dir = build_qwen35_dir(session_dir("qwen35"))
    with serve(model_dir) as (client, eng):
        assert type(eng).__name__ == (
            "BatchedEngine" if request.param == "text" else "VLMEngine"
        )
        yield client, eng


def test_tokenize_ids_equal_the_prefilled_ids(served, monkeypatch):
    client, eng = served
    seen = _spy_encode(eng, monkeypatch)
    r = chat(client, CONVERSATION, max_tokens=2)
    assert r.status_code == 200, r.text
    prefilled = seen[-1]
    assert r.json()["usage"]["prompt_tokens"] == len(prefilled)

    tok = client.post("/tokenize", json={"messages": CONVERSATION})
    assert tok.status_code == 200, tok.text
    assert tok.json()["tokens"] == prefilled
    assert tok.json()["count"] == len(prefilled)

    text = eng._tokenizer.decode(prefilled)
    # the late note stayed in place as its own user turn; the token counter is gone
    assert "[System note]\nhook note 0\n\nbudget\n[/System note]" in text
    assert "total_tokens" not in text
    tpl = client.post("/apply-template", json={"messages": CONVERSATION})
    assert tpl.status_code == 200, tpl.text
    assert tpl.json()["prompt"] == text
