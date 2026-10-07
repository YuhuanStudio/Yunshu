"""/v1/embeddings multimodal request shapes: vLLM-style messages, object items, task, usage."""

import asyncio

import pytest
from fastapi import HTTPException

from yunshu_gateway.routers import embeddings as emb


def test_messages_become_one_interleaved_item():
    item = emb.messages_to_item(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "shoes "},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA"},
                    },
                    {"type": "text", "text": " grip: "},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "QQ==", "format": "wav"},
                    },
                    {"type": "video", "frames": ["f1", "f2"]},
                ],
            }
        ]
    )
    assert item["text"] == "shoes <|image|> grip: <|audio|><|video|>"
    assert item["image"] == ["data:image/png;base64,AA"]
    assert item["audio"] == ["data:audio/wav;base64,QQ=="]
    assert item["video"] == [["f1", "f2"]]


def test_messages_reject_unknown_part():
    with pytest.raises(ValueError, match="unsupported"):
        emb.messages_to_item([{"role": "user", "content": [{"type": "file"}]}])


def test_request_accepts_messages_xor_input():
    r = emb.EmbeddingRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    assert r.input == [{"text": "hi"}]
    with pytest.raises(ValueError, match="either input or messages"):
        emb.EmbeddingRequest(
            model="m", input="x", messages=[{"role": "user", "content": "x"}]
        )
    with pytest.raises(ValueError, match="required"):
        emb.EmbeddingRequest(model="m")


def test_long_text_up_to_8k_tokens_is_not_rejected_by_the_char_cap():
    emb.EmbeddingRequest(model="m", input="word " * 8000)


class _Fake:
    def __init__(self):
        self.seen = None

    def decode(self, ids):
        return "decoded:" + ",".join(map(str, ids))

    async def embed_with_usage(self, items, instruction=None, task=None):
        self.seen = (items, instruction, task)
        return [[3.0, 4.0, 0.0, 0.0] for _ in items], 17


def test_multimodal_route_passes_task_and_reports_tokens():
    f = _Fake()
    req = emb.EmbeddingRequest(
        model="m",
        input=[{"text": "a", "image": "x"}, {"text": "b"}],
        task="SearchQuery",
        dimensions=2,
    )
    out = asyncio.run(emb._embed_multimodal(req, f))
    assert f.seen[2] == "SearchQuery" and f.seen[0][1] == {"text": "b"}
    assert out["usage"] == {"prompt_tokens": 17, "total_tokens": 17}
    assert out["data"][0]["embedding"] == pytest.approx([0.6, 0.8])


def test_multimodal_route_decodes_token_ids():
    f = _Fake()
    req = emb.EmbeddingRequest(model="m", input=[5, 6, 7])
    asyncio.run(emb._embed_multimodal(req, f))
    assert f.seen[0] == ["decoded:5,6,7"]


def test_bad_input_is_a_400_not_a_500():
    class Bad(_Fake):
        async def embed_with_usage(self, *a, **k):
            raise ValueError("unknown task 'x'")

    req = emb.EmbeddingRequest(model="m", input="a", task="x")
    with pytest.raises(HTTPException) as e:
        asyncio.run(emb._embed_multimodal(req, Bad()))
    assert e.value.status_code == 400
