"""OpenAI Responses echoes the client's `metadata` map back verbatim (up to 16
string key/value pairs). It was declared nowhere on ResponsesRequest (Pydantic
extra="ignore" silently dropped it) and every response payload hardcoded
{"user_id": req.user} instead — a client using metadata for request correlation got
nothing back. Now the field is declared, bounded to 16 pairs, and echoed by all four
payload builders (text non-stream, streaming, VLM, background).
"""
from __future__ import annotations

import inspect

import pytest

from yunshu_gateway.routers import responses
from yunshu_gateway.routers.responses import ResponsesRequest


def test_metadata_field_held_and_defaults_none():
    assert ResponsesRequest(model="qwen", input="hi", metadata={"k": "v"}).metadata == {"k": "v"}
    assert ResponsesRequest(model="qwen", input="hi").metadata is None


def test_metadata_bounded_to_16_pairs():
    with pytest.raises(ValueError):
        ResponsesRequest(model="qwen", input="hi",
                         metadata={str(i): "x" for i in range(17)})
    # exactly 16 is allowed
    assert ResponsesRequest(model="qwen", input="hi",
                            metadata={str(i): "x" for i in range(16)}).metadata is not None


def test_all_payload_builders_echo_client_metadata():
    src = inspect.getsource(responses)
    # the old hardcoded user_id-in-metadata is gone everywhere
    assert '{"user_id": req.user}' not in src
    # all four payload builders echo the client's metadata map
    assert src.count('"metadata": req.metadata,') == 4
