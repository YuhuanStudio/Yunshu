"""scoring endpoints enforce per-text length + per-list count caps.

/v1/embeddings caps every input string (8192 chars) and the list (2048 elements). The
scoring endpoints only capped rerank *documents*, leaving the rerank query, score
text_1/text_2, pooling input and classify input/labels uncapped per-text — a single ~10MB
string (under the global 10MB body limit) or a 2048-list of multi-KB strings drives one
giant uncapped forward pass (context-length / memory-pressure DoS). Now capped at the
schema boundary, matching embeddings.
"""

from __future__ import annotations

import pydantic
import pytest

from yunshu_gateway.routers.scoring import (
    _MAX_INPUT_TEXT_LENGTH,
    _MAX_INPUT_TEXTS,
    ClassifyRequest,
    PoolingRequest,
    RerankRequest,
    ScoreRequest,
)

_LONG = "x" * (_MAX_INPUT_TEXT_LENGTH + 1)


def test_pooling_rejects_overlong_str_and_list_item():
    with pytest.raises(pydantic.ValidationError):
        PoolingRequest(model="m", input=_LONG)
    with pytest.raises(pydantic.ValidationError):
        PoolingRequest(model="m", input=["ok", _LONG])
    # count cap
    with pytest.raises(pydantic.ValidationError):
        PoolingRequest(model="m", input=["a"] * (_MAX_INPUT_TEXTS + 1))
    assert PoolingRequest(model="m", input=["a", "b"]).input == ["a", "b"]


def test_score_rejects_overlong_and_count():
    with pytest.raises(pydantic.ValidationError):
        ScoreRequest(model="m", text_1=_LONG, text_2="b")
    with pytest.raises(pydantic.ValidationError):
        ScoreRequest(model="m", text_1="a", text_2=["b", _LONG])
    with pytest.raises(pydantic.ValidationError):
        ScoreRequest(model="m", text_1=["a"] * (_MAX_INPUT_TEXTS + 1), text_2="b")
    assert ScoreRequest(model="m", text_1="a", text_2="b").scoring_type == "cosine"


def test_rerank_rejects_overlong_query():
    with pytest.raises(pydantic.ValidationError):
        RerankRequest(model="m", query=_LONG, documents=["d"])
    assert RerankRequest(model="m", query="q", documents=["d"]).query == "q"


def test_classify_rejects_overlong_input_and_label():
    with pytest.raises(pydantic.ValidationError):
        ClassifyRequest(model="m", input=_LONG, labels=["a", "b"])
    with pytest.raises(pydantic.ValidationError):
        ClassifyRequest(model="m", input="hi", labels=["a", _LONG])
    with pytest.raises(pydantic.ValidationError):
        ClassifyRequest(model="m", input="hi", labels=["x"] * (_MAX_INPUT_TEXTS + 1))
    assert ClassifyRequest(model="m", input="hi", labels=["a", "b"]).labels == [
        "a",
        "b",
    ]
