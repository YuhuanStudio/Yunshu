"""vLLM score names and the cross-encoder instruction contract."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from yunshu_gateway.routers import scoring


@pytest.mark.parametrize(
    "fields",
    [
        dict(queries="q", documents=["d"]),
        dict(queries="q", items=["d"]),
        dict(data_1="q", data_2=["d"]),
    ],
)
def test_score_aliases(fields):
    r = scoring.ScoreRequest(model="m", **fields)
    assert r.text_1 == "q" and r.text_2 == ["d"]


@pytest.mark.asyncio
async def test_cross_encoder_scores_pairs_and_instruction(monkeypatch):
    engine = SimpleNamespace(is_reranker=True, rerank=AsyncMock(return_value=[0.75]))
    monkeypatch.setattr(scoring, "_resolve_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(scoring, "_check_permission", lambda *a: None)
    monkeypatch.setattr(scoring, "_check_model_access", lambda *a: None)
    r = scoring.ScoreRequest(
        model="m",
        queries="q",
        documents=["a", "b"],
        instruction="outer",
        chat_template_kwargs={"instruction": "inner"},
    )
    response = await scoring.create_score(r, None)
    assert [d["score"] for d in json.loads(response.body)["data"]] == [0.75, 0.75]
    assert engine.rerank.await_args_list[0].kwargs == {"instruction": "inner"}
    assert engine.rerank.await_args_list[1].args == ("q", ["b"])


@pytest.mark.asyncio
async def test_bi_encoder_instruction_is_not_prepended(monkeypatch):
    monkeypatch.setattr(
        scoring, "_resolve_engine", AsyncMock(return_value=SimpleNamespace())
    )
    embeddings = AsyncMock(return_value=[[1.0, 0.0]])
    monkeypatch.setattr(scoring, "_get_embeddings", embeddings)
    monkeypatch.setattr(scoring, "_check_permission", lambda *a: None)
    monkeypatch.setattr(scoring, "_check_model_access", lambda *a: None)
    await scoring.create_score(
        scoring.ScoreRequest(
            model="m", queries="q", documents="d", instruction="retrieve"
        ),
        None,
    )
    assert embeddings.await_args_list[0].args[1] == ["q"]
    assert embeddings.await_args_list[1].args[1] == ["d"]
