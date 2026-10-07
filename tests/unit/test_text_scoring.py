from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from yunshu_engine.model_manager import ModelType, _detect_model_type
from yunshu_engine.scoring_engine import (
    TextScoringEngine,
    probabilities,
    qwen_input_ids,
    sigmoid,
)
from yunshu_gateway.main import create_app
from yunshu_gateway.routers import scoring


def test_head_and_reranker_detection(tmp_path):
    p = tmp_path / "opaque-snapshot"
    p.mkdir()
    (p / "config.json").write_text(
        json.dumps(
            {"model_type": "bert", "architectures": ["BertForSequenceClassification"]}
        )
    )
    assert _detect_model_type(str(p)) == ModelType.CLASSIFIER
    q = tmp_path / "Qwen3-Reranker-0.6B"
    q.mkdir()
    (q / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
    )
    assert _detect_model_type(str(q)) == ModelType.RERANKER


def test_probability_rules_and_bad_logits():
    assert sigmoid(-1000) == 0
    assert sigmoid(1000) == 1
    assert probabilities([0]) == [0.5]
    assert probabilities([1000, 1000]) == [0.5, 0.5]
    assert probabilities([0, 0], True) == [0.5, 0.5]
    for logits in ([], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            probabilities(logits)


def test_qwen_truncation_preserves_suffix():
    class Tokenizer:
        def encode(self, text, **kw):
            return [11, 12] if "system" in text else [98, 99]

        def __call__(self, text, **kw):
            assert "<Query>: question\n<Document>: doc" in text
            assert kw["max_length"] == 3
            return {"input_ids": [1, 2, 3]}

    assert qwen_input_ids(Tokenizer(), "question", "doc", max_length=7) == [
        11,
        12,
        1,
        2,
        3,
        98,
        99,
    ]


@pytest.fixture
def head_client(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertForSequenceClassification"],
                "id2label": {"0": "negative", "1": "positive"},
            }
        )
    )
    engine = TextScoringEngine(str(tmp_path))

    async def classify(texts):
        return [[0.1, 0.9] for _ in texts]

    engine.classify = classify

    async def resolve(model):
        return engine

    monkeypatch.setattr(scoring, "_resolve_engine", resolve)
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    return TestClient(create_app()), engine


def test_head_classification_contract(head_client):
    client, _ = head_client
    r = client.post("/v1/classify", json={"model": "m", "input": ["good", "great"]})
    assert r.status_code == 200, r.text
    assert r.json()["data"] == [
        {"object": "classification", "index": i, "probs": [0.1, 0.9]} for i in range(2)
    ]
    assert r.json()["labels"] == ["negative", "positive"]
    r = client.post("/v1/classify", json={"model": "m", "input": "good"})
    assert r.json()["results"][0]["label"] == "positive"
    assert (
        client.post(
            "/v1/classify", json={"model": "m", "input": "good", "labels": ["a", "b"]}
        ).status_code
        == 400
    )


def test_rerank_and_score_joint_dispatch(head_client):
    client, engine = head_client
    engine.is_reranker = True
    calls = []

    async def score(pairs, instruction=None):
        calls.append((pairs, instruction))
        return [0.2, 0.8][: len(pairs)]

    engine.score_pairs = score
    r = client.post(
        "/v1/score", json={"model": "m", "text_1": "query", "text_2": ["a", "b"]}
    )
    assert r.status_code == 200
    assert [x["score"] for x in r.json()["data"]] == [0.2, 0.8]
    assert calls[-1][0] == [("query", "a"), ("query", "b")]
    r = client.post(
        "/v1/rerank",
        json={
            "model": "m",
            "query": "query",
            "documents": ["a", "b"],
            "top_n": 1,
            "instruction": "custom",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["results"] == [
        {"index": 1, "relevance_score": 0.8, "document": {"text": "b"}}
    ]
    assert calls[-1][1] == "custom"


def test_unsupported_head_fails_before_mlx_work(tmp_path):
    from yunshu_engine.scoring_engine import load_sequence_classifier

    with pytest.raises(ValueError, match="not supported"):
        load_sequence_classifier(str(tmp_path), {"model_type": "unknown"})


def test_parity_probe_dry_run_and_failure_rule(tmp_path):
    path = Path(__file__).parents[2] / "scripts/research/rerank_parity.py"
    spec = importlib.util.spec_from_file_location("rerank_parity", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert probe.compare([0.8, 0.2], [0.801, 0.199])["passed"]
    assert not probe.compare([0.2, 0.8], [0.8, 0.2])["passed"]
    assert not probe.compare([[0.7, 0.3]], [[0.3, 0.7]])["passed"]
    with pytest.raises(ValueError):
        probe.compare([], [])


def test_score_multi_class_head_rejected(head_client):
    client, _ = head_client
    r = client.post("/v1/score", json={"model": "m", "text_1": "q", "text_2": "d"})
    assert r.status_code == 400


@pytest.mark.parametrize("family", ["bert", "roberta", "xlm-roberta"])
def test_encoder_head_matches_transformers_fixture(tmp_path, family):
    """Small-array unit test, independently exercising the complete trained head."""
    import mlx.core as mx
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        BertConfig,
        RobertaConfig,
        XLMRobertaConfig,
    )

    from yunshu_engine.scoring_engine import load_sequence_classifier

    torch.manual_seed(23)
    cls = {
        "bert": BertConfig,
        "roberta": RobertaConfig,
        "xlm-roberta": XLMRobertaConfig,
    }[family]
    config = cls(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=32,
        num_labels=2,
        attn_implementation="eager",
    )
    model = AutoModelForSequenceClassification.from_config(config).eval()
    model.save_pretrained(tmp_path)
    mlx_model = load_sequence_classifier(
        str(tmp_path), json.loads((tmp_path / "config.json").read_text())
    )
    ids = [[2, 7, 9, 3]]
    with torch.no_grad():
        ref = model(input_ids=torch.tensor(ids)).logits[0].tolist()
    got = mlx_model(input_ids=mx.array(ids))[0].tolist()
    assert got == pytest.approx(ref, abs=1e-6)
