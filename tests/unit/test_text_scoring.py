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


def test_model_card_states_real_scoring_limits(tmp_path):
    from yunshu_engine.model_card import build_model_card

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "xlm-roberta",
                "architectures": ["XLMRobertaForSequenceClassification"],
                "max_position_embeddings": 514,
                "pad_token_id": 1,
                "id2label": {"0": "relevance"},
            }
        )
    )
    card = build_model_card(tmp_path)
    assert card.context["effective"] == 512
    assert card.context["native"] == 514
    assert {"rerank", "score", "classify"} <= set(card.capabilities())
    assert "/v1/classify" in card.api["endpoints"]
    q = tmp_path / "Qwen3-Reranker"
    q.mkdir()
    (q / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "max_position_embeddings": 40960,
            }
        )
    )
    assert build_model_card(q).context["effective"] == 8192


def test_cancelled_waiter_remains_active_until_executor_finishes(head_client):
    import asyncio
    import threading

    _, engine = head_client
    entered, release = threading.Event(), threading.Event()

    def work():
        entered.set()
        release.wait(timeout=5)

    async def run():
        task = asyncio.create_task(engine._run(work))
        await asyncio.to_thread(entered.wait, 2)
        assert engine.has_active_requests()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine.has_active_requests()
        release.set()

        for _ in range(100):
            if not engine.has_active_requests():
                break
            await asyncio.sleep(0.01)
        assert not engine.has_active_requests()

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_rejected_executor_submission_does_not_leave_engine_busy(
    head_client, monkeypatch
):
    import asyncio

    from yunshu_engine import mlx_executor

    class Rejected:
        def submit(self, *args):
            raise RuntimeError("executor shut down")

    _, engine = head_client
    monkeypatch.setattr(mlx_executor, "get_mlx_executor", lambda: Rejected())
    with pytest.raises(RuntimeError, match="shut down"):
        asyncio.run(engine._run(lambda: None))
    assert not engine.has_active_requests()


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

    async def score(pairs, instruction=None, use_activation=True):
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


def test_vllm_score_aliases_and_raw_logits(head_client):
    client, engine = head_client
    engine.is_reranker = True

    async def score(pairs, instruction=None, use_activation=True):
        assert pairs == [("q", "d")]
        assert instruction == "custom"
        return [0.75 if use_activation else 1.0986122886681098]

    engine.score_pairs = score
    r = client.post(
        "/v1/score",
        json={
            "model": "m",
            "queries": "q",
            "documents": ["d"],
            "instruction": "custom",
            "use_activation": False,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["score"] == pytest.approx(1.0986122886681098)
    assert r.json()["id"].startswith("score-")
    assert isinstance(r.json()["created"], int)


def test_media_route_checks_against_fake_http(head_client):
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts/research"))
    import route_checks  # noqa: F401 - initializes the shared registry first
    from route_checks_media import _classify_head_served, _rerank_served

    client, engine = head_client
    ctx = SimpleNamespace(
        model="m",
        req=client.request,
        fixtures={"classify_reference": lambda texts: [[0.1, 0.9] for _ in texts]},
        notes={},
    )
    _classify_head_served(ctx)
    assert ctx.notes["classifier_oracle"]["passed"]
    engine.is_reranker = True

    async def score(pairs, instruction=None, use_activation=True):
        return [0.8, 0.2, 0.4, 0.6, 0.1]

    engine.score_pairs = score
    ctx.fixtures["rerank_reference"] = lambda pairs: [0.8, 0.2, 0.4, 0.6, 0.1]
    _rerank_served(ctx)
    assert ctx.notes["rerank_oracle"]["passed"]


def test_entire_probe_on_fake_engine(tmp_path, monkeypatch):
    """Exercise argument-independent probe flow on CPU before any model is loaded."""
    import asyncio
    import math
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts/research"))
    import rerank_parity
    import route_checks  # noqa: F401

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertForSequenceClassification"],
                "id2label": {"0": "relevant"},
            }
        )
    )
    scores = [0.8, 0.2, 0.4, 0.6, 0.1]
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"scores": scores, "complete": True}))

    async def start(self):
        self._loaded = True

    async def score(self, pairs, instruction=None, use_activation=True):
        out = [scores[rerank_parity.PAIRS.index((a, b))] for a, b in pairs]
        return out if use_activation else [math.log(v / (1 - v)) for v in out]

    monkeypatch.setattr(TextScoringEngine, "start", start)
    monkeypatch.setattr(TextScoringEngine, "score_pairs", score)
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    result = asyncio.run(rerank_parity.run(str(tmp_path), str(reference)))
    assert result["passed"]
    assert result["route_checks"]["rerank_oracle"]["passed"]


@pytest.mark.parametrize("family", ["bert", "roberta", "xlm-roberta"])
@pytest.mark.parametrize("num_labels", [1, 2])
def test_encoder_head_matches_transformers_fixture(tmp_path, family, num_labels):
    """Small-array unit test, independently exercising the complete trained head."""
    import mlx.core as mx

    torch = pytest.importorskip("torch")
    pytest.importorskip("mlx_embeddings")
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
        num_labels=num_labels,
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


def test_quantized_classifier_keeps_float_trained_head(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from transformers import BertConfig, BertForSequenceClassification

    from yunshu_engine.scoring_engine import load_sequence_classifier

    config = BertConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        max_position_embeddings=64,
        num_labels=2,
    )
    BertForSequenceClassification(config).save_pretrained(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    original = load_sequence_classifier(str(tmp_path), cfg)
    nn.quantize(
        original,
        group_size=64,
        bits=4,
        class_predicate=lambda path, module: (
            path.startswith("bert.") and hasattr(module, "to_quantized")
        ),
    )
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(original.parameters()))
    )
    cfg["quantization"] = {"group_size": 64, "bits": 4}
    loaded = load_sequence_classifier(str(tmp_path), cfg)
    assert isinstance(loaded.classifier, nn.Linear)
    assert loaded.classifier.weight.tolist() == original.classifier.weight.tolist()
    ids = mx.array([[2, 7, 9, 3]])
    assert loaded(input_ids=ids).tolist() == original(input_ids=ids).tolist()
    weights = dict(tree_flatten(original.parameters()))
    weights.pop("classifier.weight")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(ValueError):
        load_sequence_classifier(str(tmp_path), cfg)


def test_jina_ranking_head_is_not_qwen_yes_no():
    from yunshu_engine.scoring_engine import scoring_kind

    assert (
        scoring_kind(
            {"model_type": "qwen3", "architectures": ["JinaForRanking"]},
            "jina-reranker-v3-4bit-mxfp4",
        )
        is None
    )


def test_published_bge_quantized_namespace_keeps_head(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

    from yunshu_engine.scoring_engine import load_sequence_classifier

    cfg = XLMRobertaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        max_position_embeddings=64,
        num_labels=1,
    )
    XLMRobertaForSequenceClassification(cfg).save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    original = load_sequence_classifier(str(tmp_path), config)
    nn.quantize(original, group_size=64, bits=8)
    # Published BGE converter removes roberta. and also quantizes the trained head.
    weights = {
        k.removeprefix("roberta."): v for k, v in tree_flatten(original.parameters())
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    config["quantization"] = {"bits": 8, "group_size": 64}
    loaded = load_sequence_classifier(str(tmp_path), config)
    assert isinstance(loaded.classifier.out_proj, nn.QuantizedLinear)
    ids = mx.array([[2, 7, 9, 3]])
    assert loaded(input_ids=ids)[0].tolist() == pytest.approx(
        original(input_ids=ids)[0].tolist(), abs=1e-5
    )
