"""Tests for scoring endpoints (/v1/pooling, /v1/score, /v1/rerank)."""
import math
import pytest


def _compute_similarity(a, b, method="cosine"):
    """Reference similarity computation."""
    if method == "cosine":
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0
    elif method == "dot":
        return sum(x * y for x, y in zip(a, b))
    elif method == "euclidean":
        return -math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class TestSimilarityComputation:
    def test_cosine_identical(self):
        v = [1.0, 2.0, 3.0]
        assert _compute_similarity(v, v, "cosine") == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _compute_similarity(a, b, "cosine") == pytest.approx(0.0)

    def test_cosine_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _compute_similarity(a, b, "cosine") == pytest.approx(-1.0)

    def test_dot_product(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert _compute_similarity(a, b, "dot") == pytest.approx(32.0)

    def test_euclidean_distance(self):
        a = [1.0, 0.0]
        b = [4.0, 4.0]
        expected = -math.sqrt(9 + 16)
        assert _compute_similarity(a, b, "euclidean") == pytest.approx(expected)


class TestPoolingRequest:
    def test_pooling_request_model(self):
        from yunshu_gateway.routers.scoring import PoolingRequest
        req = PoolingRequest(model="test-model", input="hello")
        assert req.model == "test-model"
        assert req.pooling_type == "CLS"

    def test_pooling_request_mean(self):
        from yunshu_gateway.routers.scoring import PoolingRequest
        req = PoolingRequest(model="test", input=["a", "b"], pooling_type="MEAN")
        assert req.pooling_type == "MEAN"


class TestScoreRequest:
    def test_score_request_basic(self):
        from yunshu_gateway.routers.scoring import ScoreRequest
        req = ScoreRequest(model="test", text_1="hello", text_2="world")
        assert req.scoring_type == "cosine"

    def test_score_request_broadcast(self):
        from yunshu_gateway.routers.scoring import ScoreRequest
        req = ScoreRequest(
            model="test",
            text_1=["hello"],
            text_2=["world", "foo"],
        )
        texts_a = req.text_1 if isinstance(req.text_1, list) else [req.text_1]
        texts_b = req.text_2 if isinstance(req.text_2, list) else [req.text_2]
        # Model validates but broadcast logic is in the endpoint
        assert len(texts_a) == 1
        assert len(texts_b) == 2


class TestRerankRequest:
    def test_rerank_request_basic(self):
        from yunshu_gateway.routers.scoring import RerankRequest
        req = RerankRequest(
            model="test",
            query="what is AI?",
            documents=["AI is intelligence", "The sky is blue"],
        )
        assert req.top_n is None
        assert req.return_documents is True

    def test_rerank_request_top_n(self):
        from yunshu_gateway.routers.scoring import RerankRequest
        req = RerankRequest(
            model="test",
            query="query",
            documents=["a", "b", "c"],
            top_n=2,
            return_documents=False,
        )
        assert req.top_n == 2
        assert req.return_documents is False
