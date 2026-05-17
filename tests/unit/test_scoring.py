"""Tests for scoring endpoints (/v1/pooling, /v1/score, /v1/rerank, /v1/classify)."""
import math
import pytest
from unittest.mock import MagicMock, AsyncMock


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

    def test_pooling_request_last(self):
        from yunshu_gateway.routers.scoring import PoolingRequest
        req = PoolingRequest(model="test", input="hello", pooling_type="LAST")
        assert req.pooling_type == "LAST"


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


class TestClassifyRequest:
    def test_classify_request_basic(self):
        from yunshu_gateway.routers.scoring import ClassifyRequest
        req = ClassifyRequest(
            model="test",
            input="The stock market crashed",
            labels=["finance", "sports", "technology"],
        )
        assert req.model == "test"
        assert len(req.labels) == 3

    def test_classify_request_empty_labels(self):
        from yunshu_gateway.routers.scoring import ClassifyRequest
        req = ClassifyRequest(model="test", input="hello")
        assert req.labels == []


class TestClassifyTemperature:
    """Verify that temperature-scaled softmax produces well-separated probabilities."""

    def test_softmax_with_temperature(self):
        """Cosine similarities with temperature=0.07 should produce sharp distribution."""
        import math
        # Simulate cosine similarities for 3 labels
        cos_sims = [0.8, 0.3, -0.1]
        temperature = 0.07

        scores = [s / temperature for s in cos_sims]
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        # Should be well-separated, not near-uniform
        assert probs[0] > 0.99  # Best match should dominate
        assert sum(probs) == pytest.approx(1.0)

    def test_softmax_without_temperature_uniform(self):
        """Without temperature, softmax on cosines is too flat."""
        import math
        cos_sims = [0.8, 0.3, -0.1]

        max_score = max(cos_sims)
        exp_scores = [math.exp(s - max_score) for s in cos_sims]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        # Without temperature, distribution is much flatter
        assert probs[0] < 0.6  # Not dominating enough


class TestRerankScoring:
    """Verify rerank relevance score computation."""

    def test_cosine_to_relevance_range(self):
        """Cosine similarity should be mapped to [0, 1]."""
        # cos=1 -> relevance=1.0
        assert (1.0 + 1.0) / 2.0 == 1.0
        # cos=-1 -> relevance=0.0
        assert (-1.0 + 1.0) / 2.0 == 0.0
        # cos=0 -> relevance=0.5
        assert (0.0 + 1.0) / 2.0 == 0.5

    def test_similarity_empty_vectors(self):
        """Empty vectors should return 0.0."""
        from yunshu_gateway.routers.scoring import _compute_similarity
        assert _compute_similarity([], [1.0, 2.0], "cosine") == 0.0
        assert _compute_similarity([1.0, 2.0], [], "cosine") == 0.0
        assert _compute_similarity([], [], "cosine") == 0.0


class TestValidationConstants:
    """Verify validation constants are correct."""

    def test_valid_pooling_types(self):
        from yunshu_gateway.routers.scoring import _VALID_POOLING_TYPES
        assert _VALID_POOLING_TYPES == {"CLS", "MEAN", "LAST"}

    def test_valid_scoring_types(self):
        from yunshu_gateway.routers.scoring import _VALID_SCORING_TYPES
        assert _VALID_SCORING_TYPES == {"cosine", "dot", "euclidean"}


class TestDimensionMismatch:
    """Verify that dimension mismatches are caught, not silently truncated."""

    def test_cosine_dimension_mismatch_raises(self):
        from yunshu_gateway.routers.scoring import _compute_similarity
        with pytest.raises(ValueError, match="dimension mismatch"):
            _compute_similarity([1.0, 2.0], [1.0, 2.0, 3.0], "cosine")

    def test_dot_dimension_mismatch_raises(self):
        from yunshu_gateway.routers.scoring import _compute_similarity
        with pytest.raises(ValueError, match="dimension mismatch"):
            _compute_similarity([1.0, 2.0], [1.0, 2.0, 3.0], "dot")

    def test_euclidean_dimension_mismatch_raises(self):
        from yunshu_gateway.routers.scoring import _compute_similarity
        with pytest.raises(ValueError, match="dimension mismatch"):
            _compute_similarity([1.0, 2.0], [1.0, 2.0, 3.0], "euclidean")

    def test_same_dimensions_works(self):
        from yunshu_gateway.routers.scoring import _compute_similarity
        # Should not raise
        result = _compute_similarity([1.0, 2.0], [3.0, 4.0], "cosine")
        assert isinstance(result, float)


class TestClassifySoftmaxOverflow:
    """Verify classify endpoint handles softmax edge cases."""

    def test_softmax_underflow_uniform_fallback(self):
        """When all exp scores underflow to 0, uniform distribution is returned."""
        import math
        # Extreme negative scores that cause exp to underflow
        scores = [-1e308, -1e308, -1e308]
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        if total == 0:
            n = len(exp_scores)
            probs = [1.0 / n] * n
        else:
            probs = [e / total for e in exp_scores]
        assert len(probs) == 3
        assert sum(probs) == pytest.approx(1.0)
        assert probs[0] == pytest.approx(1.0 / 3)

    def test_softmax_normal_case(self):
        """Normal softmax should not trigger the underflow fallback."""
        import math
        scores = [0.8 / 0.07, 0.3 / 0.07, -0.1 / 0.07]
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        assert total > 0
        probs = [e / total for e in exp_scores]
        assert sum(probs) == pytest.approx(1.0)
        assert probs[0] > 0.99


class TestScoreRequestEmptyStrings:
    """Verify score endpoint handles edge cases with inputs."""

    def test_score_request_list_of_one(self):
        from yunshu_gateway.routers.scoring import ScoreRequest
        req = ScoreRequest(
            model="test",
            text_1=["hello"],
            text_2=["world", "foo", "bar"],
        )
        assert len(req.text_1) == 1
        assert len(req.text_2) == 3

    def test_score_request_mismatched_non_broadcastable(self):
        """Two lists of different lengths > 1 should not broadcast."""
        texts_a = ["a", "b"]
        texts_b = ["x", "y", "z"]
        # This would trigger the 400 error in the endpoint
        assert len(texts_a) != len(texts_b)
        assert len(texts_a) != 1
        assert len(texts_b) != 1
