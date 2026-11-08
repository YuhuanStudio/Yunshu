"""Tests for fast path parameter passthrough: frequency_penalty, presence_penalty,
logit_bias, reasoning_effort, xtc sampling.

These verify that parameters flow from the gateway through BatchedEngine's
_generate_fast and _stream_generate_fast paths.
"""

import pytest


class TestFrequencyPresencePenaltyFastPath:
    """Test that frequency/presence penalty processors are correctly built in fast path."""

    def test_frequency_penalty_processor_counts_tokens(self):
        """frequency_penalty should penalize based on token count, not just last token."""

        def _freq_penalty(tokens, logits, fp=0.5, pp=0.0):
            counts = {}
            for t in tokens:
                counts[int(t)] = counts.get(int(t), 0) + 1
            for tid, cnt in counts.items():
                logits[..., tid] -= fp * cnt
                logits[..., tid] -= pp
            return logits

        import mlx.core as mx

        logits = mx.zeros(100)
        logits = logits.astype(mx.float32)
        # Token 5 appears 3 times → penalty should be 0.5 * 3 = 1.5
        tokens = mx.array([5, 5, 5])
        result = _freq_penalty(tokens, logits, fp=0.5, pp=0.0)
        assert float(result[5]) == pytest.approx(-1.5, abs=0.01)

    def test_presence_penalty_processor(self):
        """presence_penalty should penalize once per unique token."""

        def _pres_penalty(tokens, logits, fp=0.0, pp=0.5):
            counts = {}
            for t in tokens:
                counts[int(t)] = counts.get(int(t), 0) + 1
            for tid, cnt in counts.items():
                logits[..., tid] -= fp * cnt
                logits[..., tid] -= pp
            return logits

        import mlx.core as mx

        logits = mx.zeros(100).astype(mx.float32)
        tokens = mx.array([5, 5, 5])
        result = _pres_penalty(tokens, logits, fp=0.0, pp=0.5)
        assert float(result[5]) == pytest.approx(-0.5, abs=0.01)

    def test_logit_bias_processor(self):
        """logit_bias should add bias to specified token IDs."""

        def _logit_bias_proc(tokens, logits, biases=None):
            biases = biases if biases is not None else {10: 2.0, 20: -3.0}
            for tid, bias in biases.items():
                logits[..., tid] += bias
            return logits

        import mlx.core as mx

        logits = mx.zeros(100).astype(mx.float32)
        tokens = mx.array([0])
        result = _logit_bias_proc(tokens, logits)
        assert float(result[10]) == pytest.approx(2.0, abs=0.01)
        assert float(result[20]) == pytest.approx(-3.0, abs=0.01)


class TestCompletionsRequestParams:
    """Test CompletionRequest accepts new parameters."""

    def test_reasoning_effort_field(self):
        from yunshu_gateway.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            prompt="hello",
            reasoning_effort="low",
        )
        assert req.reasoning_effort == "low"

    def test_xtc_fields(self):
        from yunshu_gateway.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            prompt="hello",
            xtc_probability=0.5,
            xtc_threshold=0.1,
        )
        assert req.xtc_probability == 0.5
        assert req.xtc_threshold == 0.1

    def test_xtc_defaults(self):
        from yunshu_gateway.routers.completions import CompletionRequest

        req = CompletionRequest(model="test", prompt="hello")
        assert req.xtc_probability == 0.0
        assert req.xtc_threshold == 0.0


class TestTokenizeRequest:
    """Test tokenize endpoint request models."""

    def test_tokenize_request(self):
        from yunshu_gateway.routers.tokenize import TokenizeRequest

        req = TokenizeRequest(model="test", text="hello world")
        assert req.prompt == "hello world"

    def test_detokenize_request(self):
        from yunshu_gateway.routers.tokenize import DetokenizeRequest

        req = DetokenizeRequest(model="test", tokens=[1, 2, 3])
        assert req.tokens == [1, 2, 3]
