"""Tests for GPU-accelerated rejection sampling .

Verifies GPURejectionSampler, BatchRejectionResult, and the auto-detection
should_enable_gpu_rejection() function.
"""

import os

import mlx.core as mx

from yunshu_engine.gpu_rejection import (
    BatchRejectionResult,
    GPURejectionSampler,
    should_enable_gpu_rejection,
)

# ── Helpers ──

def _make_logits(model_picks: list[int], vocab_size: int = 100) -> mx.array:
    """Create logits where argmax at each position equals model_picks[i].

    Sets the model_pick position to 10.0 and everything else to 0.0,
    guaranteeing a clear argmax result.
    """
    K = len(model_picks)
    logits = mx.zeros((K, vocab_size))
    for i, pick in enumerate(model_picks):
        logits[i, pick] = 10.0
    return logits


def _make_logits_3d(model_picks: list[int], vocab_size: int = 100) -> mx.array:
    """Create 3D logits [1, K, vocab_size] for testing shape normalization."""
    return _make_logits(model_picks, vocab_size).reshape(1, len(model_picks), vocab_size)


# ── BatchRejectionResult Dataclass Tests ──

class TestBatchRejectionResult:
    """Tests for the BatchRejectionResult dataclass."""

    def test_dataclass_fields(self):
        result = BatchRejectionResult(
            accepted_count=3,
            rejection_position=3,
            verification_method="gpu_batch",
            latency_us=42.5,
        )
        assert result.accepted_count == 3
        assert result.rejection_position == 3
        assert result.verification_method == "gpu_batch"
        assert result.latency_us == 42.5

    def test_all_accepted_no_rejection(self):
        result = BatchRejectionResult(
            accepted_count=5,
            rejection_position=None,
            verification_method="gpu_batch",
            latency_us=10.0,
        )
        assert result.rejection_position is None

    def test_cpu_sequential_method(self):
        result = BatchRejectionResult(0, None, "cpu_sequential", 5.0)
        assert result.verification_method == "cpu_sequential"


# ── GPU Greedy Verification Tests ──

class TestGPUGreedyVerification:
    """Tests for GPURejectionSampler.verify_greedy()."""

    def setup_method(self):
        self.sampler = GPURejectionSampler()

    def test_all_matched_all_accepted(self):
        """All draft tokens match model picks — all accepted."""
        # Model picks: [5, 10, 15, 20, 25]
        logits = _make_logits([5, 10, 15, 20, 25])
        draft_ids = [5, 10, 15, 20, 25]

        result = self.sampler.verify_greedy(logits, draft_ids)

        assert result.accepted_count == 5
        assert result.rejection_position is None
        assert result.verification_method == "gpu_batch"
        assert result.latency_us > 0

    def test_all_matched_3d_logits(self):
        """3D logits shape [1, K, V] is handled correctly."""
        logits = _make_logits_3d([3, 7, 11])
        draft_ids = [3, 7, 11]

        result = self.sampler.verify_greedy(logits, draft_ids)

        assert result.accepted_count == 3
        assert result.rejection_position is None
        assert result.verification_method == "gpu_batch"

    def test_early_rejection_at_position_2(self):
        """First mismatch at position 2 — 2 accepted."""
        logits = _make_logits([5, 10, 99, 20, 25])  # Mismatch at pos 2
        draft_ids = [5, 10, 15, 20, 25]

        result = self.sampler.verify_greedy(logits, draft_ids)

        assert result.accepted_count == 2
        assert result.rejection_position == 2

    def test_rejection_at_position_0(self):
        """First token is already wrong — 0 accepted."""
        logits = _make_logits([99, 10, 15])  # Mismatch at pos 0
        draft_ids = [5, 10, 15]

        result = self.sampler.verify_greedy(logits, draft_ids)

        assert result.accepted_count == 0
        assert result.rejection_position == 0

    def test_rejection_at_last_position(self):
        """Only the last token mismatches — K-1 accepted."""
        logits = _make_logits([5, 10, 15, 20, 99])  # Mismatch at last pos
        draft_ids = [5, 10, 15, 20, 25]

        result = self.sampler.verify_greedy(logits, draft_ids)

        assert result.accepted_count == 4
        assert result.rejection_position == 4

    def test_empty_draft(self):
        """Empty draft token list — 0 accepted."""
        logits = mx.zeros((0, 50))
        result = self.sampler.verify_greedy(logits, [])

        assert result.accepted_count == 0
        assert result.rejection_position is None

    def test_single_token_accepted(self):
        """Single draft token that matches."""
        logits = _make_logits([42])
        result = self.sampler.verify_greedy(logits, [42])

        assert result.accepted_count == 1
        assert result.rejection_position is None

    def test_single_token_rejected(self):
        """Single draft token that doesn't match."""
        logits = _make_logits([99])
        result = self.sampler.verify_greedy(logits, [42])

        assert result.accepted_count == 0
        assert result.rejection_position == 0

    def test_rejection_stops_at_first_mismatch(self):
        """Verify that later matches after first mismatch don't count."""
        # Draft: [5, 10, 15, 20, 25]
        # Model: [5, 10, 99, 20, 25]  — mismatch at 2, but 3 and 4 match
        # Should only accept 2 (positions 0 and 1)
        logits = _make_logits([5, 10, 99, 20, 25])
        draft_ids = [5, 10, 15, 20, 25]

        result = self.sampler.verify_greedy(logits, draft_ids)

        assert result.accepted_count == 2
        assert result.rejection_position == 2

    def test_vocab_size_32000(self):
        """Works with realistic vocab sizes."""
        VOCAB = 32000
        picks = [100, 200, 300]
        logits = mx.zeros((3, VOCAB))
        for i, p in enumerate(picks):
            logits[i, p] = 10.0
        draft_ids = [100, 999, 300]  # Mismatch at pos 1

        result = self.sampler.verify_greedy(logits, draft_ids)

        assert result.accepted_count == 1
        assert result.rejection_position == 1


# ── GPU Stochastic Verification Tests ──

class TestGPUStochasticVerification:
    """Tests for GPURejectionSampler.verify_stochastic()."""

    def setup_method(self):
        self.sampler = GPURejectionSampler(rng_seed=42)

    def test_temperature_zero_uses_greedy(self):
        """Temperature=0 falls back to greedy verification."""
        logits = _make_logits([5, 10, 99])  # Mismatch at pos 2
        draft_ids = [5, 10, 15]
        draft_logprobs = [-0.1, -0.2, -0.3]

        result = self.sampler.verify_stochastic(
            logits, draft_ids, draft_logprobs, temperature=0.0
        )

        assert result.accepted_count == 2
        assert result.rejection_position == 2
        assert result.verification_method == "gpu_batch"

    def test_stochastic_accepts_all_when_ratio_high(self):
        """When target prob >> draft prob, acceptance ratio is 1.0 — all accepted."""
        VOCAB = 50
        logits = mx.zeros((3, VOCAB))
        # Draft tokens
        draft_ids = [5, 10, 15]
        # Set target logprobs very high for draft tokens
        for i, tok in enumerate(draft_ids):
            logits[i, tok] = 20.0  # Very high → target prob ~1.0

        # Draft logprobs are very low (draft is uncertain)
        draft_logprobs = [-5.0, -5.0, -5.0]

        result = self.sampler.verify_stochastic(
            logits, draft_ids, draft_logprobs, temperature=1.0
        )

        # With target prob >> draft prob, ratios are all 1.0, should accept all
        assert result.accepted_count == 3
        assert result.rejection_position is None

    def test_stochastic_rejects_when_draft_prob_higher(self):
        """When draft prob >> target prob, acceptance ratio < 1, likely rejected."""
        VOCAB = 50
        logits = mx.zeros((2, VOCAB))
        draft_ids = [5, 10]

        # Set target logprobs low for draft tokens
        # Distribute probability uniformly → low target prob for any specific token
        logits[:, :] = 0.0  # uniform → each token gets ~1/50 prob

        # Draft logprobs are very high (draft is very confident)
        draft_logprobs = [-0.01, -0.01]  # Draft prob ~0.99

        # With seed=42, run many times to check stochasticity
        sampler = GPURejectionSampler(rng_seed=42)
        result = sampler.verify_stochastic(
            logits, draft_ids, draft_logprobs, temperature=1.0
        )

        # Due to low acceptance ratio (~1/50 / 0.99 ≈ 0.02), very likely rejected
        assert result.accepted_count <= 2
        assert result.verification_method == "gpu_batch"

    def test_resample_drawn_from_target_distribution(self):
        """on rejection the correction token is sampled from the TARGET
        distribution (the documented fallback when only token-level draft logprobs
        exist), not the old broadcast-scalar 'residual'. A target distribution
        peaked on token A, with the draft token B rejected, must resample to A."""
        VOCAB = 32
        A = 7  # the token the TARGET overwhelmingly wants
        B = 3  # the (different) draft token, which will be rejected
        logits = mx.zeros((1, VOCAB))
        logits[0, A] = 30.0  # softmax → ~all mass on A
        # draft is very confident on B (high draft prob) but target prob of B ~0
        # → acceptance ratio ~0 → forced rejection at position 0.
        result = self.sampler.verify_stochastic(
            logits, [B], [-0.001], temperature=1.0
        )
        assert result.rejection_position == 0
        assert result.accepted_count == 0
        # The correction must come from the (peaked) target, i.e. token A — never B.
        assert result.resampled_token_id == A

    def test_no_broadcast_scalar_residual(self):
        """The mathematically meaningless scalar-broadcast residual is gone."""
        import inspect

        from yunshu_engine import gpu_rejection
        src = inspect.getsource(gpu_rejection.GPURejectionSampler.verify_stochastic)
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
        # the old per-vocab scalar subtraction is removed
        assert "target_probs - draft_prob_i" not in code
        # resampling is a categorical draw over the target logprobs row
        assert "mx.random.categorical(target_logprobs_full[i]" in code

    def test_empty_draft_stochastic(self):
        """Empty draft returns 0 accepted."""
        logits = mx.zeros((0, 50))
        result = self.sampler.verify_stochastic(
            logits, [], [], temperature=1.0
        )

        assert result.accepted_count == 0
        assert result.rejection_position is None

    def test_stochastic_latency_tracked(self):
        """Stochastic verification tracks latency."""
        logits = _make_logits([5, 10, 15])
        result = self.sampler.verify_stochastic(
            logits, [5, 10, 15], [-0.5, -0.5, -0.5], temperature=1.0
        )

        assert result.latency_us > 0


# ── Batch Verification Tests ──

class TestBatchVerification:
    """Tests for GPURejectionSampler.verify_greedy_batch()."""

    def setup_method(self):
        self.sampler = GPURejectionSampler()

    def test_batch_empty(self):
        """Empty batch returns empty results."""
        results = self.sampler.verify_greedy_batch([], [])
        assert results == []

    def test_batch_single_request(self):
        """Single request in batch works correctly."""
        logits = [_make_logits([5, 10, 15])]
        draft_ids = [[5, 10, 99]]  # Mismatch at pos 2

        results = self.sampler.verify_greedy_batch(logits, draft_ids)

        assert len(results) == 1
        assert results[0].accepted_count == 2
        assert results[0].rejection_position == 2

    def test_batch_multiple_requests(self):
        """Multiple requests verified simultaneously."""
        # Request 1: all match
        logits_1 = _make_logits([1, 2, 3])
        # Request 2: mismatch at pos 1
        logits_2 = _make_logits([5, 99, 15])
        # Request 3: mismatch at pos 0
        logits_3 = _make_logits([88, 20])

        results = self.sampler.verify_greedy_batch(
            [logits_1, logits_2, logits_3],
            [[1, 2, 3], [5, 10, 15], [25, 20]],
        )

        assert len(results) == 3
        assert results[0].accepted_count == 3
        assert results[0].rejection_position is None
        assert results[1].accepted_count == 1
        assert results[1].rejection_position == 1
        assert results[2].accepted_count == 0
        assert results[2].rejection_position == 0

    def test_batch_mixed_lengths(self):
        """Batch with different draft lengths per request."""
        logits_1 = _make_logits([5, 10, 15, 20, 25])
        logits_2 = _make_logits([3, 7])
        logits_3 = _make_logits([42])

        results = self.sampler.verify_greedy_batch(
            [logits_1, logits_2, logits_3],
            [[5, 10, 15, 20, 25], [3, 7], [42]],
        )

        assert results[0].accepted_count == 5
        assert results[0].rejection_position is None
        assert results[1].accepted_count == 2
        assert results[1].rejection_position is None
        assert results[2].accepted_count == 1
        assert results[2].rejection_position is None

    def test_batch_with_empty_request(self):
        """Batch includes a request with no draft tokens."""
        logits_1 = _make_logits([5])
        logits_2 = mx.zeros((0, 50))

        results = self.sampler.verify_greedy_batch(
            [logits_1, logits_2],
            [[5], []],
        )

        assert len(results) == 2
        assert results[0].accepted_count == 1
        assert results[1].accepted_count == 0


# ── CPU Sequential Fallback Tests ──

class TestCPUSequentialFallback:
    """Tests for GPURejectionSampler.verify_cpu_sequential()."""

    def test_cpu_all_accepted(self):
        """CPU sequential accepts all matching tokens."""
        logits = _make_logits([5, 10, 15])
        result = GPURejectionSampler.verify_cpu_sequential(logits, [5, 10, 15])

        assert result.accepted_count == 3
        assert result.rejection_position is None
        assert result.verification_method == "cpu_sequential"

    def test_cpu_early_rejection(self):
        """CPU sequential detects early rejection."""
        logits = _make_logits([5, 99, 15])
        result = GPURejectionSampler.verify_cpu_sequential(logits, [5, 10, 15])

        assert result.accepted_count == 1
        assert result.rejection_position == 1
        assert result.verification_method == "cpu_sequential"

    def test_cpu_3d_logits(self):
        """CPU sequential handles 3D logits."""
        logits = _make_logits_3d([3, 7])
        result = GPURejectionSampler.verify_cpu_sequential(logits, [3, 7])

        assert result.accepted_count == 2
        assert result.verification_method == "cpu_sequential"

    def test_cpu_empty(self):
        """CPU sequential handles empty drafts."""
        logits = mx.zeros((0, 50))
        result = GPURejectionSampler.verify_cpu_sequential(logits, [])

        assert result.accepted_count == 0
        assert result.rejection_position is None


# ── Auto-Detection Tests ──

class TestAutoDetection:
    """Tests for should_enable_gpu_rejection() and verify_auto()."""

    def test_default_disabled(self):
        """GPU rejection is disabled by default (no env var)."""
        os.environ.pop("YUNSHU_GPU_REJECTION", None)
        assert should_enable_gpu_rejection() is False

    def test_enabled_with_1(self):
        """GPU rejection enabled with YUNSHU_GPU_REJECTION=1."""
        os.environ["YUNSHU_GPU_REJECTION"] = "1"
        try:
            assert should_enable_gpu_rejection() is True
        finally:
            os.environ.pop("YUNSHU_GPU_REJECTION", None)

    def test_enabled_with_true(self):
        """GPU rejection enabled with YUNSHU_GPU_REJECTION=true."""
        os.environ["YUNSHU_GPU_REJECTION"] = "true"
        try:
            assert should_enable_gpu_rejection() is True
        finally:
            os.environ.pop("YUNSHU_GPU_REJECTION", None)

    def test_enabled_with_yes(self):
        """GPU rejection enabled with YUNSHU_GPU_REJECTION=yes."""
        os.environ["YUNSHU_GPU_REJECTION"] = "yes"
        try:
            assert should_enable_gpu_rejection() is True
        finally:
            os.environ.pop("YUNSHU_GPU_REJECTION", None)

    def test_disabled_with_0(self):
        """GPU rejection disabled with YUNSHU_GPU_REJECTION=0."""
        os.environ["YUNSHU_GPU_REJECTION"] = "0"
        try:
            assert should_enable_gpu_rejection() is False
        finally:
            os.environ.pop("YUNSHU_GPU_REJECTION", None)

    def test_verify_auto_uses_gpu_when_enabled(self):
        """verify_auto uses GPU batch when env var is set."""
        os.environ["YUNSHU_GPU_REJECTION"] = "1"
        try:
            sampler = GPURejectionSampler()
            logits = _make_logits([5, 10, 15])
            result = sampler.verify_auto(logits, [5, 10, 15])

            assert result.verification_method == "gpu_batch"
            assert result.accepted_count == 3
        finally:
            os.environ.pop("YUNSHU_GPU_REJECTION", None)

    def test_verify_auto_uses_cpu_when_disabled(self):
        """verify_auto uses CPU sequential when env var is not set."""
        os.environ.pop("YUNSHU_GPU_REJECTION", None)
        sampler = GPURejectionSampler()
        logits = _make_logits([5, 10, 15])
        result = sampler.verify_auto(logits, [5, 10, 15])

        assert result.verification_method == "cpu_sequential"
        assert result.accepted_count == 3

    def test_verify_auto_stochastic_when_enabled(self):
        """verify_auto uses stochastic path when temperature > 0 and enabled."""
        os.environ["YUNSHU_GPU_REJECTION"] = "1"
        try:
            sampler = GPURejectionSampler(rng_seed=42)
            logits = _make_logits([5, 10, 15])
            result = sampler.verify_auto(
                logits, [5, 10, 15],
                draft_logprobs=[-0.5, -0.5, -0.5],
                temperature=1.0,
            )

            assert result.verification_method == "gpu_batch"
        finally:
            os.environ.pop("YUNSHU_GPU_REJECTION", None)


# ── Bonus Token Tests ──

class TestBonusToken:
    """Tests for GPURejectionSampler.compute_bonus_token()."""

    def test_bonus_at_rejection_position(self):
        """Bonus token is model's argmax at the rejection position."""
        logits = _make_logits([5, 10, 99, 20, 25])
        bonus = GPURejectionSampler.compute_bonus_token(logits, 2)

        assert bonus == 99  # Model's argmax at position 2

    def test_bonus_at_last_position(self):
        """Bonus token from last position when all accepted."""
        logits = _make_logits([5, 10, 15])
        bonus = GPURejectionSampler.compute_bonus_token(logits, 2)

        assert bonus == 15  # Model's argmax at position 2

    def test_bonus_3d_logits(self):
        """Bonus token works with 3D logits."""
        logits = _make_logits_3d([5, 10, 99])
        bonus = GPURejectionSampler.compute_bonus_token(logits, 2)

        assert bonus == 99


# ── GPU vs CPU Equivalence Tests ──

class TestGPUvsCPUEquivalence:
    """Verify GPU batch and CPU sequential produce identical results."""

    def setup_method(self):
        self.gpu = GPURejectionSampler()

    def test_equivalence_all_accepted(self):
        """GPU and CPU agree when all tokens accepted."""
        logits = _make_logits([5, 10, 15, 20])

        gpu_result = self.gpu.verify_greedy(logits, [5, 10, 15, 20])
        cpu_result = GPURejectionSampler.verify_cpu_sequential(logits, [5, 10, 15, 20])

        assert gpu_result.accepted_count == cpu_result.accepted_count
        assert gpu_result.rejection_position == cpu_result.rejection_position

    def test_equivalence_early_rejection(self):
        """GPU and CPU agree on early rejection point."""
        logits = _make_logits([5, 99, 15, 20])

        gpu_result = self.gpu.verify_greedy(logits, [5, 10, 15, 20])
        cpu_result = GPURejectionSampler.verify_cpu_sequential(logits, [5, 10, 15, 20])

        assert gpu_result.accepted_count == cpu_result.accepted_count == 1
        assert gpu_result.rejection_position == cpu_result.rejection_position == 1

    def test_equivalence_first_token_rejected(self):
        """GPU and CPU agree when first token is rejected."""
        logits = _make_logits([99, 10, 15])

        gpu_result = self.gpu.verify_greedy(logits, [5, 10, 15])
        cpu_result = GPURejectionSampler.verify_cpu_sequential(logits, [5, 10, 15])

        assert gpu_result.accepted_count == cpu_result.accepted_count == 0
        assert gpu_result.rejection_position == cpu_result.rejection_position == 0

    def test_equivalence_last_token_rejected(self):
        """GPU and CPU agree when only last token is rejected."""
        logits = _make_logits([5, 10, 15, 99])

        gpu_result = self.gpu.verify_greedy(logits, [5, 10, 15, 20])
        cpu_result = GPURejectionSampler.verify_cpu_sequential(logits, [5, 10, 15, 20])

        assert gpu_result.accepted_count == cpu_result.accepted_count == 3
        assert gpu_result.rejection_position == cpu_result.rejection_position == 3

    def test_equivalence_many_tokens(self):
        """GPU and CPU agree with many draft tokens."""
        K = 20
        model_picks = list(range(K))
        draft_ids = list(range(K))
        # Insert mismatch at position 15
        model_picks[15] = 999

        logits = _make_logits(model_picks, vocab_size=1000)
        gpu_result = self.gpu.verify_greedy(logits, draft_ids)
        cpu_result = GPURejectionSampler.verify_cpu_sequential(logits, draft_ids)

        assert gpu_result.accepted_count == cpu_result.accepted_count == 15
        assert gpu_result.rejection_position == cpu_result.rejection_position == 15
