"""Tests for SpecDraftVerifier — speculative draft token verification.

Tests cover:
- Perfect match (all drafts accepted + bonus token)
- Partial match (some accepted, rejection at various positions)
- Total mismatch (nothing accepted, bonus from position 0)
- Edge cases (empty drafts, single draft, K=1)
- Cache trimming behavior
- Stats tracking
- verify_with_last_token (correct alignment algorithm)
- Sampler-based verification
"""

import pytest
import mlx.core as mx

from yunshu_engine.spec_draft_verifier import (
    SpecDraftVerifier,
    VerifyResult,
    _find_acceptance_boundary,
    _compute_bonus_token,
    _trim_cache,
)


# ── Helpers ──

VOCAB_SIZE = 100


class MockCache:
    """Minimal mock KV cache that supports trimming."""

    def __init__(self, n_entries: int = 10):
        self.n_entries = n_entries

    def is_trimmable(self) -> bool:
        return True

    def trim(self, num_tokens: int) -> int:
        actual = min(num_tokens, self.n_entries)
        self.n_entries -= actual
        return actual


class MockModel:
    """Mock MLX model that returns controlled logits.

    Args:
        model_picks: List of token IDs the model should "pick" at each position.
                    When called with input [d0, d1, ..., dK-1], returns logits
                    where argmax at position i equals model_picks[i].
    """

    def __init__(self, model_picks: list[int], vocab_size: int = VOCAB_SIZE):
        self._picks = model_picks
        self._vocab_size = vocab_size
        self.call_count = 0
        self.last_input = None

    def __call__(self, input_ids, cache=None):
        self.call_count += 1
        self.last_input = input_ids
        # input_ids shape: [1, seq_len]
        if hasattr(input_ids, "shape"):
            seq_len = input_ids.shape[-1]
        else:
            seq_len = len(input_ids)

        # Build logits: position i has argmax = model_picks[i % len(model_picks)]
        picks = self._picks
        # If we have fewer picks than seq_len, cycle; if more, truncate
        if len(picks) < seq_len:
            picks = (picks * ((seq_len // len(picks)) + 1))[:seq_len]
        else:
            picks = picks[:seq_len]

        logits = mx.full((1, seq_len, self._vocab_size), -10.0)
        for i, pick in enumerate(picks):
            logits[0, i, pick] = 10.0

        return logits


# ── VerifyResult dataclass tests ──

class TestVerifyResult:
    """Tests for the VerifyResult dataclass."""

    def test_all_fields_present(self):
        result = VerifyResult(
            accepted_tokens=[1, 2, 3],
            accepted_count=3,
            bonus_token=42,
            rejected_count=2,
            rejection_position=3,
            cache_trimmed=2,
            latency_us=100.0,
            all_accepted=False,
        )
        assert result.accepted_tokens == [1, 2, 3]
        assert result.accepted_count == 3
        assert result.bonus_token == 42
        assert result.rejected_count == 2
        assert result.rejection_position == 3
        assert result.cache_trimmed == 2
        assert result.latency_us > 0
        assert result.all_accepted is False

    def test_all_accepted_result(self):
        result = VerifyResult(
            accepted_tokens=[1, 2],
            accepted_count=2,
            bonus_token=99,
            rejected_count=0,
            rejection_position=None,
            cache_trimmed=0,
            latency_us=5.0,
            all_accepted=True,
        )
        assert result.all_accepted is True
        assert result.rejection_position is None
        assert result.rejected_count == 0


# ── Helper function tests ──

class TestFindAcceptanceBoundary:
    """Tests for _find_acceptance_boundary."""

    def test_perfect_match(self):
        accepted, rej = _find_acceptance_boundary([5, 10, 15], [5, 10, 15], 3)
        assert accepted == [5, 10, 15]
        assert rej is None

    def test_partial_match_at_1(self):
        accepted, rej = _find_acceptance_boundary([5, 99, 15], [5, 10, 15], 3)
        assert accepted == [5]
        assert rej == 1

    def test_no_match(self):
        accepted, rej = _find_acceptance_boundary([99, 10, 15], [5, 10, 15], 3)
        assert accepted == []
        assert rej == 0

    def test_last_position_mismatch(self):
        accepted, rej = _find_acceptance_boundary([5, 10, 99], [5, 10, 15], 3)
        assert accepted == [5, 10]
        assert rej == 2

    def test_empty(self):
        accepted, rej = _find_acceptance_boundary([], [], 0)
        assert accepted == []
        assert rej is None


class TestComputeBonusToken:
    """Tests for _compute_bonus_token."""

    def test_rejection_gives_model_pick(self):
        # Rejection at position 2, model pick was 99
        bonus = _compute_bonus_token([5, 10, 99, 20, 25], 2, 5)
        assert bonus == 99

    def test_all_accepted_gives_last_pick(self):
        bonus = _compute_bonus_token([5, 10, 15], None, 3)
        assert bonus == 15

    def test_empty_returns_none(self):
        bonus = _compute_bonus_token([], None, 0)
        assert bonus is None

    def test_rejection_at_0(self):
        bonus = _compute_bonus_token([42, 10, 15], 0, 3)
        assert bonus == 42


# ── SpecDraftVerifier.verify() tests ──

class TestVerify:
    """Tests for SpecDraftVerifier.verify()."""

    def setup_method(self):
        self.verifier = SpecDraftVerifier(track_stats=True)

    def test_perfect_match_all_accepted(self):
        """All draft tokens match model picks — all accepted + bonus."""
        model = MockModel([5, 10, 15])  # Model agrees with all drafts
        draft_ids = [5, 10, 15]
        cache = [MockCache(10)]

        result = self.verifier.verify(model, draft_ids, cache)

        assert result.accepted_count == 3
        assert result.accepted_tokens == [5, 10, 15]
        assert result.bonus_token == 15  # Last model pick
        assert result.rejection_position is None
        assert result.all_accepted is True
        assert result.rejected_count == 0
        assert result.cache_trimmed == 0
        assert model.call_count == 1

    def test_partial_match_rejection_at_1(self):
        """First token matches, second doesn't — 1 accepted + bonus."""
        model = MockModel([5, 99, 15])  # Rejects at position 1
        draft_ids = [5, 10, 15]
        cache = [MockCache(10)]

        result = self.verifier.verify(model, draft_ids, cache)

        assert result.accepted_count == 1
        assert result.accepted_tokens == [5]
        assert result.bonus_token == 99  # Model's pick at rejection
        assert result.rejection_position == 1
        assert result.all_accepted is False
        assert result.rejected_count == 2
        assert result.cache_trimmed == 2  # Trimmed 2 rejected entries

    def test_total_mismatch(self):
        """No tokens match — 0 accepted + bonus from position 0."""
        model = MockModel([99, 88, 77])
        draft_ids = [5, 10, 15]
        cache = [MockCache(10)]

        result = self.verifier.verify(model, draft_ids, cache)

        assert result.accepted_count == 0
        assert result.accepted_tokens == []
        assert result.bonus_token == 99
        assert result.rejection_position == 0
        assert result.all_accepted is False
        assert result.rejected_count == 3
        assert result.cache_trimmed == 3

    def test_rejection_at_last_position(self):
        """All but last token match — K-1 accepted + bonus."""
        model = MockModel([5, 10, 99])
        draft_ids = [5, 10, 15]
        cache = [MockCache(10)]

        result = self.verifier.verify(model, draft_ids, cache)

        assert result.accepted_count == 2
        assert result.accepted_tokens == [5, 10]
        assert result.bonus_token == 99
        assert result.rejection_position == 2
        assert result.rejected_count == 1
        assert result.cache_trimmed == 1

    def test_single_draft_accepted(self):
        """Single draft token that matches — accepted + bonus."""
        model = MockModel([42])
        draft_ids = [42]
        cache = [MockCache(10)]

        result = self.verifier.verify(model, draft_ids, cache)

        assert result.accepted_count == 1
        assert result.accepted_tokens == [42]
        assert result.bonus_token == 42
        assert result.all_accepted is True

    def test_single_draft_rejected(self):
        """Single draft token that doesn't match — rejected + bonus."""
        model = MockModel([99])
        draft_ids = [42]
        cache = [MockCache(10)]

        result = self.verifier.verify(model, draft_ids, cache)

        assert result.accepted_count == 0
        assert result.accepted_tokens == []
        assert result.bonus_token == 99
        assert result.rejection_position == 0
        assert result.rejected_count == 1
        assert result.cache_trimmed == 1

    def test_empty_draft_ids(self):
        """Empty draft list — nothing to verify."""
        model = MockModel([])
        cache = [MockCache(10)]

        result = self.verifier.verify(model, [], cache)

        assert result.accepted_count == 0
        assert result.accepted_tokens == []
        assert result.bonus_token is None
        assert result.cache_trimmed == 0
        assert result.all_accepted is True  # Vacuously true

    def test_no_cache_trimming_when_all_accepted(self):
        """Cache not trimmed when all drafts accepted."""
        model = MockModel([1, 2, 3])
        cache = [MockCache(10)]

        result = self.verifier.verify(model, [1, 2, 3], cache)

        assert result.cache_trimmed == 0
        assert cache[0].n_entries == 10  # Unchanged

    def test_cache_trimmed_correctly(self):
        """Cache trimmed by rejected_count."""
        model = MockModel([1, 99, 3])  # Reject at pos 1
        cache = [MockCache(20)]

        result = self.verifier.verify(model, [1, 2, 3], cache)

        # 2 rejected -> cache trimmed by 2
        assert result.cache_trimmed == 2
        assert cache[0].n_entries == 18

    def test_latency_is_positive(self):
        """Verification latency is measured and positive."""
        model = MockModel([5, 10])
        cache = [MockCache(10)]

        result = self.verifier.verify(model, [5, 10], cache)

        assert result.latency_us > 0

    def test_model_called_once(self):
        """Model forward pass is called exactly once per verify."""
        model = MockModel([5])
        cache = [MockCache(10)]

        self.verifier.verify(model, [5], cache)
        assert model.call_count == 1

        self.verifier.verify(model, [5], cache)
        assert model.call_count == 2


# ── Stats tracking tests ──

class TestVerifierStats:
    """Tests for SpecDraftVerifier statistics tracking."""

    def test_stats_accumulate(self):
        verifier = SpecDraftVerifier(track_stats=True)
        model = MockModel([5, 10])
        cache = [MockCache(20)]

        # First verify: 2 drafts, 2 accepted
        verifier.verify(model, [5, 10], cache)
        stats = verifier.stats
        assert stats["total_proposals"] == 2
        assert stats["total_accepted"] == 2
        assert stats["total_bonus"] == 1
        assert stats["total_verifications"] == 1
        assert stats["acceptance_rate"] == 1.0

    def test_stats_accumulate_multiple_calls(self):
        verifier = SpecDraftVerifier(track_stats=True)
        model_match = MockModel([5, 10])
        model_miss = MockModel([99, 88])
        cache = [MockCache(50)]

        verifier.verify(model_match, [5, 10], cache)
        verifier.verify(model_miss, [5, 10], cache)

        stats = verifier.stats
        assert stats["total_proposals"] == 4
        assert stats["total_accepted"] == 2  # 2 from first, 0 from second
        assert stats["total_bonus"] == 2  # One per call
        assert stats["total_verifications"] == 2
        assert stats["acceptance_rate"] == 0.5

    def test_stats_disabled(self):
        verifier = SpecDraftVerifier(track_stats=False)
        model = MockModel([5])
        cache = [MockCache(10)]

        verifier.verify(model, [5], cache)

        stats = verifier.stats
        assert stats["total_proposals"] == 0
        assert stats["total_accepted"] == 0

    def test_reset_stats(self):
        verifier = SpecDraftVerifier(track_stats=True)
        model = MockModel([5])
        cache = [MockCache(10)]

        verifier.verify(model, [5], cache)
        assert verifier.stats["total_proposals"] == 1

        verifier.reset_stats()
        assert verifier.stats["total_proposals"] == 0
        assert verifier.stats["acceptance_rate"] == 0.0


# ── verify_with_last_token() tests ──

class TestVerifyWithLastToken:
    """Tests for the correct-alignment verify_with_last_token method."""

    def setup_method(self):
        self.verifier = SpecDraftVerifier(track_stats=True)

    def test_perfect_match(self):
        """All K drafts match — K accepted + bonus from position K."""
        # model_picks has K+1 entries: [d0, d1, d2, bonus]
        model = MockModel([5, 10, 15, 99])
        draft_ids = [5, 10, 15]
        cache = [MockCache(20)]

        result = self.verifier.verify_with_last_token(
            model, last_token_id=0, draft_ids=draft_ids, prompt_cache=cache
        )

        assert result.accepted_count == 3
        assert result.accepted_tokens == [5, 10, 15]
        assert result.bonus_token == 99  # From position K=3
        assert result.all_accepted is True
        assert result.rejection_position is None
        # Cache: trimmed 1 (rollback) + 0 (no rejections)
        assert result.cache_trimmed == 1

    def test_partial_match(self):
        """Some drafts match — accepted + bonus from rejection point."""
        # model_picks: [5, 99, 15, 20] — d0=5 matches, d1=10 doesn't (99)
        model = MockModel([5, 99, 15, 20])
        draft_ids = [5, 10, 15]
        cache = [MockCache(20)]

        result = self.verifier.verify_with_last_token(
            model, last_token_id=0, draft_ids=draft_ids, prompt_cache=cache
        )

        assert result.accepted_count == 1
        assert result.accepted_tokens == [5]
        assert result.bonus_token == 99  # Model's pick at rejection position 1
        assert result.rejection_position == 1
        assert result.rejected_count == 2
        # Cache: trimmed 1 (rollback) + 2 (rejected drafts)
        assert result.cache_trimmed == 3

    def test_total_mismatch(self):
        """First draft doesn't match — 0 accepted + bonus."""
        model = MockModel([99, 88, 77, 66])
        draft_ids = [5, 10, 15]
        cache = [MockCache(20)]

        result = self.verifier.verify_with_last_token(
            model, last_token_id=0, draft_ids=draft_ids, prompt_cache=cache
        )

        assert result.accepted_count == 0
        assert result.accepted_tokens == []
        assert result.bonus_token == 99
        assert result.rejection_position == 0
        assert result.rejected_count == 3
        assert result.cache_trimmed == 4  # 1 rollback + 3 rejected

    def test_single_draft_accepted(self):
        """K=1 draft that matches — 1 accepted + bonus from position 1."""
        model = MockModel([42, 99])
        draft_ids = [42]
        cache = [MockCache(20)]

        result = self.verifier.verify_with_last_token(
            model, last_token_id=0, draft_ids=draft_ids, prompt_cache=cache
        )

        assert result.accepted_count == 1
        assert result.accepted_tokens == [42]
        assert result.bonus_token == 99  # From position K=1
        assert result.all_accepted is True
        assert result.cache_trimmed == 1  # Just the rollback

    def test_single_draft_rejected(self):
        """K=1 draft that doesn't match — 0 accepted + bonus from position 0."""
        model = MockModel([99, 50])
        draft_ids = [42]
        cache = [MockCache(20)]

        result = self.verifier.verify_with_last_token(
            model, last_token_id=0, draft_ids=draft_ids, prompt_cache=cache
        )

        assert result.accepted_count == 0
        assert result.bonus_token == 99
        assert result.rejection_position == 0
        assert result.cache_trimmed == 2  # 1 rollback + 1 rejected

    def test_empty_drafts(self):
        """Empty draft list — nothing to verify."""
        model = MockModel([])
        cache = [MockCache(10)]

        result = self.verifier.verify_with_last_token(
            model, last_token_id=42, draft_ids=[], prompt_cache=cache
        )

        assert result.accepted_count == 0
        assert result.bonus_token is None
        assert result.cache_trimmed == 0

    def test_model_receives_correct_input(self):
        """Model should receive [last_token, d0, d1, ..., dK-1]."""
        model = MockModel([5, 10, 15, 20])
        draft_ids = [5, 10, 15]
        cache = [MockCache(20)]

        self.verifier.verify_with_last_token(
            model, last_token_id=42, draft_ids=draft_ids, prompt_cache=cache
        )

        # Model was called once
        assert model.call_count == 1
        # The input should be [last_token, d0, d1, ..., dK-1]
        # Model receives shape [1, K+1], values = [42, 5, 10, 15]
        inp = model.last_input
        assert inp is not None
        # Flatten and compare
        input_list = inp.flatten().tolist()
        assert input_list == [42, 5, 10, 15]

    def test_rollback_and_rejection_trimming(self):
        """Verify cache trimming accounts for rollback + rejected drafts."""
        # model picks: [d0 matches, d1 rejects, ...] -> 1 accepted, 2 rejected
        model = MockModel([5, 99, 77, 66])
        draft_ids = [5, 10, 15]
        cache = [MockCache(50)]

        result = self.verifier.verify_with_last_token(
            model, last_token_id=0, draft_ids=draft_ids, prompt_cache=cache
        )

        assert result.accepted_count == 1
        assert result.cache_trimmed == 3  # 1 rollback + 2 rejected


# ── Sampler-based verification tests ──

class TestVerifyWithSampler:
    """Tests using a custom sampler instead of argmax."""

    def setup_method(self):
        self.verifier = SpecDraftVerifier(track_stats=True)

    def test_greedy_sampler(self):
        """Greedy sampler (argmax) gives same result as default."""
        model = MockModel([5, 10, 15])
        cache = [MockCache(10)]

        def greedy_sampler(logprobs):
            return mx.argmax(logprobs, axis=-1)

        result = self.verifier.verify(
            model, [5, 10, 15], cache, sampler=greedy_sampler
        )

        assert result.accepted_count == 3
        assert result.all_accepted is True

    def test_forced_sampler(self):
        """Sampler that always picks a specific token."""
        model = MockModel([5, 10, 15])
        cache = [MockCache(10)]

        def force_token_99(logprobs):
            # Always return token 99
            return mx.array([99, 99, 99])

        result = self.verifier.verify(
            model, [5, 10, 15], cache, sampler=force_token_99
        )

        # None match since sampler forces 99
        assert result.accepted_count == 0
        assert result.bonus_token == 99
        assert result.rejection_position == 0


# ── Large draft batch tests ──

class TestLargeDraftBatches:
    """Tests with larger draft batches (K > 5)."""

    def setup_method(self):
        self.verifier = SpecDraftVerifier(track_stats=True)

    def test_10_drafts_all_accepted(self):
        """K=10 drafts, all match — high acceptance count."""
        picks = list(range(10))
        model = MockModel(picks)
        cache = [MockCache(30)]

        result = self.verifier.verify(model, picks, cache)

        assert result.accepted_count == 10
        assert result.all_accepted is True
        assert result.bonus_token == 9

    def test_10_drafts_half_accepted(self):
        """K=10 drafts, 5 match then 5 reject."""
        # Alternating: first 5 match, then mismatch
        picks = list(range(5)) + [99, 99, 99, 99, 99]
        drafts = list(range(10))
        model = MockModel(picks)
        cache = [MockCache(30)]

        result = self.verifier.verify(model, drafts, cache)

        assert result.accepted_count == 5
        assert result.accepted_tokens == [0, 1, 2, 3, 4]
        assert result.rejection_position == 5
        assert result.bonus_token == 99
        assert result.cache_trimmed == 5


# ── Multiple cache objects test ──

class TestMultipleCacheObjects:
    """Tests with multiple KV cache objects (simulating multi-layer model)."""

    def setup_method(self):
        self.verifier = SpecDraftVerifier(track_stats=True)

    def test_trim_all_cache_layers(self):
        """All cache layers are trimmed."""
        model = MockModel([5, 99])
        cache = [MockCache(20), MockCache(20), MockCache(20)]

        result = self.verifier.verify(model, [5, 10], cache)

        assert result.accepted_count == 1
        assert result.cache_trimmed == 1
        # All cache layers should have been trimmed
        for c in cache:
            assert c.n_entries == 19


# ── Integration pattern tests ──

class TestIntegrationPattern:
    """Tests that mimic the actual usage pattern in batched_engine."""

    def setup_method(self):
        self.verifier = SpecDraftVerifier(track_stats=True)

    def test_multi_step_generation_pattern(self):
        """Simulate multi-step generation with alternating accept/reject."""
        # Step 1: 3 drafts, all accepted
        model1 = MockModel([10, 20, 30])
        cache = [MockCache(50)]
        result1 = self.verifier.verify(model1, [10, 20, 30], cache)

        assert result1.accepted_count == 3
        assert result1.bonus_token == 30

        # Step 2: 2 drafts, 1 accepted then rejected
        model2 = MockModel([50, 99])
        result2 = self.verifier.verify(model2, [50, 60], cache)

        assert result2.accepted_count == 1
        assert result2.bonus_token == 99

        # Step 3: 4 drafts, all rejected
        model3 = MockModel([1, 2, 3, 4])
        result3 = self.verifier.verify(model3, [5, 6, 7, 8], cache)

        assert result3.accepted_count == 0
        assert result3.bonus_token == 1

        # Total: 4 accepted out of 9 proposals
        stats = self.verifier.stats
        assert stats["total_proposals"] == 9
        assert stats["total_accepted"] == 4
        assert stats["total_verifications"] == 3

    def test_verify_with_last_token_multi_step(self):
        """Multi-step with verify_with_last_token, carrying forward the bonus."""
        # Step 1: last_token=0, drafts [10, 20, 30], all match
        # model_picks needs 4 entries: [10, 20, 30, bonus]
        model1 = MockModel([10, 20, 30, 99])
        cache = [MockCache(50)]
        result1 = self.verifier.verify_with_last_token(
            model1, last_token_id=0, draft_ids=[10, 20, 30], prompt_cache=cache
        )

        assert result1.accepted_count == 3
        assert result1.bonus_token == 99

        # Step 2: last_token=99, drafts [40, 50], partially match
        # model_picks: [40, 99, bonus] — d0=40 matches, d1=50 rejected
        model2 = MockModel([40, 99, 77])
        result2 = self.verifier.verify_with_last_token(
            model2, last_token_id=99, draft_ids=[40, 50], prompt_cache=cache
        )

        assert result2.accepted_count == 1
        assert result2.accepted_tokens == [40]
        assert result2.bonus_token == 99


# ── Cache trimming edge cases ──

class TestCacheTrimming:
    """Tests for KV cache trimming behavior."""

    def test_no_trim_when_all_accepted(self):
        """Cache not trimmed when all drafts accepted."""
        model = MockModel([5, 10])
        cache = [MockCache(10)]

        result = SpecDraftVerifier().verify(model, [5, 10], cache)

        assert result.cache_trimmed == 0

    def test_trim_preserves_accepted(self):
        """Cache trimmed by exactly the rejected count."""
        model = MockModel([5, 99, 99])
        cache = [MockCache(30)]

        result = SpecDraftVerifier().verify(model, [5, 10, 15], cache)

        # 2 rejected -> trim 2
        assert result.cache_trimmed == 2
        assert cache[0].n_entries == 28

    def test_trim_with_empty_cache(self):
        """Empty cache list — no crash."""
        model = MockModel([5, 99])
        cache = []

        result = SpecDraftVerifier().verify(model, [5, 10], cache)

        assert result.accepted_count == 1
        assert result.cache_trimmed == 0  # Empty cache, nothing to trim

    def test_trim_non_trimmable_cache(self):
        """Cache without trim method — no crash, trimmed=0."""
        model = MockModel([5, 99])

        class NonTrimmableCache:
            def is_trimmable(self):
                return False

        cache = [NonTrimmableCache()]

        result = SpecDraftVerifier().verify(model, [5, 10], cache)

        assert result.accepted_count == 1
        assert result.cache_trimmed == 0
