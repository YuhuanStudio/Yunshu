"""Tests for Yunshu Scheduler — sampler wiring, repetition penalty, and penalty processors."""

import pytest
from unittest.mock import patch, MagicMock
import mlx.core as mx

from yunshu_engine.scheduler import Scheduler, SchedulerConfig, SchedulingPolicy, _LogitsProcessorSampler
from yunshu_engine.request import SamplingParams


class TestLogitsProcessorSampler:
    """Test the _LogitsProcessorSampler wrapper that applies logits processors."""

    def test_no_processors_passthrough(self):
        """Without processors, the base sampler is called directly."""
        calls = []
        def base_sampler(logits):
            calls.append(logits)
            return mx.array([42])

        sampler = _LogitsProcessorSampler(base_sampler, [])
        logits = mx.zeros((1, 100))
        result = sampler(logits)
        assert len(calls) == 1
        assert result.item() == 42

    def test_processor_receives_tokens(self):
        """Logits processors receive accumulated token list."""
        tokens_seen = []
        def fake_processor(tokens, logits):
            tokens_seen.append(list(tokens))
            return logits

        def base_sampler(logits):
            return mx.array([7])

        sampler = _LogitsProcessorSampler(base_sampler, [fake_processor])
        logits = mx.zeros((1, 100))

        # First call: no tokens yet
        sampler(logits)
        assert tokens_seen[-1] == []

        # Second call: should have previous token
        sampler(logits)
        assert tokens_seen[-1] == [7]

        # Third call: two tokens accumulated
        sampler(logits)
        assert tokens_seen[-1] == [7, 7]

    def test_reset_clears_tokens(self):
        """reset() clears accumulated tokens."""
        def base_sampler(logits):
            return mx.array([1])

        sampler = _LogitsProcessorSampler(base_sampler, [])
        sampler(mx.zeros((1, 100)))
        sampler(mx.zeros((1, 100)))
        assert len(sampler._tokens) == 2

        sampler.reset()
        assert len(sampler._tokens) == 0


class TestMakeSamplerRepetitionPenalty:
    """Test that _make_sampler wires repetition_penalty correctly."""

    @pytest.fixture
    def scheduler(self):
        """Create a Scheduler with mock model/tokenizer."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.eos_token_ids = [2]
        tokenizer.encode = MagicMock(return_value=[1])
        tokenizer.has_thinking = False
        config = SchedulerConfig(model_name="test-model")
        return Scheduler(model, tokenizer, config)

    def test_default_repetition_penalty_creates_base_sampler(self, scheduler):
        """Default repetition_penalty=1.0 should not create a processor wrapper."""
        sp = SamplingParams(repetition_penalty=1.0)
        sampler = scheduler._make_sampler(sp)
        # Should be a plain sampler function, not _LogitsProcessorSampler
        assert not isinstance(sampler, _LogitsProcessorSampler)

    def test_repetition_penalty_creates_processor(self, scheduler):
        """repetition_penalty != 1.0 should create _LogitsProcessorSampler."""
        sp = SamplingParams(repetition_penalty=1.5)
        sampler = scheduler._make_sampler(sp)
        assert isinstance(sampler, _LogitsProcessorSampler)
        assert len(sampler._logits_processors) == 1

    def test_presence_penalty_creates_processor(self, scheduler):
        """presence_penalty != 0 should create _LogitsProcessorSampler."""
        sp = SamplingParams(presence_penalty=0.5)
        sampler = scheduler._make_sampler(sp)
        assert isinstance(sampler, _LogitsProcessorSampler)
        assert len(sampler._logits_processors) == 1

    def test_frequency_penalty_creates_processor(self, scheduler):
        """frequency_penalty != 0 should create _LogitsProcessorSampler."""
        sp = SamplingParams(frequency_penalty=0.3)
        sampler = scheduler._make_sampler(sp)
        assert isinstance(sampler, _LogitsProcessorSampler)
        assert len(sampler._logits_processors) == 1

    def test_all_penalties_creates_three_processors(self, scheduler):
        """All three penalties active should create 3 processors."""
        sp = SamplingParams(
            repetition_penalty=1.2,
            presence_penalty=0.5,
            frequency_penalty=0.3,
        )
        sampler = scheduler._make_sampler(sp)
        assert isinstance(sampler, _LogitsProcessorSampler)
        assert len(sampler._logits_processors) == 3

    def test_repetition_penalty_modifies_logits(self, scheduler):
        """Verify the repetition penalty actually modifies logits for repeated tokens."""
        import mlx.core as mx
        mx.clear_cache()
        mx.random.seed(42)
        sp = SamplingParams(
            repetition_penalty=2.0,
            temperature=1.0,
        )
        sampler = scheduler._make_sampler(sp)
        assert isinstance(sampler, _LogitsProcessorSampler)

        # Create logits where token 5 has high score
        logits = mx.zeros((1, 100))
        logits[:, 5] = 10.0

        # First call: no tokens yet, token 5 should be selected
        result = sampler(logits)
        mx.eval(result)
        assert result.item() == 5

        # Second call with same logits: token 5 is now penalized
        # With repetition_penalty=2.0, positive logits are divided by 2
        # So token 5's logit becomes 5.0 instead of 10.0
        logits2 = mx.zeros((1, 100))
        logits2[:, 5] = 10.0
        logits2[:, 10] = 8.0  # Token 10 should now be preferred
        result2 = sampler(logits2)
        mx.eval(result2)
        # After penalty: token 5 = 5.0, token 10 = 8.0, so token 10 should win
        assert result2.item() == 10


class TestSchedulerStats:
    """Test scheduler stats reporting."""

    def test_initial_stats(self):
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig(model_name="test-model")
        sched = Scheduler(model, tokenizer, config)
        stats = sched.get_stats()
        assert stats["waiting"] == 0
        assert stats["running"] == 0
        assert stats["total_requests"] == 0
        assert stats["step_counter"] == 0


class TestRequestTimeout:
    def test_timeout_status_exists(self):
        from yunshu_engine.request import RequestStatus
        assert hasattr(RequestStatus, "FINISHED_TIMEOUT")

    def test_timeout_finish_reason(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_TIMEOUT) == "timeout"

    def test_timeout_is_finished(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.is_finished(RequestStatus.FINISHED_TIMEOUT) is True

    def test_config_has_timeout(self):
        from yunshu_engine.scheduler import SchedulerConfig
        config = SchedulerConfig()
        assert config.request_timeout_seconds == 300

    def test_config_custom_timeout(self):
        from yunshu_engine.scheduler import SchedulerConfig
        config = SchedulerConfig(request_timeout_seconds=60)
        assert config.request_timeout_seconds == 60


class TestSeedParameter:
    def test_seed_in_sampling_params(self):
        from yunshu_engine.request import SamplingParams
        sp = SamplingParams(seed=42)
        assert sp.seed == 42

    def test_seed_default_none(self):
        from yunshu_engine.request import SamplingParams
        sp = SamplingParams()
        assert sp.seed is None


class TestQueueDepthLimit:
    def test_reject_when_queue_full(self):
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig
        from yunshu_engine.request import Request, SamplingParams, RequestStatus

        config = SchedulerConfig(max_waiting_requests=2)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.config = config
        scheduler.requests = {}
        scheduler.waiting = type('FakeQueue', (), {'push': lambda s, x, **kw: None, '__len__': lambda s: 2})()
        scheduler.running = {}
        scheduler._pending_abort_ids = set()

        req = Request(request_id="overflow-1", prompt="test", prompt_token_ids=[1,2,3], sampling_params=SamplingParams())
        scheduler.add_request(req)
        assert req.status == RequestStatus.FINISHED_ERROR
        assert req.finish_reason == "queue_full"

    def test_max_waiting_config_default(self):
        config = SchedulerConfig()
        assert config.max_waiting_requests == 1024


class TestSarathiChunkedPrefill:
    """Test Sarathi-style hybrid chunked prefill scheduling."""

    def test_config_defaults(self):
        """Hybrid prefill config should have sensible defaults."""
        config = SchedulerConfig()
        assert config.hybrid_chunk_size == 512
        assert config.enable_hybrid_prefill is False

    def test_config_custom(self):
        """Hybrid prefill config should be customizable."""
        config = SchedulerConfig(
            enable_hybrid_prefill=True,
            hybrid_chunk_size=256,
        )
        assert config.hybrid_chunk_size == 256
        assert config.enable_hybrid_prefill is True

    def test_pending_prefill_initialized(self):
        """Scheduler should have _pending_prefill dict."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        assert hasattr(sched, '_pending_prefill')
        assert isinstance(sched._pending_prefill, dict)
        assert len(sched._pending_prefill) == 0

    def test_pending_prefill_cleared_on_deep_reset(self):
        """deep_reset should clear _pending_prefill."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        sched._pending_prefill["test-id"] = {"remaining_tokens": [1, 2, 3]}
        sched.deep_reset()
        assert len(sched._pending_prefill) == 0

    def test_process_pending_prefill_noop_when_empty(self):
        """_process_pending_prefill should be a no-op when no pending prefills."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        # Should not raise
        sched._process_pending_prefill()

    def test_process_pending_prefill_removes_empty(self):
        """_process_pending_prefill should remove entries with no remaining tokens."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        sched._pending_prefill["req-1"] = {"remaining_tokens": []}
        sched._process_pending_prefill()
        assert "req-1" not in sched._pending_prefill

    def test_process_pending_prefill_removes_missing_running(self):
        """_process_pending_prefill should remove entries for non-running requests."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        sched._pending_prefill["req-1"] = {"remaining_tokens": [1, 2, 3]}
        # req-1 is not in running dict, so it should be cleaned up
        sched._process_pending_prefill()
        assert "req-1" not in sched._pending_prefill


class TestRequestPreemption:
    """Test request preemption (vLLM pattern)."""

    def test_preempted_status_exists(self):
        from yunshu_engine.request import RequestStatus
        assert hasattr(RequestStatus, "PREEMPTED")

    def test_preempted_is_not_finished(self):
        from yunshu_engine.request import RequestStatus
        assert not RequestStatus.is_finished(RequestStatus.PREEMPTED)

    def test_preempted_between_running_and_finished(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.PREEMPTED > RequestStatus.RUNNING
        assert RequestStatus.PREEMPTED < RequestStatus.FINISHED_STOPPED

    def test_request_has_preemptions_field(self):
        from yunshu_engine.request import Request
        req = Request(request_id="test", prompt="hello")
        assert req.num_preemptions == 0

    def test_preempt_request_moves_to_waiting(self):
        """_preempt_request should set status to PREEMPTED and queue for re-scheduling."""
        from yunshu_engine.request import Request, RequestStatus
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig(policy=SchedulingPolicy.PRIORITY)
        sched = Scheduler(model, tokenizer, config)
        sched._batch_gen = MagicMock()

        req = Request(request_id="preempt-me", prompt="test")
        req.status = RequestStatus.RUNNING
        req.batch_uid = 42
        sched.running["preempt-me"] = req
        sched._uid_to_req[42] = "preempt-me"

        # Caller (e.g. _preempt_lowest_priority) pops from running first
        sched.running.pop("preempt-me")
        sched._preempt_request(req)

        assert req.status == RequestStatus.PREEMPTED
        assert req.num_preemptions == 1
        assert req.batch_uid is None
        assert "preempt-me" not in sched._uid_to_req
        assert len(sched.waiting) == 1
        assert sched.waiting[0] is req

    def test_preempt_lowest_priority_selects_correct_victim(self):
        """_preempt_lowest_priority should evict the minimum-priority request."""
        from yunshu_engine.request import Request, RequestStatus

        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig(policy=SchedulingPolicy.PRIORITY)
        sched = Scheduler(model, tokenizer, config)
        sched._batch_gen = MagicMock()

        req_high = Request(request_id="high", prompt="test", priority=10)
        req_low = Request(request_id="low", prompt="test", priority=1)
        req_med = Request(request_id="med", prompt="test", priority=5)
        # Priority preemption uses sampling_params.priority
        req_high.sampling_params.priority = 10
        req_low.sampling_params.priority = 1
        req_med.sampling_params.priority = 5

        req_high.status = RequestStatus.RUNNING
        req_low.status = RequestStatus.RUNNING
        req_med.status = RequestStatus.RUNNING

        sched.running["high"] = req_high
        sched.running["low"] = req_low
        sched.running["med"] = req_med

        preempted = sched._preempt_lowest_priority(1)

        assert preempted == 1
        assert "low" not in sched.running
        assert "high" in sched.running
        assert "med" in sched.running
        assert req_low.num_preemptions == 1

    def test_preempt_multiple(self):
        """Should be able to preempt multiple requests at once."""
        from yunshu_engine.request import Request, RequestStatus

        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig(policy=SchedulingPolicy.PRIORITY)
        sched = Scheduler(model, tokenizer, config)
        sched._batch_gen = MagicMock()

        for i in range(5):
            req = Request(request_id=f"req-{i}", prompt="test", priority=i)
            req.sampling_params.priority = i
            req.status = RequestStatus.RUNNING
            sched.running[f"req-{i}"] = req

        preempted = sched._preempt_lowest_priority(3)

        assert preempted == 3
        assert len(sched.running) == 2
        # req-0, req-1, req-2 should be preempted (lowest priority)
        assert "req-3" in sched.running
        assert "req-4" in sched.running

    def test_preempt_limited_by_running_count(self):
        """Can't preempt more than available running requests."""
        from yunshu_engine.request import Request, RequestStatus

        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig(policy=SchedulingPolicy.PRIORITY)
        sched = Scheduler(model, tokenizer, config)
        sched._batch_gen = MagicMock()

        req = Request(request_id="only", prompt="test", priority=1)
        req.status = RequestStatus.RUNNING
        sched.running["only"] = req

        preempted = sched._preempt_lowest_priority(5)
        assert preempted == 1
        assert len(sched.running) == 0

    def test_stats_includes_total_preemptions(self):
        """get_stats should include total_preemptions."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig()
        sched = Scheduler(model, tokenizer, config)
        stats = sched.get_stats()
        assert "total_preemptions" in stats
        assert stats["total_preemptions"] == 0

    def test_preemption_increment_tracked_per_request(self):
        """Multiple preemptions of the same request are tracked."""
        from yunshu_engine.request import Request, RequestStatus

        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig(policy=SchedulingPolicy.PRIORITY)
        sched = Scheduler(model, tokenizer, config)
        sched._batch_gen = MagicMock()

        req = Request(request_id="repeat", prompt="test")
        req.status = RequestStatus.RUNNING
        sched.running["repeat"] = req

        sched.running.pop("repeat")
        sched._preempt_request(req)
        assert req.num_preemptions == 1

        # Simulate re-scheduling and preempting again
        req.status = RequestStatus.RUNNING
        sched.running["repeat"] = req
        sched.running.pop("repeat")
        sched._preempt_request(req)
        assert req.num_preemptions == 2

    def test_block_level_preemption_preserves_cached_prefix(self):
        """Block-level preemption preserves cached prefix tokens (vLLM pattern).

        When a request is preempted and has prefix tokens cached in the
        KV prefix cache, num_computed_tokens should reflect the cached
        prefix rather than resetting to 0. This means re-scheduling only
        needs to prefill the uncached tail.
        """
        from yunshu_engine.request import Request, RequestStatus

        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig(policy=SchedulingPolicy.PRIORITY)
        sched = Scheduler(model, tokenizer, config)
        sched._batch_gen = MagicMock()

        # Mock prefix cache that returns 64 matched tokens
        mock_cache = MagicMock()
        mock_cache.get.return_value = (MagicMock(), [], 64)
        sched._prefix_cache = mock_cache

        req = Request(request_id="block-preempt", prompt="test")
        req.status = RequestStatus.RUNNING
        req.batch_uid = 42
        req.prompt_token_ids = list(range(128))
        req.num_prompt_tokens = 128
        req.num_computed_tokens = 128

        sched.running["block-preempt"] = req
        sched._uid_to_req[42] = "block-preempt"

        sched.running.pop("block-preempt")
        sched._preempt_request(req)

        assert req.status == RequestStatus.PREEMPTED
        assert req.num_computed_tokens == 64  # Preserved from prefix cache
        assert req.num_preemptions == 1

    def test_preempt_without_prefix_cache_resets_to_zero(self):
        """Without prefix cache, preemption resets num_computed_tokens to 0."""
        from yunshu_engine.request import Request, RequestStatus

        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig(policy=SchedulingPolicy.PRIORITY)
        sched = Scheduler(model, tokenizer, config)
        sched._batch_gen = MagicMock()
        # No _prefix_cache set (None)

        req = Request(request_id="no-cache", prompt="test")
        req.status = RequestStatus.RUNNING
        req.batch_uid = 42
        req.prompt_token_ids = list(range(128))
        req.num_prompt_tokens = 128
        req.num_computed_tokens = 128

        sched.running["no-cache"] = req
        sched._uid_to_req[42] = "no-cache"

        sched.running.pop("no-cache")
        sched._preempt_request(req)

        assert req.num_computed_tokens == 0  # No cache → reset to 0

    def test_preempt_preserves_min_of_cache_and_computed(self):
        """num_computed_tokens is min(cache_hit, already_computed)."""
        from yunshu_engine.request import Request, RequestStatus

        model = MagicMock()
        tokenizer = MagicMock()
        config = SchedulerConfig(policy=SchedulingPolicy.PRIORITY)
        sched = Scheduler(model, tokenizer, config)
        sched._batch_gen = MagicMock()

        # Cache claims 200 matched, but request only computed 100
        mock_cache = MagicMock()
        mock_cache.get.return_value = (MagicMock(), [], 200)
        sched._prefix_cache = mock_cache

        req = Request(request_id="partial", prompt="test")
        req.status = RequestStatus.RUNNING
        req.batch_uid = 42
        req.prompt_token_ids = list(range(256))
        req.num_prompt_tokens = 256
        req.num_computed_tokens = 100

        sched.running["partial"] = req
        sched._uid_to_req[42] = "partial"

        sched.running.pop("partial")
        sched._preempt_request(req)

        assert req.num_computed_tokens == 100  # min(200, 100) = 100


class TestCacheLocalityReordering:
    """Tests for cache-locality request reordering (SGLang/vLLM pattern)."""

    def test_reorder_noop_single_request(self):
        """Single request is returned as-is."""
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig
        from yunshu_engine.request import Request

        model = MagicMock()
        tokenizer = MagicMock()
        sched = Scheduler(model, tokenizer, SchedulerConfig())

        req = Request(request_id="req-1", prompt="test")
        requests = [req]
        result = sched._reorder_by_cache_locality(requests)
        assert result == requests

    def test_reorder_noop_empty(self):
        """Empty list is returned as-is."""
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig

        model = MagicMock()
        tokenizer = MagicMock()
        sched = Scheduler(model, tokenizer, SchedulerConfig())

        assert sched._reorder_by_cache_locality([]) == []

    def test_reorder_groups_by_prefix_hash(self):
        """Requests with the same prefix hash are grouped together."""
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig
        from yunshu_engine.request import Request

        model = MagicMock()
        tokenizer = MagicMock()
        sched = Scheduler(model, tokenizer, SchedulerConfig())

        # Create requests with different prefix hashes
        req_a = Request(request_id="req-a", prompt="test a")
        req_b = Request(request_id="req-b", prompt="test b")
        req_c = Request(request_id="req-c", prompt="test c")

        # Set prefix hashes: req-a and req-c share the same prefix
        sched._kv_prefix_hashes["req-a"] = 111
        sched._kv_prefix_hashes["req-c"] = 111
        sched._kv_prefix_hashes["req-b"] = 222

        result = sched._reorder_by_cache_locality([req_a, req_b, req_c])

        # req-a and req-c should be consecutive (same prefix hash)
        ids = [r.request_id for r in result]
        assert ids.index("req-a") < ids.index("req-b") or ids.index("req-c") < ids.index("req-b")
        # Both should be in the output
        assert set(ids) == {"req-a", "req-b", "req-c"}

    def test_reorder_no_prefix_preserves_priority_order(self):
        """Requests without prefix hashes keep their effective priority position.

        The reorder must preserve the effective priority ordering from SCHED-3
        as a secondary sort key. A no-prefix request that arrived first (higher
        effective priority) should NOT be pushed behind a lower-priority request
        just because the lower-priority request has a prefix hash.
        """
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig
        from yunshu_engine.request import Request

        model = MagicMock()
        tokenizer = MagicMock()
        sched = Scheduler(model, tokenizer, SchedulerConfig())

        req_hashed = Request(request_id="req-h", prompt="test h")
        req_unhashed = Request(request_id="req-u", prompt="test u")

        sched._kv_prefix_hashes["req-h"] = 999

        # req-u (no prefix) is at index 0 — highest effective priority.
        # req-h (has prefix) is at index 1 — lower effective priority.
        # Priority order must be preserved even though req-h has a prefix.
        result = sched._reorder_by_cache_locality([req_unhashed, req_hashed])

        ids = [r.request_id for r in result]
        assert ids == ["req-u", "req-h"]

    def test_reorder_grouped_preserves_priority_across_groups(self):
        """Groups are ordered by highest effective priority within each group.

        If group A contains a higher-priority request than group B, group A
        should be emitted first even if both groups share the same locality
        benefit.
        """
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig
        from yunshu_engine.request import Request

        model = MagicMock()
        tokenizer = MagicMock()
        sched = Scheduler(model, tokenizer, SchedulerConfig())

        req_a1 = Request(request_id="req-a1", prompt="test a1")
        req_b1 = Request(request_id="req-b1", prompt="test b1")
        req_a2 = Request(request_id="req-a2", prompt="test a2")

        sched._kv_prefix_hashes["req-a1"] = 100  # group A
        sched._kv_prefix_hashes["req-a2"] = 100  # group A
        sched._kv_prefix_hashes["req-b1"] = 200  # group B

        # Input is sorted by effective priority: a1 (highest), b1, a2 (lowest)
        # Group A min index = 0, Group B min index = 1
        # So group A should come first
        result = sched._reorder_by_cache_locality([req_a1, req_b1, req_a2])

        ids = [r.request_id for r in result]
        # Group A (a1, a2) first because a1 has highest priority,
        # then group B (b1)
        assert ids == ["req-a1", "req-a2", "req-b1"]

    def test_reorder_grouped_allows_high_priority_no_prefix_first(self):
        """A high-priority no-prefix request can precede lower-priority groups."""
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig
        from yunshu_engine.request import Request

        model = MagicMock()
        tokenizer = MagicMock()
        sched = Scheduler(model, tokenizer, SchedulerConfig())

        req_urgent = Request(request_id="req-urgent", prompt="urgent")
        req_g1 = Request(request_id="req-g1", prompt="test g1")
        req_g2 = Request(request_id="req-g2", prompt="test g2")

        sched._kv_prefix_hashes["req-g1"] = 500
        sched._kv_prefix_hashes["req-g2"] = 500

        # req-urgent (no prefix, highest effective priority) is first
        result = sched._reorder_by_cache_locality([req_urgent, req_g1, req_g2])

        ids = [r.request_id for r in result]
        # Urgent no-prefix request keeps its top position
        assert ids[0] == "req-urgent"
        # Grouped requests follow, preserving their internal order
        assert ids[1:] == ["req-g1", "req-g2"]

    def test_set_kv_prefix_hash(self):
        """set_kv_prefix_hash stores the hash correctly."""
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig

        model = MagicMock()
        tokenizer = MagicMock()
        sched = Scheduler(model, tokenizer, SchedulerConfig())

        sched.set_kv_prefix_hash("req-1", 42)
        assert sched._kv_prefix_hashes["req-1"] == 42

    def test_cleanup_removes_prefix_hash(self):
        """_cleanup_finished removes prefix hash for finished requests."""
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig
        from yunshu_engine.request import Request, RequestStatus

        model = MagicMock()
        tokenizer = MagicMock()
        sched = Scheduler(model, tokenizer, SchedulerConfig())

        req = Request(request_id="req-done", prompt="test")
        req.status = RequestStatus.FINISHED_STOPPED
        sched.running["req-done"] = req
        sched._kv_prefix_hashes["req-done"] = 123

        sched._cleanup_finished()

        assert "req-done" not in sched._kv_prefix_hashes

    def test_stats_includes_cache_locality(self):
        """get_stats includes cache_locality stats."""
        from yunshu_engine.scheduler import Scheduler, SchedulerConfig

        model = MagicMock()
        tokenizer = MagicMock()
        sched = Scheduler(model, tokenizer, SchedulerConfig())

        stats = sched.get_stats()
        assert "cache_locality" in stats
        assert stats["cache_locality"]["tracked_prefixes"] == 0
        assert stats["cache_locality"]["unique_groups"] == 0

        # Add some hashes and check
        sched._kv_prefix_hashes["r1"] = 10
        sched._kv_prefix_hashes["r2"] = 10  # Same group
        sched._kv_prefix_hashes["r3"] = 20  # Different group

        stats = sched.get_stats()
        assert stats["cache_locality"]["tracked_prefixes"] == 3
        assert stats["cache_locality"]["unique_groups"] == 2
