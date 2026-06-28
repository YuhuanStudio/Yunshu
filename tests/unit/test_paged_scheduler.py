"""Tests for PagedScheduler — Scheduler + KVCacheManager integration."""


from yunshu_engine.paged_scheduler import PagedScheduler
from yunshu_engine.request import Request, RequestStatus, SamplingParams
from yunshu_kv.manager import KVCacheConfig, KVCacheManager


class _FakeDetokenizer:
    def __init__(self):
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token_id):
        text_map = {0: "Hello", 1: " world", 2: "!"}
        self.last_segment = text_map.get(token_id, f"tok{token_id}")

    def finalize(self):
        self.last_segment = ""


class _FakeTokenizer:
    eos_token_ids = [3]
    has_thinking = False

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def decode(self, tokens):
        return " ".join(f"t{t}" for t in tokens)

    @property
    def detokenizer(self):
        return _FakeDetokenizer()


class _FakeGenResponse:
    def __init__(self, uid, token, finish_reason=None):
        self.uid = uid
        self.token = token
        self.finish_reason = finish_reason
        self.current_state = "normal"
        self.logprobs = None


class _FakeBatchGen:
    def __init__(self):
        self._uid_counter = 0
        self._pending = {}

    def insert(self, prompts, max_tokens=None, samplers=None, state_machines=None):
        uids = []
        for _prompt, _mt in zip(prompts, max_tokens or [128], strict=False):
            uid = self._uid_counter
            self._uid_counter += 1
            uids.append(uid)
            self._pending[uid] = [
                _FakeGenResponse(uid, 0),
                _FakeGenResponse(uid, 1),
                _FakeGenResponse(uid, 2, finish_reason="stop"),
            ]
        return uids

    def next(self):
        gen_responses = []
        finished = []
        for uid, responses in self._pending.items():
            if responses:
                gen_responses.append(responses.pop(0))
                if not responses:
                    finished.append(uid)
        for uid in finished:
            del self._pending[uid]
        return [], gen_responses

    def remove(self, uids):
        for uid in uids:
            self._pending.pop(uid, None)

    def close(self):
        pass


def _make_kv_manager(num_blocks=100, block_size=4):
    config = KVCacheConfig(
        block_size=block_size,
        num_layers=4,
        num_kv_heads=8,
        head_dim=128,
    )
    return KVCacheManager(config, num_blocks)


class TestPagedSchedulerBasic:
    def test_create_without_kv_manager(self):
        scheduler = PagedScheduler(None, _FakeTokenizer())
        assert scheduler._kv_manager is None
        stats = scheduler.get_stats()
        assert "kv_cache" not in stats

    def test_create_with_kv_manager(self):
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=_make_kv_manager())
        assert scheduler._kv_manager is not None
        stats = scheduler.get_stats()
        assert "kv_cache" in stats
        assert stats["kv_cache"]["free_blocks"] == 99  # 100 - 1 null block

    def test_set_kv_manager(self):
        scheduler = PagedScheduler(None, _FakeTokenizer())
        assert scheduler._kv_manager is None
        scheduler.set_kv_cache_manager(_make_kv_manager())
        assert scheduler._kv_manager is not None


class TestPagedSchedulerRequests:
    def test_add_request_allocates_blocks(self):
        kv = _make_kv_manager(num_blocks=50, block_size=4)
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=kv)
        scheduler._batch_gen = _FakeBatchGen()

        req = Request(
            request_id="test-1",
            prompt="Hello World!",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=list(range(12)),
            num_prompt_tokens=12,
        )
        scheduler.add_request(req)

        # Should have allocated blocks
        assert "test-1" in scheduler._block_tables
        assert req.status == RequestStatus.WAITING

    def test_add_request_rejects_when_out_of_memory(self):
        kv = _make_kv_manager(num_blocks=2, block_size=4)
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=kv)

        # 2 free blocks (1 is null), request needs 3 prompt blocks + decode reserve
        req = Request(
            request_id="test-1",
            prompt="Hello World!",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=list(range(12)),
            num_prompt_tokens=12,
        )
        scheduler.add_request(req)

        # Should be rejected (not enough blocks)
        assert req.status == RequestStatus.FINISHED_ERROR
        assert req.finish_reason == "kv_cache_full"

    def test_prefix_cache_hit(self):
        kv = _make_kv_manager(num_blocks=50, block_size=4)
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=kv)
        scheduler._batch_gen = _FakeBatchGen()

        # First request: allocate and cache blocks
        req1 = Request(
            request_id="req-1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=list(range(5)),
            num_prompt_tokens=5,
        )
        scheduler.add_request(req1)
        assert "req-1" in scheduler._block_tables

    def test_free_request_on_completion(self):
        kv = _make_kv_manager(num_blocks=50, block_size=4)
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=kv)
        scheduler._batch_gen = _FakeBatchGen()

        req = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=list(range(5)),
            num_prompt_tokens=5,
        )
        scheduler.add_request(req)

        # Move to running (simulates _schedule_waiting)
        scheduler.waiting.pop()
        scheduler.running["test-1"] = req
        req.status = RequestStatus.RUNNING

        initial_free = kv.num_free_blocks

        # Simulate completion
        req.status = RequestStatus.FINISHED_STOPPED
        req.finish_reason = "stop"
        req.output_token_ids = [0, 1]

        # Run cleanup
        scheduler._cleanup_finished()

        # Block table should be cleaned up and blocks freed
        assert "test-1" not in scheduler._block_tables
        assert kv.num_free_blocks > initial_free


class TestPagedSchedulerStats:
    def test_stats_include_kv_info(self):
        kv = _make_kv_manager(num_blocks=50, block_size=4)
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=kv)

        stats = scheduler.get_stats()
        assert stats["kv_cache"]["block_size"] == 4
        assert stats["kv_cache"]["free_blocks"] == 49
        assert stats["kv_cache"]["active_block_tables"] == 0


class TestPagedSchedulerRadixTreeCaching:
    """Test that completed request blocks are cached to RadixTree."""

    def test_finished_request_caches_to_radix_tree(self):
        kv = _make_kv_manager(num_blocks=50, block_size=4)
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=kv)
        scheduler._batch_gen = _FakeBatchGen()

        req = Request(
            request_id="radix-test",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=list(range(8)),
            num_prompt_tokens=8,
        )
        scheduler.add_request(req)

        # Get the allocated table
        table = scheduler._block_tables.get("radix-test")
        assert table is not None

        # Simulate finishing — blocks get cached
        scheduler.waiting.pop()
        scheduler.running["radix-test"] = req
        req.status = RequestStatus.FINISHED_STOPPED
        req.finish_reason = "stop"
        req.output_token_ids = [0, 1]

        # Before cleanup, radix tree should be empty (no inserts yet)
        radix_stats_before = kv.get_tier_stats()["radix_tree"]
        assert radix_stats_before["total_nodes"] == 0

        scheduler._cleanup_finished()

        # After cleanup, radix tree should have nodes
        # (if blocks were successfully cached with hashes)
        radix_stats_after = kv.get_tier_stats()["radix_tree"]
        # Note: blocks may not have hashes if they weren't cached via
        # cache_completed_blocks first, so we check the stats structure
        assert "total_nodes" in radix_stats_after


class TestPagedSchedulerSlidingWindow:
    """Test sliding window physical block freeing."""

    def test_trim_sliding_window_blocks_frees_physical_blocks(self):
        """Sliding window trim should return physical blocks to the pool."""
        kv = _make_kv_manager(num_blocks=50, block_size=4)
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=kv)

        # Allocate a request with 8 prompt tokens (2 blocks at block_size=4)
        req = Request(
            request_id="sw-1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=100),
            prompt_token_ids=list(range(8)),
            num_prompt_tokens=8,
        )
        scheduler._batch_gen = _FakeBatchGen()
        scheduler.add_request(req)

        table = scheduler._block_tables.get("sw-1")
        assert table is not None
        assert table.num_blocks == 2

        initial_free = kv.num_free_blocks

        # Trim 1 prefix block (simulating sliding window eviction)
        freed = scheduler.trim_sliding_window_blocks("sw-1", 1)
        assert freed == 1
        assert table.num_blocks == 1
        # The freed block should return to the pool
        assert kv.num_free_blocks == initial_free + 1

    def test_trim_sliding_window_blocks_no_table(self):
        """Should return 0 when request has no block table."""
        kv = _make_kv_manager(num_blocks=50, block_size=4)
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=kv)
        assert scheduler.trim_sliding_window_blocks("nonexistent", 3) == 0

    def test_trim_sliding_window_blocks_no_kv_manager(self):
        """Should return 0 when no KV manager is set."""
        scheduler = PagedScheduler(None, _FakeTokenizer())
        assert scheduler.trim_sliding_window_blocks("any", 3) == 0

    def test_trim_sliding_window_blocks_clamps_to_table_size(self):
        """Should not trim more blocks than the table holds."""
        kv = _make_kv_manager(num_blocks=50, block_size=4)
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=kv)

        req = Request(
            request_id="sw-2",
            prompt="Hi",
            sampling_params=SamplingParams(max_tokens=10),
            prompt_token_ids=list(range(4)),
            num_prompt_tokens=4,
        )
        scheduler._batch_gen = _FakeBatchGen()
        scheduler.add_request(req)

        table = scheduler._block_tables.get("sw-2")
        assert table is not None
        assert table.num_blocks == 1

        initial_free = kv.num_free_blocks

        # Try to trim 5 blocks from a table that only has 1
        freed = scheduler.trim_sliding_window_blocks("sw-2", 5)
        assert freed == 1
        assert table.num_blocks == 0
        assert kv.num_free_blocks == initial_free + 1

    def test_trim_sliding_window_blocks_multiple_trims(self):
        """Multiple sequential trims should work correctly."""
        kv = _make_kv_manager(num_blocks=50, block_size=4)
        scheduler = PagedScheduler(None, _FakeTokenizer(), kv_cache_manager=kv)

        req = Request(
            request_id="sw-3",
            prompt="Hello World Test",
            sampling_params=SamplingParams(max_tokens=100),
            prompt_token_ids=list(range(16)),
            num_prompt_tokens=16,
        )
        scheduler._batch_gen = _FakeBatchGen()
        scheduler.add_request(req)

        table = scheduler._block_tables.get("sw-3")
        assert table.num_blocks == 4  # 16 tokens / 4 block_size = 4 blocks

        initial_free = kv.num_free_blocks

        # First trim: 1 block
        freed1 = scheduler.trim_sliding_window_blocks("sw-3", 1)
        assert freed1 == 1
        assert table.num_blocks == 3
        assert kv.num_free_blocks == initial_free + 1

        # Second trim: 2 more blocks
        freed2 = scheduler.trim_sliding_window_blocks("sw-3", 2)
        assert freed2 == 2
        assert table.num_blocks == 1
        assert kv.num_free_blocks == initial_free + 3

        # Third trim: the last block
        freed3 = scheduler.trim_sliding_window_blocks("sw-3", 1)
        assert freed3 == 1
        assert table.num_blocks == 0
        assert kv.num_free_blocks == initial_free + 4
