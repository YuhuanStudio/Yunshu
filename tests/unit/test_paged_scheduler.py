"""Tests for PagedScheduler — Scheduler + KVCacheManager integration."""

import pytest

from yunshu_engine.paged_scheduler import PagedScheduler
from yunshu_engine.scheduler import SchedulerConfig
from yunshu_engine.request import Request, RequestStatus, SamplingParams
from yunshu_kv.manager import KVCacheConfig, KVCacheManager
from yunshu_kv.block import BlockPool


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
        for prompt, mt in zip(prompts, max_tokens or [128]):
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
        scheduler.waiting.popleft()
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
