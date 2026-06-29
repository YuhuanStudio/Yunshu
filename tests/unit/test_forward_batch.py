"""Tests for forward_batch.py — multi-level batch representation."""

import time

from yunshu_engine.forward_batch import (
    BatchComposer,
    BatchResult,
    ForwardBatch,
    RequestSlot,
    ScheduleBatch,
)


class TestRequestSlot:
    def test_total_tokens(self):
        slot = RequestSlot(
            request_id="r1", prompt_tokens=[1, 2, 3], num_prompt_tokens=3
        )
        assert slot.total_tokens == 3
        slot.generated_tokens = [4, 5]
        assert slot.total_tokens == 5

    def test_remaining_tokens(self):
        slot = RequestSlot(request_id="r1", prompt_tokens=[], max_tokens=10)
        slot.generated_tokens = [1, 2, 3]
        assert slot.remaining_tokens == 7

    def test_is_finished_by_max_tokens(self):
        slot = RequestSlot(request_id="r1", prompt_tokens=[], max_tokens=3)
        slot.generated_tokens = [1, 2, 3]
        assert slot.is_finished

    def test_is_finished_by_eos(self):
        slot = RequestSlot(
            request_id="r1", prompt_tokens=[], max_tokens=100, eos_token_ids=[2, 0]
        )
        slot.generated_tokens = [1, 2, 0]
        assert slot.is_finished

    def test_is_not_finished(self):
        slot = RequestSlot(
            request_id="r1", prompt_tokens=[], max_tokens=100, eos_token_ids=[999]
        )
        slot.generated_tokens = [1, 2, 3]
        assert not slot.is_finished

    def test_ttft_ms(self):
        slot = RequestSlot(
            request_id="r1",
            prompt_tokens=[],
            arrival_time=100.0,
            first_token_time=100.5,
        )
        assert slot.ttft_ms == 500.0

    def test_ttft_ms_none(self):
        slot = RequestSlot(request_id="r1", prompt_tokens=[])
        assert slot.ttft_ms is None

    def test_default_values(self):
        slot = RequestSlot(request_id="r1", prompt_tokens=[1, 2])
        assert slot.priority == 0
        assert slot.is_prefill is True
        assert slot.spec_draft_tokens == []
        assert slot.thinking_budget is None


class TestScheduleBatch:
    def test_empty_batch(self):
        batch = ScheduleBatch()
        assert batch.num_slots == 0
        assert batch.total_tokens == 0

    def test_add_and_get_slot(self):
        batch = ScheduleBatch()
        slot = RequestSlot(
            request_id="r1", prompt_tokens=[1, 2, 3], num_prompt_tokens=3
        )
        batch.add_slot(slot)
        assert batch.num_slots == 1
        assert batch.get_slot("r1") is slot

    def test_remove_slot(self):
        batch = ScheduleBatch()
        batch.add_slot(RequestSlot(request_id="r1", prompt_tokens=[]))
        batch.add_slot(RequestSlot(request_id="r2", prompt_tokens=[]))
        removed = batch.remove_slot("r1")
        assert removed is not None
        assert removed.request_id == "r1"
        assert batch.num_slots == 1

    def test_remove_nonexistent(self):
        batch = ScheduleBatch()
        assert batch.remove_slot("nonexistent") is None

    def test_prefill_and_decode_slots(self):
        batch = ScheduleBatch()
        batch.add_slot(RequestSlot(request_id="r1", prompt_tokens=[], is_prefill=True))
        batch.add_slot(RequestSlot(request_id="r2", prompt_tokens=[], is_prefill=False))
        batch.add_slot(RequestSlot(request_id="r3", prompt_tokens=[], is_prefill=True))
        assert len(batch.prefill_slots) == 2
        assert len(batch.decode_slots) == 1

    def test_total_tokens_aggregation(self):
        batch = ScheduleBatch()
        batch.add_slot(
            RequestSlot(request_id="r1", prompt_tokens=[1, 2], num_prompt_tokens=2)
        )
        slot2 = RequestSlot(
            request_id="r2",
            prompt_tokens=[3, 4, 5],
            num_prompt_tokens=3,
            is_prefill=False,
        )
        slot2.generated_tokens = [6]
        batch.add_slot(slot2)
        assert batch.total_prompt_tokens == 5
        assert batch.total_generated_tokens == 1
        assert batch.total_tokens == 6

    def test_reorder_by_priority(self):
        batch = ScheduleBatch()
        batch.add_slot(RequestSlot(request_id="r1", prompt_tokens=[], priority=1))
        batch.add_slot(RequestSlot(request_id="r2", prompt_tokens=[], priority=5))
        batch.add_slot(RequestSlot(request_id="r3", prompt_tokens=[], priority=3))
        batch.reorder_by_priority()
        assert [s.request_id for s in batch.slots] == ["r2", "r3", "r1"]

    def test_split_prefill_decode(self):
        batch = ScheduleBatch(max_prefill_batch=2, max_decode_batch=3)
        for i in range(3):
            batch.add_slot(
                RequestSlot(request_id=f"p{i}", prompt_tokens=[], is_prefill=True)
            )
        for i in range(4):
            batch.add_slot(
                RequestSlot(request_id=f"d{i}", prompt_tokens=[], is_prefill=False)
            )
        prefill, decode = batch.split_prefill_decode()
        assert prefill.num_slots == 2  # limited by max_prefill_batch
        assert decode.num_slots == 3  # limited by max_decode_batch

    def test_remove_finished(self):
        batch = ScheduleBatch()
        slot1 = RequestSlot(request_id="r1", prompt_tokens=[], max_tokens=1)
        slot1.generated_tokens = [1]  # finished
        batch.add_slot(slot1)
        batch.add_slot(RequestSlot(request_id="r2", prompt_tokens=[], max_tokens=100))
        finished = batch.remove_finished()
        assert len(finished) == 1
        assert batch.num_slots == 1

    def test_compact(self):
        batch = ScheduleBatch()
        slot = RequestSlot(request_id="r1", prompt_tokens=[], max_tokens=1)
        slot.generated_tokens = [1]
        batch.add_slot(slot)
        batch.add_slot(RequestSlot(request_id="r2", prompt_tokens=[], max_tokens=100))
        batch.compact()
        assert batch.num_slots == 1

    def test_get_stats(self):
        batch = ScheduleBatch()
        batch.add_slot(
            RequestSlot(
                request_id="r1",
                prompt_tokens=[1, 2],
                num_prompt_tokens=2,
                priority=5,
                arrival_time=time.monotonic() - 1.0,
            )
        )
        stats = batch.get_stats()
        assert stats["num_slots"] == 1
        assert stats["num_prefill"] == 1
        assert stats["total_prompt_tokens"] == 2


class TestForwardBatch:
    def test_from_schedule_batch(self):
        batch = ScheduleBatch()
        batch.add_slot(
            RequestSlot(
                request_id="r1",
                prompt_tokens=[1, 2, 3],
                num_prompt_tokens=3,
                is_prefill=True,
            )
        )
        batch.add_slot(
            RequestSlot(
                request_id="r2",
                prompt_tokens=[4, 5],
                num_prompt_tokens=2,
                is_prefill=False,
                spec_draft_tokens=[6, 7],
            )
        )

        # Second slot is decode — only last token used
        slot2 = batch.slots[1]
        slot2.generated_tokens = [8]

        fb = ForwardBatch.from_schedule_batch(batch)
        assert fb.batch_size == 2
        assert fb.total_tokens == 4  # 3 prefill + 1 decode
        assert len(fb.request_ids) == 2
        assert fb.num_prefill == 1
        assert fb.num_decode == 1
        assert fb.has_spec_drafts

    def test_empty_batch(self):
        batch = ScheduleBatch()
        fb = ForwardBatch.from_schedule_batch(batch)
        assert fb.batch_size == 0
        assert fb.total_tokens == 0
        assert fb.input_ids is None

    def test_properties(self):
        fb = ForwardBatch(
            request_ids=["r1", "r2"],
            is_prefill_mask=[True, False],
            spec_draft_lengths=[0, 3],
            batch_size=2,
        )
        assert fb.num_prefill == 1
        assert fb.num_decode == 1
        assert fb.has_spec_drafts

    def test_no_spec_drafts(self):
        fb = ForwardBatch(
            request_ids=["r1"],
            is_prefill_mask=[True],
            spec_draft_lengths=[0],
        )
        assert not fb.has_spec_drafts


class TestBatchResult:
    def test_basic_result(self):
        result = BatchResult(
            request_ids=["r1", "r2"],
            generated_token_ids=[42, 43],
            finish_reasons=["stop", None],
            forward_time_ms=10.0,
            sample_time_ms=2.0,
            total_time_ms=12.0,
            memory_active_bytes=1024 * 1024,
        )
        assert result.batch_size == 2
        stats = result.get_stats()
        assert stats["batch_size"] == 2
        assert stats["forward_time_ms"] == 10.0
        assert stats["memory_active_mb"] == 1.0

    def test_per_request_results(self):
        result = BatchResult(
            request_ids=["r1", "r2"],
            generated_token_ids=[[42], [43]],
            finish_reasons=["stop", "length"],
            spec_accepted_count=[3, 0],
            spec_rejected_count=[1, 0],
        )
        per_req = result.get_per_request_results()
        assert per_req["r1"]["token_ids"] == [42]
        assert per_req["r1"]["finish_reason"] == "stop"
        assert per_req["r1"]["spec_accepted"] == 3
        assert per_req["r2"]["token_ids"] == [43]

    def test_empty_result(self):
        result = BatchResult()
        assert result.batch_size == 0
        assert result.get_stats()["spec_total_accepted"] == 0

    def test_out_of_bounds_access(self):
        result = BatchResult(
            request_ids=["r1"],
            generated_token_ids=[[42]],
        )
        per_req = result.get_per_request_results()
        assert per_req["r1"]["finish_reason"] is None


class TestBatchComposer:
    def test_compose_with_active_decode(self):
        composer = BatchComposer(max_batch_size=4, max_prefill_slots=2)
        active = [
            RequestSlot(
                request_id="d1", prompt_tokens=[], is_prefill=False, max_tokens=100
            ),
            RequestSlot(
                request_id="d2", prompt_tokens=[], is_prefill=False, max_tokens=100
            ),
        ]
        # Decode slots need generated_tokens to not be "finished"
        active[0].generated_tokens = [1]
        active[1].generated_tokens = [1]
        pending = [
            RequestSlot(
                request_id="p1",
                prompt_tokens=[1] * 100,
                num_prompt_tokens=100,
                is_prefill=True,
                priority=1,
            ),
        ]
        batch = composer.compose(pending, active)
        # 2 decode + 1 prefill
        assert batch.num_slots == 3
        assert len(batch.decode_slots) == 2

    def test_priority_ordering(self):
        composer = BatchComposer(max_batch_size=4, max_prefill_slots=4)
        pending = [
            RequestSlot(
                request_id="p1",
                prompt_tokens=[1],
                num_prompt_tokens=1,
                is_prefill=True,
                priority=1,
            ),
            RequestSlot(
                request_id="p2",
                prompt_tokens=[1],
                num_prompt_tokens=1,
                is_prefill=True,
                priority=5,
            ),
            RequestSlot(
                request_id="p3",
                prompt_tokens=[1],
                num_prompt_tokens=1,
                is_prefill=True,
                priority=3,
            ),
        ]
        batch = composer.compose(pending)
        assert batch.slots[0].request_id == "p2"  # highest priority first

    def test_memory_budget_constraint(self):
        composer = BatchComposer(max_batch_size=10, max_prefill_slots=10)
        pending = [
            RequestSlot(
                request_id="p1",
                prompt_tokens=[1] * 100,
                num_prompt_tokens=100,
                is_prefill=True,
            ),
            RequestSlot(
                request_id="p2",
                prompt_tokens=[1] * 200,
                num_prompt_tokens=200,
                is_prefill=True,
            ),
        ]
        batch = composer.compose(pending, memory_budget_tokens=150)
        assert batch.num_slots == 1  # only first fits

    def test_batch_size_limit(self):
        composer = BatchComposer(max_batch_size=2, max_prefill_slots=2)
        pending = [
            RequestSlot(
                request_id=f"p{i}",
                prompt_tokens=[1],
                num_prompt_tokens=1,
                is_prefill=True,
            )
            for i in range(5)
        ]
        batch = composer.compose(pending)
        assert batch.num_slots == 2

    def test_stats(self):
        composer = BatchComposer()
        composer.compose(
            [
                RequestSlot(
                    request_id="p1",
                    prompt_tokens=[1],
                    num_prompt_tokens=1,
                    is_prefill=True,
                ),
            ]
        )
        stats = composer.get_stats()
        assert stats["total_batches_composed"] == 1
        assert stats["total_requests_scheduled"] == 1

    def test_empty_compose(self):
        composer = BatchComposer()
        batch = composer.compose([])
        assert batch.num_slots == 0

    def test_age_bonus_scheduling(self):
        composer = BatchComposer(ttft_weight=10.0, priority_weight=0.0)
        old = RequestSlot(
            request_id="old",
            prompt_tokens=[1],
            num_prompt_tokens=1,
            is_prefill=True,
            arrival_time=time.monotonic() - 5.0,
        )
        new = RequestSlot(
            request_id="new",
            prompt_tokens=[1],
            num_prompt_tokens=1,
            is_prefill=True,
            arrival_time=time.monotonic(),
        )
        batch = composer.compose([new, old])
        assert batch.slots[0].request_id == "old"  # older gets priority
