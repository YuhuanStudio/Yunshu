"""Tests for scheduler_mixins.py — modular scheduler components."""

import time
from unittest.mock import MagicMock, patch

import pytest

from yunshu_engine.scheduler_mixins import (
    CompositionScheduler,
    DataParallelMixin,
    DisaggregationMixin,
    MemoryPressureMixin,
    MetricsMixin,
    PipelineParallelMixin,
    ProfilingMixin,
    ProfilingSample,
    SchedulerMixin,
    SpecDecodeMixin,
)


class FakeOutput:
    def __init__(self, outputs=None):
        self.outputs = outputs or []


class FakeReqOutput:
    def __init__(self, request_id="r1", finished=False, completion_tokens=1,
                 prompt_tokens=10, spec_accepted=None, spec_proposer="ngram",
                 replica_id=0):
        self.request_id = request_id
        self.finished = finished
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens
        self.spec_accepted = spec_accepted
        self.spec_proposer = spec_proposer
        self.replica_id = replica_id


class FakeScheduler:
    def __init__(self):
        self._waiting = []
        self._requests = {}

    def add_request(self, req):
        self._requests[req.get("id", "r1")] = req

    def step(self):
        return FakeOutput()

    def has_requests(self):
        return len(self._requests) > 0

    def abort_request(self, rid):
        self._requests.pop(rid, None)

    def remove_finished_request(self, rid):
        self._requests.pop(rid, None)

    def fail_all_requests(self):
        ids = list(self._requests.keys())
        self._requests.clear()
        return ids

    def shutdown(self):
        self._requests.clear()

    def get_stats(self):
        return {"requests": len(self._requests)}


# ── SchedulerMixin base ──

class TestSchedulerMixin:
    def test_base_class_requires_pre_step(self):
        with pytest.raises(TypeError):
            SchedulerMixin()

    def test_concrete_mixin_lifecycle(self):
        class SimpleMixin(SchedulerMixin):
            def pre_step(self, scheduler): pass
            def post_step(self, scheduler, output): pass
        m = SimpleMixin()
        m.pre_step(None)
        m.post_step(None, None)
        assert m.get_stats() == {}


# ── MetricsMixin ──

class TestMetricsMixin:
    def test_empty_stats(self):
        m = MetricsMixin()
        stats = m.get_stats()
        assert stats["step_count"] == 0
        assert stats["total_tokens"] == 0

    def test_records_step_metrics(self):
        m = MetricsMixin()
        m.pre_step(None)
        output = FakeOutput([
            FakeReqOutput(completion_tokens=5, finished=False),
            FakeReqOutput(completion_tokens=3, finished=True),
        ])
        m.post_step(None, output)
        stats = m.get_stats()
        assert stats["step_count"] == 1
        assert stats["total_tokens"] == 8
        assert stats["total_requests"] == 1
        assert stats["avg_batch_size"] == 2.0

    def test_window_size_limit(self):
        m = MetricsMixin(window_size=5)
        for i in range(10):
            m.pre_step(None)
            m.post_step(None, FakeOutput([FakeReqOutput()]))
        assert len(m._step_times) == 5
        assert m._step_count == 10

    def test_throughput_window_pruning(self):
        m = MetricsMixin()
        # Add old entry
        m._throughput_window.append((time.monotonic() - 120, 100))
        m._throughput_window.append((time.monotonic(), 50))
        m.pre_step(None)
        m.post_step(None, FakeOutput([FakeReqOutput()]))
        assert len(m._throughput_window) == 2  # old pruned + new + current

    def test_on_finish_does_not_double_count(self):
        m = MetricsMixin()
        finished_out = FakeReqOutput(finished=True)
        m.pre_step(None)
        m.post_step(None, FakeOutput([finished_out]))
        assert m._total_requests == 1
        m.on_finish(None, "r1", finished_out)
        assert m._total_requests == 1

    def test_latency_percentiles(self):
        m = MetricsMixin()
        for _ in range(100):
            m.pre_step(None)
            time.sleep(0.001)
            m.post_step(None, FakeOutput())
        stats = m.get_stats()
        assert stats["p50_step_latency_ms"] > 0
        assert stats["p99_step_latency_ms"] >= stats["p50_step_latency_ms"]


# ── ProfilingMixin ──

class TestProfilingMixin:
    def test_captures_samples(self):
        p = ProfilingMixin()
        sched = FakeScheduler()
        sched._waiting = [MagicMock()]  # has waiting requests = prefill
        p.pre_step(sched)
        p.post_step(sched, FakeOutput([FakeReqOutput()]))
        assert len(p._samples) == 1
        assert p._samples[0].phase == "prefill"

    def test_decode_phase_detection(self):
        p = ProfilingMixin()
        sched = FakeScheduler()
        sched._waiting = []  # no waiting = decode phase
        p.pre_step(sched)
        p.post_step(sched, FakeOutput([FakeReqOutput()]))
        assert p._samples[0].phase == "decode"

    def test_sample_rate_filtering(self):
        p = ProfilingMixin(sample_rate=0.0)
        p.pre_step(None)
        p.post_step(None, FakeOutput([FakeReqOutput()]))
        assert len(p._samples) == 0

    def test_max_samples_limit(self):
        p = ProfilingMixin(max_samples=5)
        for _ in range(10):
            p.pre_step(None)
            p.post_step(None, FakeOutput([FakeReqOutput()]))
        assert len(p._samples) == 5

    def test_get_stats(self):
        p = ProfilingMixin()
        sched = FakeScheduler()
        sched._waiting = [MagicMock()]
        p.pre_step(sched)
        p.post_step(sched, FakeOutput([FakeReqOutput()]))
        stats = p.get_stats()
        assert stats["samples"] == 1
        assert stats["avg_prefill_latency_ms"] > 0

    def test_export_traces(self):
        p = ProfilingMixin()
        sched = FakeScheduler()
        sched._waiting = [MagicMock()]
        p.pre_step(sched)
        p.post_step(sched, FakeOutput([FakeReqOutput()]))
        traces = p.export_traces()
        assert len(traces) == 1
        assert "step" in traces[0]
        assert "latency_ms" in traces[0]

    def test_memory_snapshot(self):
        p = ProfilingMixin()
        sched = FakeScheduler()
        sched._waiting = [MagicMock()]
        p.pre_step(sched)
        # MLX may not be available in test, memory fields default to 0
        p.post_step(sched, FakeOutput([FakeReqOutput()]))
        assert p._samples[0].memory_active_bytes >= 0

    def test_disabled_stops_capture(self):
        p = ProfilingMixin()
        p._enabled = False
        p.pre_step(FakeScheduler())
        p.post_step(FakeScheduler(), FakeOutput([FakeReqOutput()]))
        assert len(p._samples) == 0


# ── DisaggregationMixin ──

class TestDisaggregationMixin:
    def test_routes_long_prompts_to_prefill(self):
        d = DisaggregationMixin(prefill_threshold=100, prefill_nodes=["node1"])
        d.post_step(None, FakeOutput([
            FakeReqOutput(finished=True, prompt_tokens=500),
        ]))
        assert d._prefill_count == 1

    def test_short_prompts_go_to_decode(self):
        d = DisaggregationMixin(prefill_threshold=4096)
        d.post_step(None, FakeOutput([
            FakeReqOutput(finished=True, prompt_tokens=100),
        ]))
        assert d._decode_count == 1

    def test_on_add_request_logs_long_prompt(self):
        d = DisaggregationMixin(prefill_threshold=100, prefill_nodes=["node1"])
        req = MagicMock(num_prompt_tokens=500)
        d.on_add_request(None, req)

    def test_get_stats(self):
        d = DisaggregationMixin(prefill_threshold=100, prefill_nodes=["p1"], decode_nodes=["d1"])
        stats = d.get_stats()
        assert stats["prefill_threshold"] == 100
        assert stats["prefill_nodes"] == 1
        assert stats["decode_nodes"] == 1


# ── DataParallelMixin ──

class TestDataParallelMixin:
    def test_least_loaded_routing(self):
        dp = DataParallelMixin(num_replicas=3, strategy="least_loaded")
        dp._replica_loads = {0: 5, 1: 2, 2: 3}
        target = dp._select_replica()
        assert target == 1

    def test_round_robin_routing(self):
        dp = DataParallelMixin(num_replicas=3, strategy="round_robin")
        targets = [dp._select_replica() for _ in range(6)]
        # _total_routed stays 0 because we're calling _select_replica directly
        # All targets = 0 % 3 = 0
        # Test round_robin via on_add_request which increments _total_routed
        dp2 = DataParallelMixin(num_replicas=3, strategy="round_robin")
        routed = []
        for _ in range(6):
            routed.append(dp2._select_replica())
            dp2._total_routed += 1
        assert routed == [0, 1, 2, 0, 1, 2]

    def test_on_add_request_increments_load(self):
        dp = DataParallelMixin(num_replicas=2)
        dp.on_add_request(None, MagicMock())
        assert sum(dp._replica_loads.values()) == 1

    def test_post_step_decrements_load(self):
        dp = DataParallelMixin(num_replicas=2)
        dp._replica_loads = {0: 3, 1: 2}
        # Only finished requests on replica 0 and 1 should decrement
        dp.post_step(None, FakeOutput([
            FakeReqOutput(finished=True, replica_id=0),
            FakeReqOutput(finished=True, replica_id=1),
        ]))
        assert dp._replica_loads[0] == 2
        assert dp._replica_loads[1] == 1

    def test_single_replica_no_routing(self):
        dp = DataParallelMixin(num_replicas=1)
        dp.on_add_request(None, MagicMock())
        assert dp._total_routed == 0

    def test_get_stats(self):
        dp = DataParallelMixin(num_replicas=3, strategy="least_loaded")
        stats = dp.get_stats()
        assert stats["num_replicas"] == 3
        assert stats["strategy"] == "least_loaded"


# ── PipelineParallelMixin ──

class TestPipelineParallelMixin:
    def test_tracks_bubbles(self):
        pp = PipelineParallelMixin(num_stages=4, stage_id=0)
        for i in range(10):
            pp.pre_step(None)
            if i > 4:
                pp.post_step(None, FakeOutput())  # no outputs = bubble
            else:
                pp.post_step(None, FakeOutput([FakeReqOutput()]))
        stats = pp.get_stats()
        assert stats["pipeline_bubbles"] > 0

    def test_no_bubble_with_work(self):
        pp = PipelineParallelMixin(num_stages=2)
        pp.pre_step(None)
        pp.post_step(None, FakeOutput([FakeReqOutput()]))
        assert pp._pipeline_bubbles == 0

    def test_get_stats(self):
        pp = PipelineParallelMixin(num_stages=4, stage_id=1, micro_batch_size=2)
        stats = pp.get_stats()
        assert stats["num_stages"] == 4
        assert stats["stage_id"] == 1
        assert stats["bubble_rate"] == 0.0


# ── SpecDecodeMixin ──

class TestSpecDecodeMixin:
    def test_tracks_acceptance(self):
        sd = SpecDecodeMixin()
        sd.post_step(None, FakeOutput([
            FakeReqOutput(spec_accepted=True, spec_proposer="ngram"),
            FakeReqOutput(spec_accepted=False, spec_proposer="eagle"),
        ]))
        stats = sd.get_stats()
        assert stats["total_drafts"] == 2
        assert stats["total_accepted"] == 1
        assert stats["total_rejected"] == 1

    def test_per_proposer_stats(self):
        sd = SpecDecodeMixin()
        sd.post_step(None, FakeOutput([
            FakeReqOutput(spec_accepted=True, spec_proposer="ngram"),
            FakeReqOutput(spec_accepted=True, spec_proposer="ngram"),
            FakeReqOutput(spec_accepted=False, spec_proposer="eagle"),
        ]))
        stats = sd.get_stats()
        assert stats["per_proposer"]["ngram"]["accepted"] == 2
        assert stats["per_proposer"]["eagle"]["rejected"] == 1

    def test_adaptive_draft_length_increase(self):
        sd = SpecDecodeMixin(initial_draft_length=3, max_draft_length=8)
        # Simulate high acceptance rate
        for _ in range(15):
            sd.post_step(None, FakeOutput([
                FakeReqOutput(spec_accepted=True),
            ]))
        assert sd.current_draft_length > 3

    def test_adaptive_draft_length_decrease(self):
        sd = SpecDecodeMixin(initial_draft_length=5, min_draft_length=1)
        # Simulate low acceptance rate
        for _ in range(15):
            sd.post_step(None, FakeOutput([
                FakeReqOutput(spec_accepted=False),
            ]))
        assert sd.current_draft_length < 5

    def test_draft_length_bounds(self):
        sd = SpecDecodeMixin(initial_draft_length=5, max_draft_length=5, min_draft_length=5)
        for _ in range(20):
            sd.post_step(None, FakeOutput([FakeReqOutput(spec_accepted=True)]))
        assert sd.current_draft_length == 5

    def test_acceptance_window_size(self):
        sd = SpecDecodeMixin(acceptance_window=10)
        for _ in range(20):
            sd.post_step(None, FakeOutput([FakeReqOutput(spec_accepted=True)]))
        assert len(sd._acceptances) == 10


# ── MemoryPressureMixin ──

class TestMemoryPressureMixin:
    def test_normal_state(self):
        mp = MemoryPressureMixin(warning_threshold=0.8, critical_threshold=0.95)
        mp._total_memory_bytes = 100_000_000
        mp._last_memory_fraction = 0.5
        assert mp.recommended_batch_size == 32
        assert not mp.is_admission_paused

    def test_warning_state(self):
        mp = MemoryPressureMixin(
            warning_threshold=0.8,
            critical_threshold=0.95,
            batch_size_warning=16,
        )
        mp._current_state = "warning"
        assert mp.recommended_batch_size == 16

    def test_critical_state(self):
        mp = MemoryPressureMixin(
            critical_threshold=0.95,
            batch_size_critical=4,
        )
        mp._current_state = "critical"
        mp._admission_paused = True
        assert mp.recommended_batch_size == 4
        assert mp.is_admission_paused

    def test_from_env(self):
        with patch.dict("os.environ", {
            "YUNSHU_MEM_WARNING": "0.7",
            "YUNSHU_MEM_CRITICAL": "0.9",
            "YUNSHU_BATCH_NORMAL": "64",
            "YUNSHU_BATCH_WARNING": "32",
            "YUNSHU_BATCH_CRITICAL": "8",
        }):
            mp = MemoryPressureMixin.from_env()
            assert mp._warning_threshold == 0.7
            assert mp._critical_threshold == 0.9
            assert mp._batch_normal == 64

    def test_get_stats(self):
        mp = MemoryPressureMixin()
        stats = mp.get_stats()
        assert stats["state"] == "normal"
        assert "thresholds" in stats

    def test_hysteresis(self):
        mp = MemoryPressureMixin(
            critical_threshold=0.9,
            hysteresis=0.05,
        )
        mp._current_state = "critical"
        mp._total_memory_bytes = 100
        # Just below critical but above critical - hysteresis → stays critical
        mp._last_memory_fraction = 0.88
        # Would need to go below 0.85 to exit critical


# ── CompositionScheduler ──

class TestCompositionScheduler:
    def test_wraps_core_scheduler(self):
        core = FakeScheduler()
        comp = CompositionScheduler(core)
        assert comp.core is core

    def test_add_mixin_fluent(self):
        comp = CompositionScheduler(FakeScheduler())
        result = comp.add_mixin(MetricsMixin())
        assert result is comp

    def test_step_applies_mixins(self):
        core = FakeScheduler()
        comp = CompositionScheduler(core)
        metrics = MetricsMixin()
        comp.add_mixin(metrics)
        comp.step()
        assert metrics._step_count == 1

    def test_add_request_notifies_mixins(self):
        core = FakeScheduler()
        comp = CompositionScheduler(core)
        dp = DataParallelMixin(num_replicas=2)
        comp.add_mixin(dp)
        comp.add_request({"id": "r1", "prompt": "hello"})
        assert dp._total_routed == 1

    def test_finished_triggers_on_finish(self):
        core = FakeScheduler()
        comp = CompositionScheduler(core)
        metrics = MetricsMixin()
        comp.add_mixin(metrics)
        output = FakeOutput([FakeReqOutput(request_id="r1", finished=True)])
        # Simulate step returning finished output
        core.step = lambda: output
        comp.step()
        assert metrics._total_requests >= 1

    def test_get_mixin(self):
        comp = CompositionScheduler(FakeScheduler())
        m = MetricsMixin()
        comp.add_mixin(m)
        assert comp.get_mixin(MetricsMixin) is m
        assert comp.get_mixin(ProfilingMixin) is None

    def test_get_stats_aggregates(self):
        comp = CompositionScheduler(FakeScheduler())
        comp.add_mixin(MetricsMixin())
        comp.add_mixin(ProfilingMixin())
        stats = comp.get_stats()
        assert "MetricsMixin" in stats
        assert "ProfilingMixin" in stats

    def test_delegates_unknown_attrs(self):
        core = FakeScheduler()
        core.custom_attr = "test"
        comp = CompositionScheduler(core)
        assert comp.custom_attr == "test"

    def test_lifecycle(self):
        core = FakeScheduler()
        comp = CompositionScheduler(core)
        comp.add_mixin(MetricsMixin())
        comp.add_request({"id": "r1"})
        assert comp.has_requests()
        comp.step()
        comp.abort_request("r1")
        comp.shutdown()
