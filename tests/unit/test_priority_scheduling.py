"""Priority scheduling tests."""

from yunshu_engine.request import SamplingParams
from yunshu_engine.scheduler import SchedulerConfig, SchedulingPolicy


class TestSchedulingPolicy:
    def test_fcfs_is_default(self):
        config = SchedulerConfig()
        assert config.policy == SchedulingPolicy.FCFS

    def test_priority_config(self):
        config = SchedulerConfig(policy=SchedulingPolicy.PRIORITY)
        assert config.policy == SchedulingPolicy.PRIORITY

    def test_max_num_seqs_default(self):
        config = SchedulerConfig()
        assert config.max_num_seqs == 256

    def test_max_num_seqs_custom(self):
        config = SchedulerConfig(max_num_seqs=64)
        assert config.max_num_seqs == 64


class TestSamplingParamsPriority:
    def test_default_priority(self):
        sp = SamplingParams()
        assert sp.priority == 0

    def test_custom_priority(self):
        sp = SamplingParams(priority=10)
        assert sp.priority == 10

    def test_negative_priority(self):
        sp = SamplingParams(priority=-5)
        assert sp.priority == -5

    def test_priority_sorting(self):
        """Test that requests sort correctly by priority."""
        params = [
            SamplingParams(priority=0),
            SamplingParams(priority=5),
            SamplingParams(priority=2),
            SamplingParams(priority=10),
            SamplingParams(priority=1),
        ]
        sorted_params = sorted(params, key=lambda s: s.priority, reverse=True)
        priorities = [p.priority for p in sorted_params]
        assert priorities == [10, 5, 2, 1, 0]
