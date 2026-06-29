"""Tests for memory_monitor.py — system memory monitoring."""

from yunshu_engine.memory_monitor import MemoryMonitor


class TestMemoryMonitor:
    def test_create_monitor(self):
        monitor = MemoryMonitor()
        assert monitor is not None

    def test_get_memory_info(self):
        monitor = MemoryMonitor()
        info = monitor.get_memory_info()
        assert info.total_bytes > 0

    def test_set_model_info(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            num_attention_heads=32,
        )

    def test_set_baseline_memory(self):
        monitor = MemoryMonitor()
        monitor.set_baseline_memory()

    def test_is_under_pressure(self):
        monitor = MemoryMonitor()
        assert not monitor.is_under_pressure(threshold_pct=99.0)

    def test_estimate_block_memory(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(num_layers=32, num_kv_heads=8, head_dim=128)
        block_mem = monitor.estimate_block_memory(block_size=64)
        assert block_mem > 0

    def test_estimate_prompt_kv_bytes(self):
        monitor = MemoryMonitor()
        monitor.set_model_info(num_layers=32, num_kv_heads=8, head_dim=128)
        kv_bytes = monitor.estimate_prompt_kv_bytes(num_tokens=1000)
        assert kv_bytes > 0

    def test_get_stats(self):
        monitor = MemoryMonitor()
        stats = monitor.get_stats()
        assert "total_bytes" in stats
