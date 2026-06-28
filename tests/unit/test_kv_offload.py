"""Tests for kv_offload.py — KV cache offloading between tiers."""

import pytest


class TestKVOffloadImport:
    def test_import_config(self):
        from yunshu_engine.kv_offload import KVOffloadConfig
        config = KVOffloadConfig()
        assert config is not None

    def test_import_offload_request(self):
        from yunshu_engine.kv_offload import OffloadRequest
        assert OffloadRequest is not None

    def test_import_offload_result(self):
        from yunshu_engine.kv_offload import OffloadResult
        assert OffloadResult is not None

    def test_import_manager(self):
        from yunshu_engine.kv_offload import KVOffloadConfig, KVOffloadManager
        config = KVOffloadConfig()
        mgr = KVOffloadManager(config)
        assert mgr is not None

    def test_import_policies(self):
        from yunshu_engine.kv_offload import LRUPolicy, PriorityPolicy, ThresholdPolicy
        assert ThresholdPolicy is not None
        assert LRUPolicy is not None
        assert PriorityPolicy is not None

    @pytest.mark.asyncio
    async def test_manager_start_stop(self):
        from yunshu_engine.kv_offload import KVOffloadConfig, KVOffloadManager
        config = KVOffloadConfig()
        mgr = KVOffloadManager(config)
        await mgr.start()
        await mgr.stop()

    def test_manager_get_stats(self):
        from yunshu_engine.kv_offload import KVOffloadConfig, KVOffloadManager
        config = KVOffloadConfig()
        mgr = KVOffloadManager(config)
        stats = mgr.get_stats()
        assert isinstance(stats, dict)
