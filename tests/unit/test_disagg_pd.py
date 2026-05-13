"""Tests for C20: Disaggregated Prefill/Decode serving."""
import pytest

from yunshu_mesh.disagg_pd import (
    DisaggConfig,
    DisaggNodeInfo,
    DisaggRouter,
    DisaggStats,
    KVTransferRequest,
    NodeRole,
)


class TestNodeRole:
    def test_roles(self):
        assert NodeRole.PREFILL.name == "PREFILL"
        assert NodeRole.DECODE.name == "DECODE"
        assert NodeRole.HYBRID.name == "HYBRID"


class TestDisaggConfig:
    def test_defaults(self):
        cfg = DisaggConfig()
        assert not cfg.enabled
        assert cfg.prefill_threshold_tokens == 512
        assert cfg.kv_transfer_batch_size == 16
        assert cfg.kv_transfer_timeout_ms == 5000.0
        assert cfg.auto_role_detection

    def test_from_env_disabled(self):
        cfg = DisaggConfig.from_env()
        assert not cfg.enabled

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_DISAGG_PD", "1")
        monkeypatch.setenv("YUNSHU_PREFILL_THRESHOLD", "1024")
        cfg = DisaggConfig.from_env()
        assert cfg.enabled
        assert cfg.prefill_threshold_tokens == 1024

    def test_to_dict(self):
        cfg = DisaggConfig(enabled=True)
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert "prefill_threshold_tokens" in d


class TestDisaggNodeInfo:
    def test_to_dict(self):
        info = DisaggNodeInfo(node_id="n1", role=NodeRole.DECODE, memory_gb=192)
        d = info.to_dict()
        assert d["node_id"] == "n1"
        assert d["role"] == "DECODE"
        assert d["memory_gb"] == 192


class TestDisaggStats:
    def test_initial(self):
        s = DisaggStats()
        assert s.total_prefill_requests == 0
        assert s.total_decode_requests == 0

    def test_get_stats(self):
        s = DisaggStats()
        s.total_prefill_requests = 10
        stats = s.get_stats()
        assert stats["total_prefill_requests"] == 10

    def test_reset(self):
        s = DisaggStats()
        s.total_prefill_requests = 10
        s.reset()
        assert s.total_prefill_requests == 0


class TestKVTransferRequest:
    def test_creation(self):
        t = KVTransferRequest(
            request_id="r1",
            source_node="pf1",
            target_node="dc1",
            num_blocks=32,
        )
        assert t.status == "pending"
        assert t.created_at > 0


class TestDisaggRouter:
    def test_add_node_hybrid(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("n1", NodeRole.HYBRID)
        assert "n1" in router.get_stats()["nodes"]

    def test_add_node_auto_role(self):
        router = DisaggRouter()
        # 192GB + 40 cores → DECODE
        router.add_node("n1", memory_gb=192, gpu_cores=40)
        stats = router.get_stats()
        assert stats["decode_nodes"] == 1

    def test_add_node_auto_role_prefill(self):
        router = DisaggRouter()
        # 64GB + 10 cores → PREFILL
        router.add_node("n2", memory_gb=64, gpu_cores=10)
        stats = router.get_stats()
        assert stats["prefill_nodes"] == 1

    def test_add_node_auto_role_hybrid(self):
        router = DisaggRouter()
        # 32GB + 8 cores → HYBRID
        router.add_node("n3", memory_gb=32, gpu_cores=8)
        stats = router.get_stats()
        assert stats["hybrid_nodes"] == 1

    def test_remove_node(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("n1", NodeRole.HYBRID)
        router.remove_node("n1")
        assert "n1" not in router.get_stats()["nodes"]

    def test_route_short_request(self):
        router = DisaggRouter(DisaggConfig(
            auto_role_detection=False,
            prefill_threshold_tokens=512,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)
        node_id, role = router.route_request(prompt_tokens=100)
        # Short request should go to decode node
        assert node_id == "dc1"

    def test_route_long_request(self):
        router = DisaggRouter(DisaggConfig(
            auto_role_detection=False,
            prefill_threshold_tokens=512,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)
        node_id, role = router.route_request(prompt_tokens=1024)
        # Long request should go to prefill node
        assert node_id == "pf1"

    def test_route_fallback_to_hybrid(self):
        router = DisaggRouter(DisaggConfig(
            auto_role_detection=False,
            prefill_threshold_tokens=512,
        ))
        router.add_node("h1", NodeRole.HYBRID)
        node_id, role = router.route_request(prompt_tokens=100)
        assert node_id == "h1"

    def test_route_no_nodes(self):
        router = DisaggRouter()
        node_id, role = router.route_request(prompt_tokens=100)
        assert node_id == ""

    def test_mark_unavailable(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("n1", NodeRole.HYBRID)
        router.mark_unavailable("n1")
        node_id, _ = router.route_request(prompt_tokens=100)
        assert node_id == ""

    def test_mark_available(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("n1", NodeRole.HYBRID)
        router.mark_unavailable("n1")
        router.mark_available("n1")
        node_id, _ = router.route_request(prompt_tokens=100)
        assert node_id == "n1"

    def test_kv_transfer(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)
        transfer = router.request_kv_transfer("r1", "pf1", "dc1", 64)
        assert transfer.status == "pending"
        assert len(router.get_pending_transfers()) == 1

    def test_complete_kv_transfer(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)
        router.request_kv_transfer("r1", "pf1", "dc1", 64)
        router.complete_kv_transfer("r1", success=True)
        assert len(router.get_pending_transfers()) == 0
        assert router.stats.total_kv_transfers == 1

    def test_failed_kv_transfer(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("pf1", NodeRole.PREFILL)
        router.request_kv_transfer("r1", "pf1", "dc1", 64)
        router.complete_kv_transfer("r1", success=False)
        assert router.stats.kv_transfer_failures == 1

    def test_update_node_load(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("n1", NodeRole.HYBRID)
        router.update_node_load("n1", active_prefills=3, active_decodes=5)
        stats = router.get_stats()
        assert stats["nodes"]["n1"]["active_prefills"] == 3
        assert stats["nodes"]["n1"]["active_decodes"] == 5

    def test_least_loaded_selection(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("dc1", NodeRole.DECODE)
        router.add_node("dc2", NodeRole.DECODE)
        router.update_node_load("dc1", active_decodes=5)
        router.update_node_load("dc2", active_decodes=1)
        node_id, _ = router.route_request(prompt_tokens=10)
        assert node_id == "dc2"  # Less loaded

    def test_utilization(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("dc1", NodeRole.DECODE)
        router.add_node("dc2", NodeRole.DECODE)
        router.update_node_load("dc1", active_decodes=4)
        router.update_node_load("dc2", active_decodes=2)
        router.compute_utilization()
        assert router.stats.decode_node_utilization == 3.0  # (4+2)/2

    def test_get_stats(self):
        router = DisaggRouter()
        stats = router.get_stats()
        assert "config" in stats
        assert "nodes" in stats
        assert "pending_transfers" in stats
        assert "prefill_nodes" in stats
        assert "decode_nodes" in stats
        assert "hybrid_nodes" in stats
        assert "stats" in stats

    def test_reset(self):
        router = DisaggRouter(DisaggConfig(auto_role_detection=False))
        router.add_node("pf1", NodeRole.PREFILL)
        router.route_request(1024)
        router.reset()
        assert router.stats.total_prefill_requests == 0
        assert len(router.get_pending_transfers()) == 0

    def test_full_pipeline(self):
        """Test complete request lifecycle: classify → route → transfer → decode."""
        router = DisaggRouter(DisaggConfig(
            auto_role_detection=False,
            prefill_threshold_tokens=256,
        ))
        router.add_node("pf1", NodeRole.PREFILL)
        router.add_node("dc1", NodeRole.DECODE)

        # Long request → prefill
        pf_node, pf_role = router.route_request(prompt_tokens=1024)
        assert pf_node == "pf1"

        # After prefill, transfer KV to decode
        transfer = router.request_kv_transfer("req-1", "pf1", "dc1", 128)
        assert transfer.status == "pending"

        # Complete transfer
        router.complete_kv_transfer("req-1", success=True)
        assert router.stats.total_kv_transfers == 1

        # Now decode continues on dc1
        router.update_node_load("dc1", active_decodes=1)
        stats = router.get_stats()
        assert stats["stats"]["total_prefill_requests"] == 1
        assert stats["nodes"]["dc1"]["active_decodes"] == 1
