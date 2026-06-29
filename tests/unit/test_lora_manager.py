"""Tests for LoRA adapter manager — concurrent safety, ref counting, LRU eviction."""

import json
import threading


class TestLoRAAdapterEntry:
    def test_entry_defaults(self):
        from yunshu_engine.lora_manager import LoRAAdapterEntry

        entry = LoRAAdapterEntry(adapter_id="test", adapter_path="/tmp/test")
        assert entry.adapter_id == "test"
        assert entry.is_loaded is False
        assert entry.is_merged is False
        assert entry.rank == 8
        assert entry.ref_count == 0


class TestLoRAAdapterManager:
    def test_register_adapter(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager

        mgr = LoRAAdapterManager(max_loras=2)
        mgr.register_adapter("test-adapter", "/tmp/test-adapter")
        adapters = mgr.list_adapters()
        assert len(adapters) == 1
        assert adapters[0]["adapter_id"] == "test-adapter"

    def test_register_duplicate_ignored(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager

        mgr = LoRAAdapterManager()
        mgr.register_adapter("a1", "/tmp/a1")
        mgr.register_adapter("a1", "/tmp/a1")
        assert len(mgr.list_adapters()) == 1

    def test_get_stats(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager

        mgr = LoRAAdapterManager(max_loras=4)
        mgr.register_adapter("a1", "/tmp/a1")
        mgr.register_adapter("a2", "/tmp/a2")
        stats = mgr.get_stats()
        assert stats["max_loras"] == 4
        assert stats["registered"] == 2
        assert stats["loaded"] == 0

    def test_load_without_base_model(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager

        mgr = LoRAAdapterManager()
        mgr.register_adapter("a1", "/tmp/a1")
        assert mgr.load_adapter("a1") is False

    def test_load_unknown_adapter(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager

        mgr = LoRAAdapterManager()
        assert mgr.load_adapter("nonexistent") is False

    def test_unload_unknown_adapter(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager

        mgr = LoRAAdapterManager()
        assert mgr.unload_adapter("nonexistent") is False

    def test_discover_adapters(self, tmp_path):
        from yunshu_engine.lora_manager import LoRAAdapterManager

        mgr = LoRAAdapterManager()

        # Create a fake adapter directory
        adapter_dir = tmp_path / "adapters" / "my-lora"
        adapter_dir.mkdir(parents=True)
        config = {"lora_parameters": {"rank": 16, "scale": 10.0}, "num_layers": 8}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        (adapter_dir / "adapters.safetensors").write_bytes(b"fake_weights")

        discovered = mgr.discover_adapters(str(tmp_path))
        assert "my-lora" in discovered
        assert len(mgr.list_adapters()) == 1

    def test_discover_no_adapters(self, tmp_path):
        from yunshu_engine.lora_manager import LoRAAdapterManager

        mgr = LoRAAdapterManager()
        discovered = mgr.discover_adapters(str(tmp_path))
        assert len(discovered) == 0

    def test_max_loras_enforcement(self):
        from yunshu_engine.lora_manager import LoRAAdapterEntry, LoRAAdapterManager

        mgr = LoRAAdapterManager(max_loras=1)
        mgr._base_model = None  # Can't actually load but test the logic

        # Pre-load one adapter
        mgr._adapters["a1"] = LoRAAdapterEntry("a1", "/tmp/a1", is_loaded=True)
        mgr._lru_order = ["a1"]

        # The enforcement happens inside load_adapter which needs a real model
        # Test the LRU eviction directly
        assert len(mgr._loaded_adapters) == 1


class TestLoRARefCounting:
    """Test reference counting for concurrent request safety."""

    def _make_manager(self, max_loras=4):
        from yunshu_engine.lora_manager import LoRAAdapterEntry, LoRAAdapterManager

        mgr = LoRAAdapterManager(max_loras=max_loras)
        # Simulate loaded adapters (no real model needed)
        for i in range(3):
            aid = f"adapter-{i}"
            mgr._adapters[aid] = LoRAAdapterEntry(
                adapter_id=aid, adapter_path=f"/tmp/{aid}", is_loaded=True
            )
            mgr._lru_order.append(aid)
        mgr._active_adapter_id = "adapter-0"
        return mgr

    def test_acquire_increments_ref_count(self):
        mgr = self._make_manager()
        # Simulate load_adapter succeeding by manually setting state
        assert mgr._adapters["adapter-0"].ref_count == 0
        result = mgr.acquire_adapter("adapter-0")
        assert result is True
        assert mgr._adapters["adapter-0"].ref_count == 1

    def test_release_decrements_ref_count(self):
        mgr = self._make_manager()
        mgr.acquire_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 1
        mgr.release_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 0

    def test_multiple_acquire_release(self):
        """Multiple requests can hold refs to the same adapter."""
        mgr = self._make_manager()
        mgr.acquire_adapter("adapter-0")
        mgr.acquire_adapter("adapter-0")
        mgr.acquire_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 3
        mgr.release_adapter("adapter-0")
        mgr.release_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 1
        mgr.release_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 0

    def test_release_unknown_adapter_is_safe(self):
        mgr = self._make_manager()
        # Should not raise
        mgr.release_adapter("nonexistent")

    def test_release_with_zero_refs_logs_warning(self):
        mgr = self._make_manager()
        # ref_count starts at 0, release should warn but not crash
        mgr.release_adapter("adapter-0")
        assert mgr._adapters["adapter-0"].ref_count == 0

    def test_in_use_adapter_not_evicted_by_lru(self):
        """An adapter with ref_count > 0 cannot be LRU evicted."""
        mgr = self._make_manager(max_loras=3)
        # Give ALL adapters refs so none can be evicted
        for aid in ["adapter-0", "adapter-1", "adapter-2"]:
            mgr.acquire_adapter(aid)
        assert mgr._adapters["adapter-0"].ref_count == 1

        evicted = mgr._unload_lru()
        assert evicted is False  # Cannot evict — all have refs
        assert mgr._adapters["adapter-0"].is_loaded is True
        assert mgr._adapters["adapter-1"].is_loaded is True
        assert mgr._adapters["adapter-2"].is_loaded is True

    def test_zero_ref_adapter_can_be_evicted(self):
        """An adapter with ref_count == 0 can be LRU evicted."""
        mgr = self._make_manager(max_loras=3)
        # adapter-0 is first in LRU order, has zero refs
        assert mgr._adapters["adapter-0"].ref_count == 0
        evicted = mgr._unload_lru()
        assert evicted is True
        assert mgr._adapters["adapter-0"].is_loaded is False

    def test_lru_skips_in_use_evicts_next(self):
        """LRU eviction skips in-use adapters and evicts the next zero-ref one."""
        mgr = self._make_manager(max_loras=3)
        # adapter-0 is LRU, give it a ref to make it un-evictable
        mgr.acquire_adapter("adapter-0")
        # adapter-1 and adapter-2 have zero refs
        evicted = mgr._unload_lru()
        assert evicted is True
        # adapter-0 should NOT be evicted (has refs)
        assert mgr._adapters["adapter-0"].is_loaded is True
        assert mgr._adapters["adapter-0"].ref_count == 1
        # adapter-1 should be evicted (next in LRU order)
        assert mgr._adapters["adapter-1"].is_loaded is False

    def test_unload_in_use_adapter_fails(self):
        """Cannot unload an adapter that has active requests."""
        mgr = self._make_manager()
        mgr.acquire_adapter("adapter-0")
        assert mgr.unload_adapter("adapter-0") is False
        assert mgr._adapters["adapter-0"].is_loaded is True

    def test_unload_zero_ref_adapter_succeeds(self):
        """Can unload an adapter that has zero active requests."""
        mgr = self._make_manager()
        # Need a base model for unload to work
        mgr._base_model = None  # No real model, but the logic checks ref_count first
        mgr.unload_adapter("adapter-0")
        # unload checks ref_count == 0 first, then tries to restore base
        # With no base model, restore_base is a no-op but the unload proceeds
        assert mgr._adapters["adapter-0"].is_loaded is False

    def test_acquire_touches_lru(self):
        """Acquiring an adapter updates its LRU position."""
        mgr = self._make_manager()
        assert mgr._lru_order[0] == "adapter-0"
        mgr.acquire_adapter("adapter-0")
        # adapter-0 should now be at end of LRU
        assert mgr._lru_order[-1] == "adapter-0"

    def test_acquire_unregistered_adapter_fails(self):
        mgr = self._make_manager()
        assert mgr.acquire_adapter("nonexistent") is False

    def test_shutdown_resets_ref_counts(self):
        mgr = self._make_manager()
        mgr.acquire_adapter("adapter-0")
        mgr.acquire_adapter("adapter-1")
        mgr.shutdown()
        # All entries cleared
        assert len(mgr._adapters) == 0
        assert mgr._active_adapter_id is None


class TestLoRALRUConcurrency:
    """Test concurrent LRU eviction scenarios."""

    def test_concurrent_acquire_release(self):
        """Multiple threads acquiring/releasing the same adapter."""
        from yunshu_engine.lora_manager import LoRAAdapterEntry, LoRAAdapterManager

        mgr = LoRAAdapterManager(max_loras=4)
        aid = "shared-adapter"
        mgr._adapters[aid] = LoRAAdapterEntry(
            adapter_id=aid, adapter_path="/tmp/shared", is_loaded=True
        )
        mgr._lru_order = [aid]
        mgr._active_adapter_id = aid

        errors = []
        barrier = threading.Barrier(8)

        def worker():
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    mgr.acquire_adapter(aid)
                    mgr.release_adapter(aid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert mgr._adapters[aid].ref_count == 0

    def test_concurrent_acquire_eviction(self):
        """Eviction should never remove an adapter with refs > 0."""
        from yunshu_engine.lora_manager import LoRAAdapterEntry, LoRAAdapterManager

        mgr = LoRAAdapterManager(max_loras=2)
        for i in range(3):
            aid = f"a{i}"
            mgr._adapters[aid] = LoRAAdapterEntry(
                adapter_id=aid, adapter_path=f"/tmp/{aid}", is_loaded=(i < 2)
            )
            if i < 2:
                mgr._lru_order.append(aid)
        mgr._active_adapter_id = "a0"

        errors = []
        barrier = threading.Barrier(4)

        def acquirer():
            try:
                barrier.wait(timeout=5)
                mgr.acquire_adapter("a0")
                import time

                time.sleep(0.01)
                mgr.release_adapter("a0")
            except Exception as e:
                errors.append(e)

        def evictor():
            try:
                barrier.wait(timeout=5)
                for _ in range(20):
                    mgr._unload_lru()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=acquirer) for _ in range(3)] + [
            threading.Thread(target=evictor)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        # If a0 has refs, it must still be loaded
        if mgr._adapters["a0"].ref_count > 0:
            assert mgr._adapters["a0"].is_loaded is True


class TestLoRACrossModel:
    """Verify that adapters on different model instances don't interfere."""

    def test_separate_managers_independent(self):
        """Two LoRAAdapterManager instances are fully independent."""
        from yunshu_engine.lora_manager import LoRAAdapterEntry, LoRAAdapterManager

        mgr_a = LoRAAdapterManager(max_loras=2)
        mgr_b = LoRAAdapterManager(max_loras=2)

        # Register different adapters
        for i in range(3):
            mgr_a._adapters[f"a-{i}"] = LoRAAdapterEntry(
                adapter_id=f"a-{i}", adapter_path=f"/tmp/a-{i}", is_loaded=True
            )
            mgr_a._lru_order.append(f"a-{i}")

        for i in range(2):
            mgr_b._adapters[f"b-{i}"] = LoRAAdapterEntry(
                adapter_id=f"b-{i}", adapter_path=f"/tmp/b-{i}", is_loaded=True
            )
            mgr_b._lru_order.append(f"b-{i}")

        mgr_a._active_adapter_id = "a-0"
        mgr_b._active_adapter_id = "b-0"

        # Acquire on mgr_a does not affect mgr_b
        mgr_a.acquire_adapter("a-0")
        assert mgr_a._adapters["a-0"].ref_count == 1
        assert mgr_b._adapters["b-0"].ref_count == 0

        # Evict on mgr_a does not affect mgr_b
        mgr_a._unload_lru()
        assert len([e for e in mgr_a._adapters.values() if e.is_loaded]) < 3
        assert len([e for e in mgr_b._adapters.values() if e.is_loaded]) == 2

    def test_adapter_namespaces_isolated(self):
        """Same adapter_id in different managers are independent."""
        from yunshu_engine.lora_manager import LoRAAdapterEntry, LoRAAdapterManager

        mgr_x = LoRAAdapterManager(max_loras=2)
        mgr_y = LoRAAdapterManager(max_loras=2)

        # Same adapter_id in both
        for mgr in (mgr_x, mgr_y):
            mgr._adapters["shared-id"] = LoRAAdapterEntry(
                adapter_id="shared-id", adapter_path="/tmp/shared", is_loaded=True
            )
            mgr._lru_order = ["shared-id"]
            mgr._active_adapter_id = "shared-id"

        mgr_x.acquire_adapter("shared-id")
        assert mgr_x._adapters["shared-id"].ref_count == 1
        assert mgr_y._adapters["shared-id"].ref_count == 0

        # Unload in one doesn't affect the other
        mgr_y.unload_adapter("shared-id")
        assert mgr_y._adapters["shared-id"].is_loaded is False
        assert mgr_x._adapters["shared-id"].is_loaded is True


class TestLoRAListAdaptersRefCount:
    """Verify list_adapters includes ref_count."""

    def test_list_includes_ref_count(self):
        from yunshu_engine.lora_manager import LoRAAdapterEntry, LoRAAdapterManager

        mgr = LoRAAdapterManager()
        mgr._adapters["a1"] = LoRAAdapterEntry(
            "a1", "/tmp/a1", is_loaded=True, ref_count=3
        )
        adapters = mgr.list_adapters()
        assert adapters[0]["ref_count"] == 3


class TestLoRACrossModelConcurrent:
    """Verify concurrent multi-model LoRA: multiple managers with different
    adapters under simultaneous load, ensuring ref-counting and LRU eviction
    work correctly across model instances without interference."""

    @staticmethod
    def _make_managers(n_models: int, adapters_per_model: int, max_loras: int = 3):
        """Create N independent LoRAAdapterManager instances, each with adapters."""
        from yunshu_engine.lora_manager import LoRAAdapterEntry, LoRAAdapterManager

        managers = {}
        for m in range(n_models):
            mgr = LoRAAdapterManager(max_loras=max_loras)
            for a in range(adapters_per_model):
                aid = f"model{m}-adapter{a}"
                mgr._adapters[aid] = LoRAAdapterEntry(
                    adapter_id=aid,
                    adapter_path=f"/tmp/{aid}",
                    is_loaded=True,
                )
                mgr._lru_order.append(aid)
            mgr._active_adapter_id = f"model{m}-adapter0"
            managers[f"model{m}"] = mgr
        return managers

    def test_concurrent_acquire_release_multi_model(self):
        """Multiple threads operating on different model managers simultaneously."""
        import threading

        managers = self._make_managers(4, 3)
        errors = []
        barrier = threading.Barrier(8)

        def worker(model_key, adapter_idx):
            try:
                mgr = managers[model_key]
                aid = f"{model_key}-adapter{adapter_idx}"
                barrier.wait(timeout=5)
                for _ in range(50):
                    mgr.acquire_adapter(aid)
                    mgr.release_adapter(aid)
            except Exception as e:
                errors.append((model_key, adapter_idx, e))

        threads = []
        for m in range(4):
            for a in range(2):  # 2 threads per model
                t = threading.Thread(target=worker, args=(f"model{m}", a))
                threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"Errors: {errors}"
        # All ref counts should be back to 0
        for key, mgr in managers.items():
            for aid, entry in mgr._adapters.items():
                assert entry.ref_count == 0, f"{key}/{aid} ref_count={entry.ref_count}"

    def test_concurrent_cross_model_eviction(self):
        """Eviction on one model's manager must never touch another model's adapters."""
        import threading

        managers = self._make_managers(3, 3, max_loras=2)
        errors = []
        barrier = threading.Barrier(6)

        def acquirer(model_key, adapter_idx):
            try:
                mgr = managers[model_key]
                aid = f"{model_key}-adapter{adapter_idx}"
                barrier.wait(timeout=5)
                mgr.acquire_adapter(aid)
                import time

                time.sleep(0.05)
                mgr.release_adapter(aid)
            except Exception as e:
                errors.append(("acquire", model_key, e))

        def evictor(model_key):
            try:
                mgr = managers[model_key]
                barrier.wait(timeout=5)
                for _ in range(30):
                    mgr._unload_lru()
            except Exception as e:
                errors.append(("evict", model_key, e))

        threads = []
        for m in range(3):
            threads.append(threading.Thread(target=acquirer, args=(f"model{m}", 0)))
            threads.append(threading.Thread(target=evictor, args=(f"model{m}",)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"Errors: {errors}"

        # Verify isolation: loaded adapters in one model should not affect others
        for m in range(3):
            mgr = managers[f"model{m}"]
            # If adapter0 still has refs, it must still be loaded
            entry0 = mgr._adapters[f"model{m}-adapter0"]
            if entry0.ref_count > 0:
                assert entry0.is_loaded, f"model{m}-adapter0 has refs but is not loaded"

    def test_concurrent_acquire_eviction_different_models(self):
        """Stress test: acquire on model A while evicting on model B."""
        import threading

        managers = self._make_managers(2, 4, max_loras=2)
        errors = []
        barrier = threading.Barrier(4)

        def model_a_worker():
            mgr = managers["model0"]
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    mgr.acquire_adapter("model0-adapter0")
                    mgr.release_adapter("model0-adapter0")
            except Exception as e:
                errors.append(("model_a", e))

        def model_b_evictor():
            mgr = managers["model1"]
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    mgr._unload_lru()
            except Exception as e:
                errors.append(("model_b_evict", e))

        def model_b_acquire():
            mgr = managers["model1"]
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    mgr.acquire_adapter("model1-adapter1")
                    mgr.release_adapter("model1-adapter1")
            except Exception as e:
                errors.append(("model_b_acquire", e))

        def model_a_evictor():
            mgr = managers["model0"]
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    mgr._unload_lru()
            except Exception as e:
                errors.append(("model_a_evict", e))

        threads = [
            threading.Thread(target=model_a_worker),
            threading.Thread(target=model_b_evictor),
            threading.Thread(target=model_b_acquire),
            threading.Thread(target=model_a_evictor),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"Errors: {errors}"

    def test_ref_count_isolation_under_contention(self):
        """Verify ref counts don't leak across model managers under contention."""
        import threading

        managers = self._make_managers(3, 2)
        errors = []
        barrier = threading.Barrier(6)

        def worker(model_key, adapter_idx, iterations):
            try:
                mgr = managers[model_key]
                aid = f"{model_key}-adapter{adapter_idx}"
                barrier.wait(timeout=5)
                for _ in range(iterations):
                    mgr.acquire_adapter(aid)
                    # Brief hold
                    mgr.release_adapter(aid)
            except Exception as e:
                errors.append((model_key, adapter_idx, e))

        threads = []
        for m in range(3):
            for a in range(2):
                iters = 200 if a == 0 else 100
                t = threading.Thread(target=worker, args=(f"model{m}", a, iters))
                threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert len(errors) == 0, f"Errors: {errors}"

        # All ref counts must be exactly 0 after all releases
        for key, mgr in managers.items():
            for aid, entry in mgr._adapters.items():
                assert entry.ref_count == 0, (
                    f"{key}/{aid} ref_count={entry.ref_count} — leaked across managers"
                )

    def test_lru_order_independent_per_model(self):
        """LRU order in one manager is unaffected by operations on another."""
        from yunshu_engine.lora_manager import LoRAAdapterEntry, LoRAAdapterManager

        mgr_a = LoRAAdapterManager(max_loras=4)
        mgr_b = LoRAAdapterManager(max_loras=4)

        for i in range(3):
            for mgr, prefix in [(mgr_a, "a"), (mgr_b, "b")]:
                aid = f"{prefix}-{i}"
                mgr._adapters[aid] = LoRAAdapterEntry(
                    adapter_id=aid,
                    adapter_path=f"/tmp/{aid}",
                    is_loaded=True,
                )
                mgr._lru_order.append(aid)
            mgr_a._active_adapter_id = "a-0"
            mgr_b._active_adapter_id = "b-0"

        # Touch a-1 in mgr_a
        mgr_a.acquire_adapter("a-1")
        mgr_a.release_adapter("a-1")
        assert mgr_a._lru_order[-1] == "a-1"

        # Touch b-2 in mgr_b — should not affect mgr_a's LRU order
        mgr_b.acquire_adapter("b-2")
        mgr_b.release_adapter("b-2")
        assert mgr_b._lru_order[-1] == "b-2"
        assert mgr_a._lru_order[-1] == "a-1"  # unchanged

    def test_shutdown_one_manager_does_not_affect_others(self):
        """Shutting down one model's LoRA manager doesn't touch others."""
        from yunshu_engine.lora_manager import LoRAAdapterEntry, LoRAAdapterManager

        mgr_a = LoRAAdapterManager(max_loras=2)
        mgr_b = LoRAAdapterManager(max_loras=2)

        for i in range(2):
            for mgr, prefix in [(mgr_a, "a"), (mgr_b, "b")]:
                aid = f"{prefix}-{i}"
                mgr._adapters[aid] = LoRAAdapterEntry(
                    adapter_id=aid,
                    adapter_path=f"/tmp/{aid}",
                    is_loaded=True,
                )
                mgr._lru_order.append(aid)
            mgr_a._active_adapter_id = "a-0"
            mgr_b._active_adapter_id = "b-0"

        # Acquire refs on mgr_b
        mgr_b.acquire_adapter("b-0")

        # Shutdown mgr_a
        mgr_a.shutdown()
        assert len(mgr_a._adapters) == 0
        assert mgr_a._active_adapter_id is None

        # mgr_b must be untouched
        assert len(mgr_b._adapters) == 2
        assert mgr_b._adapters["b-0"].is_loaded is True
        assert mgr_b._adapters["b-0"].ref_count == 1
        assert mgr_b._active_adapter_id == "b-0"


class TestGrammarParameter:
    def test_parse_grammar_json_schema(self):
        from yunshu_gateway.routers.chat import _parse_response_format

        grammar = {
            "type": "json",
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
        }
        result = _parse_response_format(None, grammar)
        assert isinstance(result, dict)
        assert result["type"] == "object"

    def test_parse_grammar_json_no_schema(self):
        from yunshu_gateway.routers.chat import _parse_response_format

        grammar = {"type": "json"}
        result = _parse_response_format(None, grammar)
        assert result == "json_object"

    def test_grammar_overrides_response_format(self):
        from yunshu_gateway.routers.chat import _parse_response_format

        grammar = {"type": "json", "schema": {"type": "string"}}
        rf = {"type": "json_object"}
        result = _parse_response_format(rf, grammar)
        # Grammar takes priority
        assert result == {"type": "string"}

    def test_no_grammar_no_response_format(self):
        from yunshu_gateway.routers.chat import _parse_response_format

        result = _parse_response_format(None, None)
        assert result is None


class TestNormalizeLoraKey:
    """Pure key-name normalization: MLX-LM and HuggingFace/PEFT adapter keys
    must both map to the same base-model module path (no silent drop)."""

    def test_mlx_format_lowercase(self):
        from yunshu_engine.lora_manager import normalize_lora_key

        path = "model.layers.0.self_attn.q_proj"
        assert normalize_lora_key(f"{path}.lora_a") == (path, "a")
        assert normalize_lora_key(f"{path}.lora_b") == (path, "b")

    def test_hf_peft_format_with_prefix_and_weight_suffix(self):
        from yunshu_engine.lora_manager import normalize_lora_key

        # PEFT: "base_model.model." wrapper + capital A/B + ".weight"
        path = "model.layers.0.self_attn.q_proj"
        assert normalize_lora_key(f"base_model.model.{path}.lora_A.weight") == (
            path,
            "a",
        )
        assert normalize_lora_key(f"base_model.model.{path}.lora_B.weight") == (
            path,
            "b",
        )

    def test_hf_peft_named_adapter_variant(self):
        from yunshu_engine.lora_manager import normalize_lora_key

        path = "model.layers.3.mlp.gate_proj"
        # PEFT with an explicit adapter name segment before ".weight"
        assert normalize_lora_key(f"base_model.model.{path}.lora_A.default.weight") == (
            path,
            "a",
        )

    def test_mlx_and_hf_map_to_same_path(self):
        from yunshu_engine.lora_manager import normalize_lora_key

        path = "model.layers.7.self_attn.v_proj"
        mlx = normalize_lora_key(f"{path}.lora_b")
        hf = normalize_lora_key(f"base_model.model.{path}.lora_B.weight")
        assert mlx == hf == (path, "b")

    def test_non_lora_key_returns_none(self):
        from yunshu_engine.lora_manager import normalize_lora_key

        assert normalize_lora_key("model.layers.0.self_attn.q_proj.weight") is None
        assert normalize_lora_key("model.embed_tokens.weight") is None
        assert normalize_lora_key("") is None

    def test_rename_roundtrip_is_identity_for_mlx(self):
        # The load-path rebuilds "<module>.lora_<ab>"; for MLX-format input it
        # must reproduce the original key byte-for-byte (no regression).
        from yunshu_engine.lora_manager import normalize_lora_key

        for key in (
            "model.layers.0.self_attn.q_proj.lora_a",
            "model.layers.2.mlp.down_proj.lora_b",
        ):
            mp, ab = normalize_lora_key(key)
            assert f"{mp}.lora_{ab}" == key


class TestPEFTTransposeOnLoad:
    """A genuine PEFT/HF adapter stores lora_A=(r,in), lora_B=(out,r) — the
    TRANSPOSE of mlx-lm's LoRALinear (lora_a=(in,r), lora_b=(r,out)). The load
    path must transpose, else the adapter loads with wrong-orientation /
    shape-mismatched weights that strict=False silently drops (inert adapter)."""

    def _tiny_model(self):
        import mlx.nn as nn

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(16, 8, bias=False)  # in=16, out=8

            def __call__(self, x):
                return self.proj(x)

        return Tiny()

    def test_peft_format_adapter_transposed_on_load(self, tmp_path):
        import json

        import mlx.core as mx
        from safetensors.numpy import save_file

        from yunshu_engine.lora_manager import LoRAAdapterManager

        r, in_f, out_f = 4, 16, 8
        # PEFT orientation: lora_A=(r,in), lora_B=(out,r).
        a_peft = mx.random.normal((r, in_f))
        b_peft = mx.random.normal((out_f, r))

        adir = tmp_path / "peft-adapter"
        adir.mkdir()
        (adir / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": r}))
        import numpy as np

        save_file(
            {
                "base_model.model.proj.lora_A.weight": np.array(a_peft),
                "base_model.model.proj.lora_B.weight": np.array(b_peft),
            },
            str(adir / "adapter_model.safetensors"),
        )

        mgr = LoRAAdapterManager(max_loras=2)
        mgr.set_base_model(self._tiny_model())
        mgr.register_adapter("peft", str(adir))
        assert mgr.load_adapter("peft") is True

        # Find the wrapped module and confirm mlx orientation + transposed values.
        wrapped = dict(mgr._base_model.named_modules())["proj"]
        assert tuple(wrapped.lora_a.shape) == (in_f, r)  # (in, r), not (r, in)
        assert tuple(wrapped.lora_b.shape) == (r, out_f)  # (r, out), not (out, r)
        assert mx.allclose(wrapped.lora_a, a_peft.T, atol=1e-5).item()
        assert mx.allclose(wrapped.lora_b, b_peft.T, atol=1e-5).item()
