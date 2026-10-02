from __future__ import annotations

"""Engine diagnostics extracted from batched_engine.

Patchable helpers resolve through the compatibility facade. Concrete self types
retain the shared BatchedEngine state; misc ignores allow that mixin self type.
"""

from typing import Any


class EngineDiagnosticsMixin:
    def has_active_requests(self: _engine.BatchedEngine) -> bool:  # type: ignore[misc]
        """Check if engine has in-flight requests (including fast-path)."""
        if getattr(self, "_active_fast_path_count", 0) > 0:
            return True
        if self._engine_core:
            return bool(self._engine_core.has_active_requests)
        return False

    def resolve_model_id(self: _engine.BatchedEngine, model_id: str) -> bool:  # type: ignore[misc]
        """Check if a model ID matches this engine."""
        if not self.model_name:
            return False
        display = (
            self.model_name.rsplit("/", 1)[-1]
            if "/" in self.model_name
            else self.model_name
        )
        known = {display, self.model_name}
        known_lower = {k.lower() for k in known if k}
        if model_id in known or model_id.lower() in known_lower:
            return True
        if "/" in model_id:
            stripped = model_id.rsplit("/", 1)[-1]
            if stripped in known or stripped.lower() in known_lower:
                return True
        return False

    def get_stats(self: _engine.BatchedEngine) -> dict:  # type: ignore[misc]
        if self._engine_core:
            stats = self._engine_core.get_stats()
            stats["model"] = self.model_name
            stats["loaded"] = self._loaded
        else:
            stats = {"model": self.model_name, "loaded": self._loaded}
        if self._adaptive_spec is not None:
            stats["adaptive_spec"] = self._adaptive_spec.get_stats()
        if getattr(self, "_spec_decoder", None) is not None:
            stats["spec_decode"] = {
                **self._spec_decoder._stats,
                "enabled": getattr(self, "_spec_enabled", False),
            }
        if self._ngram_proposer is not None:
            stats["ngram"] = {**self._ngram_stats, **self._ngram_proposer.get_stats()}
        if self._mtp_decoder is not None:
            s = self._mtp_decoder.stats
            stats["mtp"] = {
                "accepts": s.accepts,
                "rejects": s.rejects,
                "cooldowns": s.cooldowns,
                "tokens_generated": s.tokens_generated,
                "total_cycles": s.total_cycles,
            }
        if self._lookahead_reasoning is not None:
            stats["lookahead_reasoning"] = self._lookahead_reasoning.get_stats()
        # Metal kernel stats removed (kernels deleted — slower than mx.fast).
        # ANE embedding co-processor status (when enabled via YUNSHU_ANE_EMBEDDINGS=1)
        try:
            from .ane_embedding import get_ane_embedding_stats

            stats["ane_embeddings"] = get_ane_embedding_stats()
        except Exception:
            _engine.logger.debug("ane embedding stats failed", exc_info=True)
            stats["ane_embeddings"] = {"enabled": False, "active": False}
        # Model preprocessor registry stats
        if (
            hasattr(self, "_preprocessor_registry")
            and self._preprocessor_registry is not None
        ):
            stats["model_preprocessor"] = self._preprocessor_registry.get_stats()
        stats["reasoning_tokens"] = getattr(self, "_total_reasoning_tokens", 0)
        stats["response_cache"] = {
            "hits": getattr(self, "_response_cache_hits", 0),
            "misses": getattr(self, "_response_cache_misses", 0),
        }
        # Prompt cache stats (exact-match KV state reuse)
        if hasattr(self, "_prompt_cache") and self._prompt_cache is not None:
            stats["prompt_cache"] = self._prompt_cache.get_stats()
        # Warm prompt preloading stats (prefill popular prefixes at startup)
        stats["warm_prompt_prefill"] = getattr(
            self,
            "_warm_prompt_stats",
            {
                "prompts_loaded": 0,
                "prompts_prefilled": 0,
                "prompts_skipped_cached": 0,
                "prompts_failed": 0,
                "total_tokens_prefilled": 0,
                "prefill_time_s": 0.0,
                "source": "none",
            },
        )
        # Inflight prefix sharing stats
        try:
            from .inflight_prefix_sharing import get_inflight_tracker

            stats["inflight_prefix_sharing"] = get_inflight_tracker().get_stats()
        except Exception:
            _engine.logger.debug("inflight prefix stats unavailable", exc_info=True)
            stats["inflight_prefix_sharing"] = {"enabled": False}
        return stats

    def backend_capabilities(self: _engine.BatchedEngine, model: Any = None) -> Any:  # type: ignore[misc]
        """Derive this LM backbone's serving capabilities via the shared,
        backbone-agnostic `model_backend` layer (Part 2/3) — the same classifier
        VLMEngine uses, so both backends make the SAME reuse decision from the
        SAME logic. LM models are never mRoPE. Not memoized (cheap; called rarely
        for introspection). See docs/archive/legacy_vlm_loop/VLM_TEXT_KV_PREFIX.md."""
        from .model_backend import BackendKind, derive_capabilities

        if model is None:
            model = getattr(self, "model", None) or getattr(self, "_model", None)
        layers = []
        try:
            from mlx_lm.models.cache import make_prompt_cache

            layers = make_prompt_cache(model) if model is not None else []
        except Exception:
            _engine.logger.debug(
                "LM cache probe failed; assuming non-reusable", exc_info=True
            )
        return derive_capabilities(BackendKind.LM, layers, is_mrope=False)

    def _cache_supports_trim(self: _engine.BatchedEngine, model: Any) -> bool:  # type: ignore[misc]
        """Whether this model's KV cache can be safely trimmed/snapshotted.

        Prompt and prefix KV reuse both rely on trimming the cached KV. Hybrid
        models (e.g. Qwen3.5) mix full-attention KVCache layers (trimmable) with
        linear/recurrent ArraysCache layers whose state cannot be sliced back —
        reusing them corrupts the recurrent layers. Probe an empty cache once
        and memoize the verdict per model object.
        """
        cached = getattr(self, "_cache_trimmable_flag", None)
        if (
            cached is not None
            and getattr(self, "_cache_trimmable_model", None) is model
        ):
            return cached
        ok = True
        try:
            from mlx_lm.models.cache import make_prompt_cache

            probe = make_prompt_cache(model)
            # Sliding-window (RotatingKVCache) layers report is_trimmable()=True ONLY
            # while empty (offset < max_size); once a prompt exceeds the window the
            # ring buffer rotates and trim becomes UNSOUND (can_trim flips to False).
            # The empty probe here would memoize a stale True, so the prefix/prompt
            # cache would later trim a rotated cache → misordered KV reuse. Treat any
            # RotatingKVCache as non-trimmable up front (same bypass as hybrid).
            _rotating = False
            try:
                from mlx_lm.models.cache import RotatingKVCache

                _rotating = any(isinstance(c, RotatingKVCache) for c in probe)
            except Exception:
                _rotating = False
            if _rotating:
                ok = False
            else:
                try:
                    from mlx_lm.models.cache import can_trim_prompt_cache

                    ok = bool(can_trim_prompt_cache(probe))
                except Exception:
                    # Fall back to per-layer is_trimmable() inspection.
                    def _trimmable(c: Any) -> bool:
                        f = getattr(c, "is_trimmable", None)
                        if callable(f):
                            try:
                                return bool(f())
                            except Exception:
                                return False
                        # Standard attention caches expose keys/values and slice fine.
                        return hasattr(c, "keys") and hasattr(c, "values")

                    ok = all(_trimmable(c) for c in probe)
        except Exception:
            _engine.logger.debug(
                "cache trimmability probe failed; assuming trimmable", exc_info=True
            )
            ok = True
        self._cache_trimmable_flag = ok
        self._cache_trimmable_model = model
        if not ok:
            _engine.logger.info(
                "KV prefix/prompt caching disabled for %s: cache is not trimmable "
                "(hybrid/recurrent model)",
                self.model_name,
            )
        return ok

    def get_kv_cache_stats(self: _engine.BatchedEngine) -> dict:  # type: ignore[misc]
        """Return KV cache statistics (prompt cache + prefix cache + paged KV)."""
        if self._kv_prefix_cache is not None:
            result = {"prefix_cache": self._kv_prefix_cache.get_stats()}
        else:
            result = {"prefix_cache": {"enabled": False}}
        # : surface the PromptCacheManager stats too. Exact-repeat requests
        # hit the prompt cache (full-KV exact match) which SHADOWS the partial-
        # prefix cache (see _run: `if not _pc_hit`), so prefix_cache.hit_rate
        # legitimately stays 0 for repeats while the prompt cache serves them.
        # Reporting only prefix_cache made caching look broken (hit_rate always 0)
        # when it was actually working via the prompt cache.
        if getattr(self, "_prompt_cache", None) is not None:
            try:
                result["prompt_cache"] = self._prompt_cache.get_stats()
            except Exception:
                _engine.logger.debug("prompt cache stats unavailable", exc_info=True)
        if self._engine_core:
            paged = self._engine_core.get_kv_cache_stats()
            result["paged_kv"] = paged
        return result

    def get_radix_tree_stats(self: _engine.BatchedEngine) -> dict:  # type: ignore[misc]
        """Return RadixTree statistics (node count, eviction metrics, block usage)."""
        if not self._engine_core:
            return {"enabled": False}
        # Attr is `scheduler`, not `_scheduler` — the old typo made this always
        # return {"enabled": False}, masking real RadixTree state .
        scheduler = getattr(self._engine_core, "scheduler", None) or getattr(
            self._engine_core, "_scheduler", None
        )
        if scheduler is None:
            return {"enabled": False}
        kv_mgr = getattr(scheduler, "_kv_manager", None) or getattr(
            scheduler, "kv_manager", None
        )
        if kv_mgr is None:
            return {"enabled": False}
        # In tiered mode (YUNSHU_SSD_CACHE_DIR), kv_mgr is a TieredKVCacheManager
        # whose RadixTree lives on `.hot` (: reach through the wrapper, else
        # the stat is silently empty in tiered config).
        tree = getattr(kv_mgr, "_radix_tree", None)
        if tree is None:
            hot = getattr(kv_mgr, "hot", None)
            tree = getattr(hot, "_radix_tree", None) if hot is not None else None
        if tree is None:
            return {"enabled": False}
        return {"enabled": True, **tree.get_stats()}

    @staticmethod
    def _extract_model_arch(model: Any) -> dict:
        """Extract model architecture parameters for KV cache sizing.

        Reads from the model's config attribute (standard HuggingFace pattern).
        Returns kwargs dict for EngineCoreConfig.
        """
        if model is None:
            return {}

        config = getattr(model, "config", None) or getattr(model, "args", None)

        def _first(obj, names, default=0):
            for n in names:
                v = getattr(obj, n, None)
                if isinstance(v, int) and v > 0:
                    return v
            return default

        # Try several naming conventions (HF + mlx-lm ModelArgs variants).
        num_layers = (
            _first(config, ("num_hidden_layers", "n_layers", "num_layers"))
            if config
            else 0
        )
        # Robust fallback: count the actual decoder layers on the model.
        if not num_layers:
            layers = getattr(model, "layers", None)
            if layers is None:
                layers = getattr(getattr(model, "model", None), "layers", None)
            try:
                num_layers = len(layers) if layers is not None else 0
            except TypeError:
                num_layers = 0

        num_kv_heads = (
            _first(config, ("num_key_value_heads", "n_kv_heads", "num_kv_heads"))
            if config
            else 0
        )
        num_attn_heads = (
            _first(config, ("num_attention_heads", "n_heads", "num_heads"))
            if config
            else 0
        )
        if not num_kv_heads:
            num_kv_heads = num_attn_heads  # MHA models: kv heads == attn heads
        head_dim = _first(config, ("head_dim", "kv_head_dim")) if config else 0
        if not head_dim:
            hidden = (
                _first(config, ("hidden_size", "dim", "model_dim")) if config else 0
            )
            if hidden and num_attn_heads:
                head_dim = hidden // num_attn_heads

        if num_layers and num_kv_heads and head_dim:
            return {
                "num_layers": num_layers,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }
        return {}

    def _ensure_memory_guard(self):
        """Lazily build a per-engine MemoryGuard so the preflight check actually
        fires on the DEFAULT FAST PATH.

        Previously `_check_memory_guard` read `self._engine_core._memory_guard`,
        but `_engine_core` is None in default serving, so the preflight was a
        guaranteed no-op for the path that actually serves users — a safety valve
        that never fired. We build a guard from the loaded model's arch on first
        use. It is a SAFETY VALVE, not an admission throttle: it uses a permissive
        KV budget (0.6 of the working set, vs the engine-loop's 0.25) so it only
        rejects a request whose KV alone would blow memory, never throttling
        normal traffic. Fail-open: any construction error → None (no rejection).
        The gateway's token-count `validate_prefill_memory` remains the primary
        per-request guard; this is defense-in-depth on actual memory.
        """
        g = getattr(self, "_fastpath_memory_guard", "unset")
        if g != "unset":
            return g
        # Prefer the engine-loop's already-configured guard if present.
        guard = getattr(self._engine_core, "_memory_guard", None)
        if guard is None:
            try:
                arch = self._extract_model_arch(self._model)
                if not arch:
                    self._fastpath_memory_guard = None
                    return None
                from .memory_guard import MemoryGuard
                from .memory_monitor import MemoryMonitor

                kv_budget = 0
                try:
                    from .utils.hardware import get_hardware_info

                    kv_budget = int(get_hardware_info().max_working_set_bytes * 0.6)
                except Exception:
                    _engine.logger.debug(
                        "fast-path guard: hw info failed", exc_info=True
                    )
                monitor = MemoryMonitor(max_kv_cache_memory=kv_budget)
                _cfg = getattr(self._model, "config", None) or getattr(
                    self._model, "args", None
                )
                monitor.set_model_info(
                    num_layers=arch["num_layers"],
                    num_kv_heads=arch["num_kv_heads"],
                    head_dim=arch["head_dim"],
                    num_attention_heads=getattr(_cfg, "num_attention_heads", None)
                    if _cfg
                    else None,
                )
                monitor.set_baseline_memory()
                guard = MemoryGuard(memory_monitor=monitor)
            except Exception:
                _engine.logger.debug(
                    "fast-path memory guard construction failed", exc_info=True
                )
                guard = None
        self._fastpath_memory_guard = guard
        return guard

    def _check_memory_guard(  # type: ignore[misc]
        self: _engine.BatchedEngine,
        prompt: str | list,
        max_tokens: int,
        raise_on_reject: bool = False,
    ) -> _engine.GenerationOutput | None:
        """Run memory guard preflight check. Returns None if OK.

        Returns a GenerationOutput with finish_reason="memory_limit" if the memory guard
        rejects the request (streams surface its ``error``); ``raise_on_reject`` raises
        :class:`MemoryGuardRejectedError` instead, for the non-streaming path whose caller
        would otherwise return the refusal as an empty, successful completion.
        """
        guard = self._ensure_memory_guard()
        if guard is None:
            return None

        # Estimate prompt tokens. This is a defense-in-depth MEMORY preflight (the gateway
        # already ran validate_prefill_memory with an exact count) — it only needs a rough,
        # conservative estimate, so DO NOT pay a full tokenizer.encode on the event loop
        # here (it inflates TTFT; the real encode happens later in _encode_prompt). For
        # chat messages especially, the old `encode(str(prompt))` tokenized the Python
        # repr — both wasteful and inaccurate. Use a cheap char-based estimate (~3 chars/
        # token, deliberately low divisor → over-estimate, the safe direction for a guard).
        if isinstance(prompt, list):
            _chars = 0
            for _m in prompt:
                if isinstance(_m, dict):
                    _c = _m.get("content")
                    if isinstance(_c, str):
                        _chars += len(_c)
                    elif isinstance(_c, list):
                        for _part in _c:
                            if isinstance(_part, dict) and isinstance(
                                _part.get("text"), str
                            ):
                                _chars += len(_part["text"])
                else:
                    _chars += len(str(_m))
            num_prompt_tokens = _chars // 3 + 16  # +16 for chat-template framing
        elif isinstance(prompt, str):
            num_prompt_tokens = len(prompt) // 3 + 16
        else:
            num_prompt_tokens = len(str(prompt).split()) * 2

        ok, reason = guard.preflight_check(
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=max_tokens,
        )
        if not ok:
            _engine.logger.info(f"Memory guard rejected request: {reason}")
            if raise_on_reject:
                from .exceptions import MemoryGuardRejectedError

                raise MemoryGuardRejectedError(f"Memory guard rejected: {reason}")
            return _engine.GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=num_prompt_tokens,
                completion_tokens=0,
                error=f"Memory guard rejected: {reason}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        return None


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
