# Upstream (inspired): ml-explore/mlx-lm (MIT) mlx_lm/sample_utils.py @ e5dd6100
from __future__ import annotations

"""Yunshu BatchedEngine — user-facing continuous batching engine.

Written from scratch:
- Wraps AsyncEngineCore for clean separation of concerns
- Lazy model loading on first request
- Chat template application before generation
- GeneratorExit-safe cleanup for streaming
- OpenAI-compatible response types

Architecture:
  BatchedEngine (user-facing API)
    → AsyncEngineCore (orchestration)
      → Scheduler (BatchGenerator management)
      → RequestOutputCollector (per-request output buffer)

This is the engine that ModelManager and the gateway routers use.
"""

import asyncio as asyncio
import contextvars as contextvars
import logging as logging
import threading as threading
import time as time
from collections.abc import AsyncIterator as AsyncIterator
from contextlib import contextmanager as contextmanager
from contextlib import suppress as suppress
from dataclasses import dataclass as dataclass
from typing import TYPE_CHECKING
from typing import Any as Any

from . import settings as settings
from .context_window import ContextBudgetError as ContextBudgetError
from .context_window import reject_overlong_prompt as reject_overlong_prompt
from .fast_path_stats import FastPathStats as FastPathStats
from .stream_bridge import StreamBridge as StreamBridge
from .stream_bridge import make_stream_queue as make_stream_queue
from .text_utils import StopHoldbackBuffer as StopHoldbackBuffer

if TYPE_CHECKING:
    from .speculative_decoder import SpeculativeDecoder

logger = logging.getLogger(__name__)


_REASONING_EFFORT_MAP = {"low": 2048, "medium": 8192, "high": 32768}
_MAX_STREAMING_TEXT_BUFFER = (
    1 * 1024 * 1024
)  # 1MB safety limit for streaming text buffer


_CONFIG_EOS_CACHE: dict[str, frozenset] = {}


@contextmanager
def _wired_limit_ctx(model):
    """Raise wired memory limit during generation to prevent weight swapping.

    Follows mlx-lm's generate.py pattern: set limit to recommended max before
    generation, restore + synchronize after.
    """
    mx = None
    old_limit = None
    try:
        import mlx.core as mx

        max_rec = mx.metal.recommended_max_working_memory_size()
        model_bytes = sum(p.nbytes for p in model.parameters())
        old_limit = mx.set_wired_limit(max_rec) if model_bytes > max_rec * 0.5 else None
    except Exception:
        logger.debug("wired limit setup failed", exc_info=True)
        mx = None
        old_limit = None
    try:
        yield
    finally:
        if old_limit is not None and mx is not None:
            try:
                mx.synchronize()
                mx.set_wired_limit(old_limit)
            except Exception:
                logger.debug("wired limit restore failed", exc_info=True)


@dataclass
class GenerationOutput:
    """Output from generation."""

    text: str = ""
    new_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finished: bool = False
    finish_reason: str | None = None
    cached_tokens: int = 0
    logprobs: list[dict] | None = None
    ttft_ms: float = 0.0
    reasoning_tokens: int = 0
    current_state: str | None = None  # "reasoning" or "normal" — matches RequestOutput
    error: str | None = None  # Error message if generation failed
    prefill_progress: tuple[int, int] | None = (
        None  # (processed, total) during chunked prefill
    )
    # True ONLY when a user-supplied stop sequence fired (not natural EOS). finish_reason
    # is "stop" for both, so protocols that must distinguish (Anthropic stop_sequence vs
    # end_turn) read this flag rather than guessing from finish_reason.
    stopped_by_stop_sequence: bool = False
    # Per-prompt-token logprobs (eval/perplexity). vLLM shape — a list
    # aligned to the prompt tokens; element 0 is None (no preceding context).
    prompt_logprobs: list | None = None


from .engine_cache import _PROGRESSIVE_QUANT_INTERVAL as _PROGRESSIVE_QUANT_INTERVAL

# Per-request top-nσ, set by generate()/stream_generate() and read by
# _build_temp_sampler() within the same event-loop task. Default None → off.
_REQUEST_TOP_N_SIGMA: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "yunshu_request_top_n_sigma", default=None
)


# Per-request OpenAI tool schemas for NATIVE chat-template rendering. Set by the router
# (in the request's event-loop task) when it chose native tools over prompt-injection;
# read by _apply_chat_template in the SAME task (prompt build is task-side, not on the
# MLX executor — same propagation guarantee as _REQUEST_TOP_N_SIGMA). Per-task → no
# cross-request leak.
_REQUEST_TOOLS: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "yunshu_request_tools", default=None
)


# How the reply must use the request's native tools: {"tool_choice": ..., "parallel":
# bool}. Set with _REQUEST_TOOLS (same task-side propagation); read by _generate_fast
# to build the tool-call guide.
_REQUEST_TOOL_USE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "yunshu_request_tool_use", default=None
)


from .engine_cache import (
    _create_prompt_cache_with_quant as _create_prompt_cache_with_quant,
)
from .engine_cache import (
    _maybe_quantize_kv_cache as _maybe_quantize_kv_cache,
)
from .engine_cache import (
    _prefix_cache_provenance as _prefix_cache_provenance,
)
from .engine_cache import (
    _progressive_quantize_kv_cache as _progressive_quantize_kv_cache,
)
from .engine_diagnostics import EngineDiagnosticsMixin
from .engine_embeddings import EngineEmbeddingsMixin
from .engine_fast import EngineFastMixin
from .engine_ngram import EngineNgramMixin
from .engine_policy import (
    _is_cancelled as _is_cancelled,
)
from .engine_policy import (
    _parse_quant_config_env as _parse_quant_config_env,
)
from .engine_policy import (
    _prefill_step_size as _prefill_step_size,
)
from .engine_policy import (
    _read_config_eos_ids as _read_config_eos_ids,
)
from .engine_policy import (
    _resolve_model_max_ctx as _resolve_model_max_ctx,
)
from .engine_sampling import (
    _apply_spec_bonus_penalties as _apply_spec_bonus_penalties,
)
from .engine_sampling import (
    _build_constrained_sampler as _build_constrained_sampler,
)
from .engine_sampling import (
    _build_gpu_sampler_text as _build_gpu_sampler_text,
)
from .engine_sampling import (
    _build_grammar_constraint as _build_grammar_constraint,
)
from .engine_sampling import (
    _build_noncached_sampler_text as _build_noncached_sampler_text,
)
from .engine_sampling import (
    _build_temp_sampler as _build_temp_sampler,
)
from .engine_sampling import (
    _wrap_custom_logits_processor as _wrap_custom_logits_processor,
)
from .engine_speculative import EngineSpeculativeMixin
from .engine_stream import EngineStreamMixin
from .engine_templates import EngineTemplatesMixin
from .engine_text import (
    _clean_special_tokens as _clean_special_tokens,
)
from .engine_text import (
    _recover_channel_reasoning as _recover_channel_reasoning,
)
from .engine_text import (
    _resolve_think_token_ids as _resolve_think_token_ids,
)
from .engine_text import (
    _template_supports_tools as _template_supports_tools,
)


class BatchedEngine(
    EngineTemplatesMixin,
    EngineEmbeddingsMixin,
    EngineDiagnosticsMixin,
    EngineFastMixin,
    EngineStreamMixin,
    EngineSpeculativeMixin,
    EngineNgramMixin,
):
    """User-facing continuous batching engine.

    Wraps EngineCore with:
    - Lazy model loading
    - Chat template preprocessing
    - SamplingParams construction
    - GeneratorExit-safe cleanup
    - Special token cleaning
    - Speculative decoding (Phase 4: single-request EAGLE-3 path)
    """

    def __init__(
        self,
        model_name: str = "",
        stream_interval: int = 1,
        enable_thinking: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.stream_interval = stream_interval
        self.enable_thinking = enable_thinking
        # LoRA concurrency keystone: this engine acquires/applies + releases/
        # restores the LoRA adapter ITSELF, inside its executor closures (serialized with
        # generation). The gateway must therefore NOT apply on the event loop (that was the
        # cross-thread race). VLM/legacy engines lack this flag → gateway applies for them.
        self._self_manages_lora = True
        # Friendly identifier used for Prometheus labels, audit, and any
        # outward-facing display. ModelManager passes the full filesystem
        # path as `model_name` so it can be fed to mlx_lm.load_model;
        # using that raw path as a label leaks absolute paths into
        # Prometheus.
        # Derive a short basename for label use without disturbing
        # callers that still rely on `model_name` for the loader.
        import os as _os

        self.model_label = (
            (_os.path.basename(model_name.rstrip("/")) if model_name else "")
            or model_name
            or "default"
        )

        self._model = None
        self._tokenizer = None
        self._engine_core = None
        self._loaded = False
        self._starting = False  # Guard against concurrent start() calls
        self._kv_manager = None  # Set from EngineCore._kv_manager after start

        # Fast-path active request tracking (prevents model eviction mid-generation)
        self._active_fast_path_count = 0
        self._fast_path_lock = threading.Lock()

        # ── Wired production modules ──
        # Model preprocessor registry (auto-detects model family for multimodal input)
        from .model_preprocessor import PreprocessorRegistry

        self._preprocessor_registry = PreprocessorRegistry()

        # Generic external LM drafting (the historical route name is "eagle").
        self._spec_decoder: SpeculativeDecoder | None = None
        self._spec_enabled = False

        # Gemma-4 dual-load assistant drafter (EAGLE-style external drafter that
        # shares the target's KV; validated 2.08×). Gated by YUNSHU_GEMMA4_ASSISTANT.
        self._gemma4_assistant_proposer = None  # Gemma4AssistantProposer

        # N-gram proposer for model-free speculative decoding
        self._ngram_proposer = None  # NgramProposer, created on demand
        # Route greedy requests through the (lossless) n-gram spec path by default,
        # not only when spec_decode=true. Set in _init_spec_decode from env.
        self._ngram_greedy_default = False
        self._ngram_stats = {"proposals": 0, "accepted": 0, "total_draft": 0}

        # Response cache hit/miss counters (YUNSHU_RESPONSE_CACHE=1)
        self._response_cache_hits = 0
        self._response_cache_misses = 0

        # Spec draft verifier: production-grade draft verification with
        # KV cache trimming and bonus token emission
        from .spec_draft_verifier import SpecDraftVerifier

        self._spec_draft_verifier = SpecDraftVerifier(track_stats=True)

        # Adaptive speculative decode controller (built with the n-gram proposer)
        self._adaptive_spec = None

        # Lookahead reasoning: boosts spec decode during <think/> blocks
        from .speculative_decoder import LookaheadReasoning

        self._lookahead_reasoning = LookaheadReasoning()

        # Medusa speculative decoding (multi-head prediction on hidden state)
        self._medusa_proposer = None  # MedusaProposer instance
        self._medusa_strategy = None  # MedusaStrategy wrapper

        # KV prefix cache for multi-turn speedup
        # The 4-bit WARM tier (YUNSHU_PREFIX_HOT_LIMIT > 0) is lossy on reuse,
        # so it is opt-in; by default every cached prefix is full precision.
        from .kv_prefix_cache import KVPrefixCache

        self._kv_prefix_cache = KVPrefixCache(
            max_entries=settings.get("YUNSHU_PREFIX_MAX_ENTRIES"),
            hot_limit=settings.get("YUNSHU_PREFIX_HOT_LIMIT") or None,
            min_prefix_length=32,
        )
        # HYBRID-model prefix reuse on the fast path. Hybrid
        # models (Qwen3.5: KVCache + ArraysCache) normally bypass the prefix
        # cache because trimming the recurrent ArraysCache state corrupts it.
        # With this on, the fast path instead chunk-prefills and stores trim=0
        # block-boundary snapshots, which reuse losslessly (verified); the
        # bypass remains the fallback on any error. Only affects HYBRID models —
        # standard models are trimmable and never enter this path. block=128 is
        # the knee of the reuse/cold-penalty curve (≈3.1x reuse for ≈+18% cold
        # prefill).
        self._hybrid_prefix_block = 128
        # Warm prompt prefill stats (tracked across _warm_prompt_prefill calls)
        self._warm_prompt_stats: dict = {
            "prompts_loaded": 0,
            "prompts_prefilled": 0,
            "prompts_skipped_cached": 0,
            "prompts_failed": 0,
            "total_tokens_prefilled": 0,
            "prefill_time_s": 0.0,
            "source": "none",
        }

        # Prompt cache for exact-match KV state reuse (supplements KVPrefixCache)
        # When the same prompt text is submitted multiple times, the prompt cache
        # returns the full KV state directly — zero prefill compute.
        from .prompt_cache import PromptCacheManager

        self._prompt_cache = PromptCacheManager(
            max_entries=256,
            max_memory_mb=512.0,
            ttl_seconds=3600.0,
        )

        # SSD KV cache persistence (opt-in via YUNSHU_SSD_CACHE=1)
        if settings.get_bool("YUNSHU_SSD_CACHE"):
            self._kv_prefix_cache.enable_ssd_cache(
                cache_dir=settings.get("YUNSHU_SSD_CACHE_DIR"),
                max_size_bytes=int(settings.get("YUNSHU_SSD_CACHE_MAX_GB") * 1024**3),
                model_name=model_name,
                precision=settings.get("YUNSHU_SSD_CACHE_PRECISION"),
                fingerprint=self._ssd_fingerprint(model_name),
            )

        # KV cache quantization config (mlx-lm pattern: to_quantized — group-wise
        # affine along head_dim for BOTH keys and values). Enable via
        # YUNSHU_KV_QUANT_BITS=2, 3, 4, 8 or 'auto' (default 'off': lossy, so opt-in; see
        # _effective_kv_quant_bits). NOTE: this is mlx-lm's group quant,
        # NOT KIVI's per-channel-key / per-token-value scheme — mx.quantize is
        # last-axis only, so KIVI's per-channel key quant would need a forked
        # quantized-attention path (out of scope: this engine wraps MLX). 2-bit
        # is usable (verified quality-preserving) but only pays off at very long
        # context where the KV cache dominates bandwidth; it is a no-op otherwise,
        # which is why it auto-engages only above ~2GB est. KV (_effective_kv_quant_bits).
        _qbits = settings.get("YUNSHU_KV_QUANT_BITS")
        self._kv_quant_auto = _qbits == "auto"
        self._kv_quant_bits: int | None = (
            int(_qbits) if _qbits not in ("auto", "off") else None
        )
        self._kv_quant_group_size: int = 64
        self._kv_quant_start: int = 0

        # Memory pressure eviction config (vllm-mlx pattern)
        # Stored as a percentage (0-100) for consistency with
        # KVPrefixCache.evict_under_pressure(). Converted to a 0-1
        # fraction when calling KVCacheManager.memory_pressure_evict().
        self._mem_pressure_threshold = settings.get("YUNSHU_MEM_PRESSURE_THRESHOLD")
        # If the value looks like a fraction (<=1.0), convert to percentage
        if 0 < self._mem_pressure_threshold <= 1.0:
            self._mem_pressure_threshold *= 100.0

        # Per-model settings (loaded from model_settings.json + env vars)
        self._settings = None
        self._total_reasoning_tokens = 0

        # LoRA adapter manager
        self._lora_manager = None

        # mx.compile() for Metal kernel caching (done once, at warmup)
        self._compiled = False

        # The hand-written Metal kernels were REMOVED. They were never
        # invoked in the served path and, benchmarked head-to-head on this M3 Max,
        # were uniformly SLOWER than Apple's MLX ops (fp16 gemv 1.07-1.19x, q4 gemv
        # 1.41-2.23x, sdpa was just mx.einsum) — mx.fast.*/mx.matmul/
        # mx.quantized_matmul are already hand-tuned for Apple Silicon. Production
        # attention/matmul stays on mx.fast.
        self._metal_kernel_manager = None
        self._metal_kernels_enabled = False

        # Engine loop default (continuous batching mode, experimental)
        self._engine_loop_default = settings.get_bool("YUNSHU_ENGINE_LOOP")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def config(self):
        """Engine config — delegates to EngineCoreConfig when available.

        This property exists so that the admin ``/config/engine`` endpoint
        (which reads ``engine.config``) works with both the legacy ``Engine``
        class (which stores config directly) and ``BatchedEngine`` (which
        keeps the config inside ``EngineCore``).
        """
        if self._engine_core is not None:
            return self._engine_core.config
        return None

    @property
    def is_running(self) -> bool:
        """True when the engine core loop is active (continuous batching).

        Used by ``_ensure_engine_started`` in the gateway engine module
        to decide whether ``start()`` needs to be called.
        """
        return self._loaded and (
            self._engine_core is not None and self._engine_core.is_running
        )

    async def start(self) -> None:
        """Load model and start EngineCore."""
        if self._loaded:
            return

        # Guard against concurrent start() calls from multiple coroutines.
        # Without this, two concurrent generate() calls that both see
        # _loaded=False can race and both load the model simultaneously,
        # doubling memory usage and causing model ref leaks.
        if getattr(self, "_starting", False):
            # Another coroutine is already starting — wait for it
            import asyncio

            while getattr(self, "_starting", False):
                await asyncio.sleep(0.05)
            if self._loaded:
                return
            # If the other starter failed, we need to start ourselves
        self._starting = True

        try:
            await self._do_start()
        finally:
            self._starting = False

    async def _do_start(self) -> None:
        """Internal start implementation (called under _starting guard)."""
        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        # Load model on MLX executor thread (non-blocking)
        def _load():
            from mlx_lm.utils import load as load_model

            kwargs = {}
            # Check for quantization override from env or settings.
            # The old code did kwargs["quantization"] = <str> and passed it to
            # mlx_lm.utils.load — which has NO `quantization` parameter (its params are
            # path/tokenizer_config/model_config/adapter_path/lazy/return_config/revision),
            # so setting YUNSHU_QUANT_CONFIG raised TypeError and FAILED the whole model
            # load (the except below only caught ValueError). The mlx-lm contract is to
            # override the model's config.json quantization via `model_config`. Parse the
            # env as JSON ({"group_size":64,"bits":4}) or a compact "bits" / "bits,group".
            qconfig = settings.get("YUNSHU_QUANT_CONFIG")
            if qconfig:
                _qd = _parse_quant_config_env(qconfig)
                if _qd:
                    kwargs["model_config"] = {
                        **kwargs.get("model_config", {}),
                        "quantization": _qd,
                    }
                else:
                    logger.warning(
                        "YUNSHU_QUANT_CONFIG=%r not parseable as JSON or bits[,group_size]; ignored",
                        qconfig,
                    )
            try:
                return load_model(self.model_name, **kwargs)
            except TypeError as e:
                # An override kwarg the installed mlx-lm doesn't accept must NOT
                # fail the whole engine — retry a clean load without overrides.
                logger.warning(
                    "load() rejected an override kwarg (%s); retrying without overrides",
                    str(e)[:120],
                )
                return load_model(self.model_name)
            except ValueError as e:
                # Some checkpoints ship weights the model class intentionally
                # omits — e.g. Gemma-4's KV-shared layers store vestigial
                # k/v/k_norm the architecture never uses → "Received N
                # parameters not in model". These EXTRA weights are safe to
                # drop. Retry strict=False ONLY for the extra-params case;
                # a "missing parameters" error must still hard-fail.
                if "not in model" not in str(e):
                    raise
                logger.warning(
                    "Checkpoint has extra params not in the model class; "
                    "retrying load with strict=False (%s)",
                    str(e)[:100],
                )
                from pathlib import Path as _Path

                from mlx_lm.utils import (
                    load_model as _load_strict_false,
                )
                from mlx_lm.utils import (
                    load_tokenizer as _load_tok,
                )

                mp = _Path(self.model_name)
                if not mp.exists():
                    from mlx_lm.utils import hf_repo_to_path

                    mp = hf_repo_to_path(self.model_name)
                ret = _load_strict_false(mp, strict=False)
                model = ret[0] if isinstance(ret, tuple) else ret
                return model, _load_tok(mp)

        self._model, self._tokenizer = await loop.run_in_executor(executor, _load)
        # On-the-fly weight quantization at load (YUNSHU_QUANT_MODE =
        # mxfp4 / nvfp4 / mxfp8 / affine). Lets a bf16/fp16 checkpoint be quantized
        # IN-MEMORY at load — smaller + faster decode on bandwidth-bound hardware
        # (4-bit weights ≈ 2.9x over bf16; mxfp4's per-block microscaling can beat
        # affine-4bit on quality-per-bit). Runs on the MLX executor. Already-quantized
        # layers are skipped by the predicate (no to_quantized), so it's safe to set
        # even on a pre-quantized model (no-op there).
        _qmode = settings.get("YUNSHU_QUANT_MODE")
        if _qmode:
            await loop.run_in_executor(executor, self._quantize_on_load, _qmode)
        _cache_tokenizer_vocab(self._tokenizer)

        try:
            await self._finish_start(loop, executor)
            self._loaded = True
        except Exception:
            # Partial init: clean up model that was loaded but subsystems failed
            logger.error(
                "BatchedEngine start failed after model load, cleaning up",
                exc_info=True,
            )
            await self.stop()
            raise

    async def _finish_start(self, loop, executor) -> None:
        """Complete engine initialization after model loading.

        Split from start() so that if this phase fails, the already-loaded
        model can be properly cleaned up via stop().
        """
        # Apply model-specific patches (DeepSeek MLA, Qwen 3.5 YARN, Gemma softcap)
        try:
            from .model_patches import apply_model_patches

            patches = apply_model_patches(self._model, self._tokenizer, self.model_name)
            if patches:
                logger.info(f"Model patches applied: {patches}")
        except Exception:
            logger.warning("Model patches skipped", exc_info=True)

        # Detect model architecture optimizations (RoPE scaling, attention type, MoE)
        try:
            from .model_optimizations import (
                AttentionOptimizer,
                MoEEfficiencyOptimizer,
                RoPEScalingOptimizer,
            )

            rope_opt = RoPEScalingOptimizer()
            target_ctx = getattr(self._model, "max_seq_len", None)
            if target_ctx is None:
                config = getattr(self._model, "config", None) or getattr(
                    self._model, "args", None
                )
                target_ctx = (
                    getattr(config, "max_position_embeddings", 4096) if config else 4096
                )
            rope_opt.configure(self._model, target_context_length=target_ctx)
            attn_opt = AttentionOptimizer()
            attn_opt.detect_attention_type(self._model)
            moe_opt = MoEEfficiencyOptimizer()
            # Auto-detect MoE config from model
            moe_num_experts = 0
            moe_top_k = 0
            if hasattr(self._model, "config"):
                cfg = self._model.config
                moe_num_experts = getattr(
                    cfg, "num_experts", getattr(cfg, "num_local_experts", 0)
                )
                moe_top_k = getattr(
                    cfg, "num_experts_per_tok", getattr(cfg, "num_selected_experts", 0)
                )
            if moe_num_experts > 0 and moe_top_k > 0:
                moe_opt.configure(self._model, moe_num_experts, moe_top_k)
            logger.info(
                f"Model optimizations detected: RoPE={rope_opt.get_scaling_config().scaling_type}, "
                f"Attention={attn_opt.get_stats().get('attention_type', 'unknown')}, "
                f"MoE={moe_opt.get_stats().get('num_experts', 0)} experts"
            )
        except Exception:
            logger.warning("Model optimization detection skipped", exc_info=True)

        # Load per-model settings from model_settings.json + env overrides
        self._load_model_settings()

        # Detect model's cache types for type-aware KV management
        try:
            from mlx_lm.models.cache import make_prompt_cache

            from yunshu_kv.model_cache_config import ModelCacheConfig

            test_cache = make_prompt_cache(self._model)
            self._cache_config = ModelCacheConfig.build_from_cache(test_cache)
            logger.info(
                f"Cache config: {self._cache_config.num_layers} layers, "
                f"{self._cache_config.sliceable_count} sliceable"
            )
        except Exception as e:
            logger.debug(f"Cache type detection skipped: {e}")
            self._cache_config = None

        # Warmup: ModelWarmupManager handles compile caching + KV prefill
        # The basic generate_step warmup is wrapped inside _model_warmup()
        def _warmup():
            import mlx.core as mx
            from mlx_lm.generate import generate_step
            from mlx_lm.sample_utils import make_sampler

            ids = mx.array(self._tokenizer.encode("Hi"))
            sampler = make_sampler(temp=0.0)
            for _ in generate_step(ids, self._model, max_tokens=1, sampler=sampler):
                break
            mx.synchronize()
            mx.clear_cache()

        def _model_warmup():
            """Full warmup using ModelWarmupManager (compile + KV cache prefill)."""
            from .model_optimizations import ModelWarmupManager

            mgr = ModelWarmupManager()
            model_type = "generic"
            name_lower = self.model_name.lower()
            for family in ("qwen", "llama", "deepseek", "gemma"):
                if family in name_lower:
                    model_type = family
                    break
            result = mgr.warmup(
                self._model, model_type=model_type, compile=not self._compiled
            )
            if not result.output_valid:
                raise RuntimeError(
                    f"Model warmup produced invalid outputs for {self.model_name} "
                    f"— model weights or config may be corrupted"
                )
            logger.info(
                f"Model warmup: {result.warmup_time_s:.3f}s, "
                f"compile={result.compile_cached}, prompts={result.prompts_warmed}"
            )
            # Also do basic warmup to ensure MX compile cache is populated
            _warmup()

        await loop.run_in_executor(executor, _model_warmup)

        # The previous block called `mx.compile(self._model)`
        # and DISCARDED the return value. mx.compile returns a NEW compiled
        # callable; it does not mutate the model in place — so this compiled
        # nothing (the model was still invoked via its original __call__) while
        # logging a misleading "Model compiled — Metal kernels cached". There is
        # no CUDA-graph equivalent on Apple Silicon, and mlx-lm already applies
        # @mx.compile internally to the hot sampling/RoPE paths; whole-model
        # compile also recompiles on every prefill shape change (ragged batching),
        # which can hurt. Removed the dead attempt — no engine-level whole-model
        # compile is performed. (ModelWarmupManager's own compile path above is
        # unchanged.)

        # Initialize speculative decoding if model supports it (Phase 4)
        await loop.run_in_executor(executor, self._init_spec_decode)

        # Initialize LoRA adapter manager
        self._init_lora()

        # Warm prompt prefill: pre-populate KV cache with common system prompts
        await self._warm_prompt_prefill()

        # Auto-start EngineCore so the scheduler is ready for concurrent requests.
        try:
            await self._ensure_engine_core()
            logger.info("EngineCore auto-started — continuous batching ready")
        except Exception as e:
            logger.warning(
                f"EngineCore auto-start failed ({e}), falling back to fast-path only"
            )
            self._engine_core = None

        # KV-5: Apply auto-tuner KV quantization recommendation if available
        self._apply_auto_tuner_kv_quant()

        # Metal kernel init removed (kernels deleted — slower than mx.fast).

    def _load_model_settings(self):
        """Load per-model settings from model directory and apply to engine."""
        from .model_settings import load_model_settings

        model_path = ""
        if self._model is not None:
            config = getattr(self._model, "config", None)
            if config is not None:
                if isinstance(config, dict):
                    model_path = config.get("_name_or_path", self.model_name)
                else:
                    model_path = getattr(config, "_name_or_path", self.model_name)
        self._settings = load_model_settings(
            model_path or self.model_name, self.model_name
        )
        self._apply_settings()

    def _apply_settings(self):
        """Apply loaded ModelSettings to engine config."""
        if self._settings is None:
            return
        s = self._settings
        if s.kv_cache_quant_bits is not None:
            # Validate the bits BEFORE assigning. model_settings.json (and the
            # YUNSHU_MODEL_{ID}_KV_CACHE_QUANT_BITS override) flowed through here
            # WITHOUT validation, so an invalid value (e.g. 5) reached
            # to_quantized(bits=5) and crashed every request that hit the quant
            # threshold. MLX only supports 2/3/4/8.
            if s.kv_cache_quant_bits in (2, 3, 4, 8):
                self._kv_quant_bits = s.kv_cache_quant_bits
            else:
                logger.warning(
                    "CONFIG: model_settings kv_cache_quant_bits=%s is not a "
                    "supported value (MLX supports 2, 3, 4, 8). Ignoring.",
                    s.kv_cache_quant_bits,
                )
        if s.kv_cache_quant_group_size != 64:
            self._kv_quant_group_size = s.kv_cache_quant_group_size
        # NOTE: kv_cache_quant_start_layer is a LAYER INDEX — turbo_quant
        # uses it for per-layer fp16/int8/int4 tiering (turbo_quant.py:178-188). It
        # must NOT be assigned to self._kv_quant_start, which is mlx-lm's
        # quantized_kv_start: a TOKEN-POSITION threshold compared against cache.offset.
        # Conflating them made "quantize from layer N" silently mean "quantize once the
        # sequence reaches N tokens". _kv_quant_start stays 0; the layer field is
        # consumed by turbo_quant only.
        if not s.prefix_cache_enabled:
            self._kv_prefix_cache = None

        if s.ssd_cache_enabled and self._kv_prefix_cache is not None:
            self._kv_prefix_cache.enable_ssd_cache(
                cache_dir=s.ssd_cache_dir,
                max_size_bytes=s.ssd_cache_max_gb * 1024**3,
                model_name=self.model_name,
                precision=settings.get("YUNSHU_SSD_CACHE_PRECISION"),
                fingerprint=self._ssd_fingerprint(self.model_name),
            )
        if s.enable_thinking is not None:
            self.enable_thinking = s.enable_thinking
        if s.moe_top_k > 0 and self._model is not None:
            from .moe_optimization import apply_moe_top_k

            result = apply_moe_top_k(self._model, s.moe_top_k)
            if result["patched_layers"] > 0:
                logger.info(f"MoE top-k applied: {result}")

    def get_settings(self):
        return self._settings

    def _quantize_on_load(self, mode: str) -> None:
        """Quantize the loaded model's weights in-memory (mxfp4/nvfp4/
        mxfp8/affine) via mlx-lm's native quantization. Runs on the MLX executor.
        Skips layers without to_quantized or whose width isn't group-aligned (the
        same predicate mlx-lm's quantize_model uses) — so already-quantized models
        are a safe no-op."""
        try:
            import mlx.nn as nn
        except Exception:
            return
        defaults = {
            "affine": (64, 4),
            "mxfp4": (32, 4),
            "nvfp4": (16, 4),
            "mxfp8": (32, 8),
        }
        group_size, bits = defaults.get(mode, (64, 4))

        def _pred(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            w = getattr(module, "weight", None)
            return w is not None and w.shape[-1] % group_size == 0

        try:
            nn.quantize(self._model, group_size, bits, mode=mode, class_predicate=_pred)
            logger.info(
                "On-the-fly quantization applied: mode=%s group_size=%d bits=%d (%s)",
                mode,
                group_size,
                bits,
                self.model_name,
            )
        except Exception as e:
            logger.error(
                "On-the-fly quantization (mode=%s) failed (%s); serving the model "
                "as loaded.",
                mode,
                e,
                exc_info=True,
            )

    def _kv_bytes_per_token(self) -> int:
        """Estimate the model's bf16 KV-cache bytes per token (K+V across all
        layers): 2 (K,V) * num_layers * num_kv_heads * head_dim * 2 bytes.
        Returns 0 if the dims can't be read."""
        model = getattr(self, "_model", None)
        if model is None:
            return 0
        cfg = getattr(model, "config", None) or getattr(model, "args", None)
        if cfg is None:
            return 0
        # Some models nest text config (VLM/omni).
        text_cfg = getattr(cfg, "text_config", None)

        def _g(names):
            for obj in (text_cfg, cfg):
                if obj is None:
                    continue
                for n in names:
                    v = getattr(obj, n, None)
                    if v:
                        return int(v)
            return 0

        n_layers = _g(("num_hidden_layers", "n_layers", "num_layers"))
        n_kv = _g(("num_key_value_heads", "n_kv_heads", "num_kv_heads")) or _g(
            ("num_attention_heads", "n_heads", "num_heads")
        )
        hidden = _g(("hidden_size", "d_model", "n_embd"))
        n_heads = _g(("num_attention_heads", "n_heads", "num_heads")) or 1
        head_dim = _g(("head_dim",)) or (hidden // n_heads if hidden else 0)
        if not (n_layers and n_kv and head_dim):
            return 0
        return 2 * n_layers * n_kv * head_dim * 2  # K+V, bf16

    @staticmethod
    def _ssd_fingerprint(model_name: str) -> str:
        """Identity of the checkpoint and the persisted layout for the SSD KV cache."""
        from yunshu_kv.fingerprint import checkpoint_fingerprint

        return checkpoint_fingerprint(
            model_name,
            extra={
                "ssd_precision": settings.get("YUNSHU_SSD_CACHE_PRECISION"),
                "kv_quant_bits": str(settings.get("YUNSHU_KV_QUANT_BITS")),
            },
        )

    def _effective_kv_quant_bits(self, total_tokens: int) -> int | None:
        """KV-quant bits for THIS request.

        Explicit config (YUNSHU_KV_QUANT_BITS=N / per-model settings) always
        wins. Otherwise apply a SIZE-gated default: 8-bit (near-lossless) once the
        estimated KV cache is large enough that it dominates decode bandwidth.

        An earlier version gated on TOKEN COUNT (>=8192), but a real-model benchmark
        (scripts/bench/bench_kv_quant_longctx.py) showed that's WRONG for small
        models: Qwen2.5-0.5B at 9001 tokens has only ~108MB of KV (~12KB/token)
        vs ~300MB of 4-bit weights, so KV doesn't dominate and quantization is
        NET-NEGATIVE (0.98x — the dequant overhead exceeds the saved KV bandwidth).
        8-bit KV only wins when KV bytes are a large fraction of the per-step read,
        which depends on layers*kv_heads*head_dim*tokens, NOT tokens alone. So gate
        on estimated total KV bytes (default >=2GB, big-model/very-long-context
        regime). YUNSHU_KV_QUANT_BITS=off disables it. Falls back to a token
        threshold (16384) only when model dims can't be read."""
        if self._kv_quant_bits is not None:
            return self._kv_quant_bits
        if not getattr(self, "_kv_quant_auto", True):
            return None
        per_tok = self._kv_bytes_per_token()
        if per_tok > 0:
            return 8 if (per_tok * total_tokens) >= 2 * 1024**3 else None
        # Dims unreadable — fall back to the conservative token threshold.
        return 8 if total_tokens >= 16384 else None

    def _apply_auto_tuner_kv_quant(self) -> None:
        """KV-5: Apply auto-tuner's kv_quantization_bits recommendation.

        Checks the AutoTuner's current params for KV quantization recommendation
        and updates the engine's _kv_quant_bits if the auto-tuner suggests
        quantization that isn't already configured. This enables adaptive KV
        quantization based on observed memory pressure.

        Priority: env var > model settings > auto_tuner recommendation.
        """
        # Lossy: only when the user opted into automatic KV quantization.
        if not getattr(self, "_kv_quant_auto", False):
            return
        # Only apply if KV quantization isn't already explicitly configured
        if self._kv_quant_bits is not None:
            return  # Already configured via env var or model settings

        # Check EngineCore's auto-tuner if available
        if self._engine_core is not None:
            auto_tuner = getattr(self._engine_core, "_auto_tuner", None)
            if auto_tuner is not None:
                recommended_bits = auto_tuner.params.kv_quantization_bits
                if recommended_bits < 16:  # 16 means no quantization
                    self._kv_quant_bits = recommended_bits
                    logger.info(
                        f"KV-5: Auto-tuner recommends {recommended_bits}-bit KV quantization"
                    )

    def _init_lora(self):
        """Initialize LoRA adapter manager after model load."""
        max_loras = settings.get("YUNSHU_MAX_LORAS")
        from .lora_manager import LoRAAdapterManager, set_lora_manager

        self._lora_manager = LoRAAdapterManager(max_loras=max_loras)
        self._lora_manager.set_base_model(self._model)
        # Wire LoRA memory tracking into model manager budget. The singleton
        # accessor lives in the gateway layer (not yunshu_engine.model_manager);
        # this is a best-effort runtime hook, so the lazy import here is fine and
        # fails gracefully when the engine runs standalone (no gateway).
        try:
            from yunshu_gateway.engine import get_model_manager

            _mm = get_model_manager()
            if _mm is not None:
                self._lora_manager.set_memory_callback(_mm.track_lora_memory)
        except Exception as e:
            # WARNING (not DEBUG) so silent budget-tracking failures are
            # surfaced — LoRA memory accounting won't update against the
            # model-manager budget if this wiring is skipped.
            logger.warning(
                "LoRA memory callback wiring failed (model-manager budget "
                "will not track LoRA memory): %s",
                e,
                exc_info=True,
            )
        set_lora_manager(self._lora_manager, engine_id=self.model_name or "default")

        # Auto-discover adapters in model directory
        model_path = ""
        config = getattr(self._model, "config", None)
        if config is not None:
            if isinstance(config, dict):
                model_path = config.get("_name_or_path", "")
            else:
                model_path = getattr(config, "_name_or_path", "")
        if model_path:
            discovered = self._lora_manager.discover_adapters(model_path)
            if discovered:
                logger.info(f"Discovered {len(discovered)} LoRA adapters: {discovered}")

    def get_lora_manager(self):
        return self._lora_manager

    async def _ensure_engine_core(self):
        """Lazy-create EngineCore only when continuous batching is needed."""
        if self._engine_core is not None:
            return
        from .engine_core import EngineCore, EngineCoreConfig
        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        arch_kwargs = self._extract_model_arch(self._model)

        self._engine_core = EngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=EngineCoreConfig(
                stream_interval=self.stream_interval,
                # Decode-batch width is configurable, but the
                # default of 32 is near-optimal on Apple-Silicon UMA and should
                # rarely be raised. Decode is memory-bandwidth-bound, so
                # aggregate throughput SATURATES around batch ~32; measured on
                # Qwen3.5-2B, raising it to 64 made N=96/128 WORSE (130 then a
                # crash) because a wider decode batch only adds KV memory
                # pressure without more throughput. The N>=32 throughput plateau
                # is the hardware ceiling, not a tunable software limit.
                completion_batch_size=settings.get("YUNSHU_COMPLETION_BATCH_SIZE"),
                **arch_kwargs,
            ),
            executor=executor,
        )
        self._engine_core.scheduler.config.model_name = self.model_name

        # Propagate paged KV manager for memory pressure eviction in fast paths
        self._kv_manager = self._engine_core._kv_manager

        # Setup memory guard with model dimensions
        try:
            model_cfg = (
                getattr(self._model, "config", self._model) if self._model else None
            )
            if model_cfg is not None:
                num_layers = getattr(model_cfg, "num_hidden_layers", 0)
                num_kv_heads = getattr(model_cfg, "num_key_value_heads", 0)
                head_dim = getattr(model_cfg, "hidden_size", 0) // max(
                    getattr(model_cfg, "num_attention_heads", 1), 1
                )
                num_attn_heads = getattr(model_cfg, "num_attention_heads", None)
                if num_layers and num_kv_heads and head_dim:
                    self._engine_core.setup_memory_guard(
                        num_layers=num_layers,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        num_attention_heads=num_attn_heads,
                    )
        except Exception:
            logger.warning("MemoryGuard setup skipped", exc_info=True)

        # Setup TurboQuant per-layer mixed-precision KV quantization
        try:
            model_cfg = (
                getattr(self._model, "config", self._model) if self._model else None
            )
            if model_cfg is not None:
                num_layers = getattr(model_cfg, "num_hidden_layers", 0)
                if num_layers > 0:
                    quant_bits = getattr(model_cfg, "kv_cache_quant_bits", None)
                    quant_start = int(
                        getattr(model_cfg, "kv_cache_quant_start_layer", 0)
                    )
                    quant_group = int(
                        getattr(model_cfg, "kv_cache_quant_group_size", 64)
                    )
                    # Enable when the model config carries quant settings.
                    if quant_bits is not None:
                        self._engine_core.setup_turbo_quant(
                            total_layers=num_layers,
                            kv_quant_bits=quant_bits,
                            kv_quant_start_layer=quant_start,
                            kv_quant_group_size=quant_group,
                        )
        except Exception:
            logger.debug("TurboQuant setup skipped", exc_info=True)

        # Setup HybridKVCache layer type registration for Mamba/hybrid models
        try:
            model_cfg = (
                getattr(self._model, "config", self._model) if self._model else None
            )
            if model_cfg is not None and self._model is not None:
                num_layers = getattr(model_cfg, "num_hidden_layers", 0)
                if num_layers > 0:
                    self._engine_core.setup_hybrid_kv(
                        model=self._model,
                        total_layers=num_layers,
                    )
        except Exception:
            logger.debug("HybridKVCache setup skipped", exc_info=True)

        # Wire prefix cache into scheduler for batch-path insert_segments (C16)
        self._engine_core.set_prefix_cache(self._kv_prefix_cache)
        # Wire HybridKVCache into scheduler for layer-type-aware KV routing
        if self._engine_core._hybrid_kv is not None:
            self._engine_core.scheduler.set_hybrid_kv_cache(
                self._engine_core._hybrid_kv
            )
        # Wire speculative decoding decoders into the scheduler
        self._wire_spec_decoders_to_scheduler()
        await self._engine_core.start()

        logger.info(f"BatchedEngine started: {self.model_name}")

    def _wire_spec_decoders_to_scheduler(self) -> None:
        """Wire speculative decoding decoders into the scheduler's batch path.

        Propagate the text n-gram proposer when the scheduler lacks one.
        Native Qwen MTP and DFlash are served by the VLM runner; unverified
        cross-model/text MTP decoders are not installed in the text scheduler.
        """
        if self._engine_core is None:
            return
        scheduler = self._engine_core.scheduler

        # Propagate N-gram proposer to scheduler if needed
        # (BatchedEngine creates its own N-gram proposer, but the scheduler
        # may not have one if ngram_spec_enabled was False in EngineCoreConfig).
        # Only the n-gram family has min_n/max_n config; the suffix proposer is
        # driven solely on the single-request fast path, so skip the batch-loop
        # sync for it (the legacy BatchGenerator loop has no suffix support).
        if (
            self._ngram_proposer is not None
            and scheduler._ngram_proposer is None
            and hasattr(self._ngram_proposer.config, "min_n")
        ):
            scheduler.enable_ngram_spec(
                min_n=self._ngram_proposer.config.min_n,
                max_n=self._ngram_proposer.config.max_n,
                k=self._ngram_proposer.config.k,
                mode=self._ngram_proposer.config.mode,
            )
            logger.info("Wired N-gram proposer into scheduler batch path")

    async def stop(self) -> None:
        """Stop engine and release resources.

        Idempotent: safe to call multiple times or before start().
        Order of shutdown:
        1. EngineCore (stops engine loop, drains in-flight requests)
        2. Subsystem cleanup (KV caches, spec decoder, LoRA, Metal kernels)
        3. Model/tokenizer release + GC + MLX cache clear
        """
        if not self._loaded and self._engine_core is None:
            return  # Never started or already stopped

        # 1. Stop EngineCore (continuous batching loop, drain in-flight)
        if self._engine_core is not None:
            await self._engine_core.stop()
            self._engine_core = None

        # 2. Release KV prefix cache (holds MLX array refs) and flush SSD tier
        if self._kv_prefix_cache is not None:
            self._kv_prefix_cache.close()
            self._kv_prefix_cache.clear()
            self._kv_prefix_cache = None

        # Prompt cache: clear on shutdown (in-memory only; for cross-restart
        # persistence use SSD cache via YUNSHU_SSD_CACHE=1)
        if self._prompt_cache is not None:
            self._prompt_cache.invalidate_all()
            self._prompt_cache = None

        # Speculative decoding subsystems
        self._spec_decoder = None
        self._spec_enabled = False
        self._gemma4_assistant_proposer = None
        self._ngram_proposer = None
        self._adaptive_spec = None
        self._medusa_proposer = None
        self._medusa_strategy = None
        self._warm_prompts = None
        self._metal_kernel_manager = None

        # Stop LoRA adapter manager (unload all adapters, release weights)
        if self._lora_manager is not None:
            try:
                self._lora_manager.shutdown()
            except Exception:
                logger.debug("LoRA manager cleanup failed", exc_info=True)
            self._lora_manager = None
            from .lora_manager import set_lora_manager

            set_lora_manager(None, engine_id=self.model_name or "default")

        # 3. Release model + tokenizer refs, then GC + clear MLX cache
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._compiled = False

        import gc

        gc.collect()

        # Per-engine cleanup: synchronize GPU work and clear the MLX cache.
        # Do NOT call shutdown_mlx_executor() — it destroys the GLOBAL
        # singleton ThreadPoolExecutor shared by all engines (BatchedEngine,
        # VLM, Video, OCR, etc.). The global executor should only be shut
        # down at process teardown, not per-engine.
        from .mlx_executor import get_mlx_executor

        loop = asyncio.get_running_loop()
        import mlx.core as mx

        def _cleanup():
            mx.synchronize()
            mx.clear_cache()

        try:
            await loop.run_in_executor(get_mlx_executor(), _cleanup)
        except RuntimeError:
            logger.debug("MLX executor cleanup skipped (executor already shut down)")
        logger.info(f"BatchedEngine stopped: {self.model_name}")

    # ── Speculative decoding routes (the single gating table) ────────────────
    #
    # | route            | taken when                                          | evidence                                   |
    # |------------------|-----------------------------------------------------|--------------------------------------------|
    # | gemma4_assistant | spec_decode, non-stream, fast path, eligible request | dual-load drafter, 2.08x, greedy-exact     |
    # |                  | (YUNSHU_GEMMA4_ASSISTANT loaded)                     | (_gemma4_spec_eligible)                    |
    # | ngram            | non-stream, fast path, greedy, no logprobs, proposer | tests/unit/test_ngram_spec_lossless.py:    |
    # |                  | on, and (spec_decode or YUNSHU_NGRAM_DEFAULT=1);     | output == plain greedy with drafts accepted|
    # |                  | non-trimmable caches fall back inside the method     | opt-in: ~2.5x slower on low-acceptance     |
    # |                  |                                                      | prose/code, ~1.7x faster on repetitive     |
    # | external LM      | SPEC_UNVERIFIED=eagle + draft path; greedy,         | Unverified; ordinary LM draft, not EAGLE. |
    # |                  | non-streaming fast path, no logprobs                 | Existing Qwen2 checkpoints make it reachable. |
    # | (streaming ngram)| never: _stream_generate_ngram_spec early-terminates  | live repro: "count to 8" streamed "1 "     |
    #
    # Everything else is plain decoding.
    def _spec_route(
        self,
        *,
        spec_decode: bool,
        stream: bool,
        temperature: float | None,
        logprobs: bool,
        use_engine_loop: bool,
        gemma4_eligible=None,
        external_eligible: bool = True,
    ) -> str | None:
        if use_engine_loop or logprobs:
            return None
        greedy = temperature is None or temperature <= 0.0
        if (
            spec_decode
            and greedy
            and not stream
            and external_eligible
            and getattr(self, "_spec_enabled", False)
            and getattr(self, "_spec_decoder", None) is not None
            and settings.get("YUNSHU_SPEC_UNVERIFIED") == "eagle"
        ):
            return "eagle"
        if stream:
            return None
        if spec_decode and gemma4_eligible is not None and gemma4_eligible():
            return "gemma4_assistant"
        if (
            greedy
            and getattr(self, "_ngram_proposer", None) is not None
            and (spec_decode or getattr(self, "_ngram_greedy_default", False))
        ):
            return "ngram"
        return None

    def _should_use_engine_loop(self, use_engine_loop: bool | None) -> bool:
        """Route through EngineCore continuous batching or the fast path.

        The choice is configuration, never load: ``YUNSHU_ENGINE_LOOP=1`` (or an
        explicit per-call ``use_engine_loop``) selects the loop; otherwise every
        request takes the single-request fast path. (It used to switch to the
        loop whenever EngineCore had active requests, so numerics and available
        features depended on timing.)
        """
        # Safety, not a heuristic: the BatchGenerator shares the model and
        # generation_stream with the fast path on the single MLX executor, and
        # overlapping insert()/next() with a fast-path generate_step corrupts
        # BatchGenerator state (IndexError in mlx_lm.generate._next). While a
        # fast-path request runs, nothing may enter the loop.
        if getattr(self, "_active_fast_path_count", 0) > 0:
            return False
        if use_engine_loop is not None:
            return use_engine_loop
        return bool(getattr(self, "_engine_loop_default", False))

    # ── Non-generative tasks: embeddings, pooling ────────────────────────────

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        json_schema: dict | str | None = None,
        spec_decode: bool = False,
        use_engine_loop: bool | None = None,
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        thinking_budget: int | None = None,
        reasoning_effort: str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        top_n_sigma: float = 0.0,
        cancel_event: asyncio.Event | None = None,
        priority: int = 0,
        logits_processors: list | None = None,
        timeout_seconds: float | None = None,
        images: list | None = None,
        lora_adapter: str | None = None,
        kv_cache_breakpoints: list[int] | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        suppress_tokens: list[int] | None = None,
        prompt_logprobs: int | None = None,
    ) -> GenerationOutput:
        """Non-streaming text generation.

        Args:
            spec_decode: If True, use speculative decoding path when available.
                         Falls back to standard generation if spec decode is
                         not configured or the model lacks spec heads.
            use_engine_loop: If True, route through EngineCore's continuous
                             batching loop. If False, use fast path.
                             If None (default), uses YUNSHU_ENGINE_LOOP env var
                             (defaults to False if not set).
            logprobs: If True, return log probabilities for each generated token.
            top_logprobs: Number of top logprobs to return per token (max 20).
            timeout_seconds: Per-request generation timeout. If None, uses the
                             engine's default (300s). Overrides the engine default
                             for this request only.
        """
        if not self._loaded:
            await self.start()

        # Early return: max_tokens <= 0 produces no output
        if max_tokens <= 0:
            _pt = 0
            if self._tokenizer is not None:
                try:
                    if (
                        isinstance(prompt, list)
                        and prompt
                        and isinstance(prompt[0], dict)
                    ):
                        _text = self._apply_chat_template(prompt, enable_thinking)
                    else:
                        _text = prompt if isinstance(prompt, str) else str(prompt)
                    _pt = len(self._tokenizer.encode(_text))
                except Exception:
                    pass
            return GenerationOutput(
                text="",
                new_text="",
                prompt_tokens=_pt,
                completion_tokens=0,
                finished=True,
                finish_reason="length",
            )

        # publish per-request top-nσ for _build_temp_sampler (same event-loop task)
        _REQUEST_TOP_N_SIGMA.set(
            top_n_sigma if top_n_sigma and top_n_sigma > 0 else None
        )
        _use_engine_loop = self._should_use_engine_loop(use_engine_loop)

        _spec = self._spec_route(
            spec_decode=spec_decode,
            stream=False,
            temperature=temperature,
            logprobs=bool(logprobs),
            use_engine_loop=_use_engine_loop,
            # The external adapter does not implement the full fast-path
            # parameter contract. Never drop a client's requested behavior.
            external_eligible=(
                json_schema is None
                and not logits_processors
                and not logit_bias
                and not stop
                and thinking_budget is None
                and reasoning_effort is None
                and not lora_adapter
                and repetition_penalty == 1.0
                and frequency_penalty == 0.0
                and presence_penalty == 0.0
                and top_p == 1.0
                and top_k == 0
                and min_p == 0.0
                and xtc_probability == 0.0
                and not top_n_sigma
                and not min_tokens
                and not ignore_eos
                and not suppress_tokens
            ),
            gemma4_eligible=lambda: self._gemma4_spec_eligible(
                logprobs=logprobs,
                json_schema=json_schema,
                logits_processors=logits_processors,
                logit_bias=logit_bias,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                xtc_probability=xtc_probability,
                lora_adapter=lora_adapter,
            ),
        )
        spec_decode = _spec is not None

        # Memory guard preflight check
        # (raises: a refused request is an error, not an empty "length" completion)
        self._check_memory_guard(prompt, max_tokens, raise_on_reject=True)

        # Resolve reasoning_effort → thinking_budget if not explicitly set
        if thinking_budget is None and reasoning_effort is not None:
            thinking_budget = _REASONING_EFFORT_MAP.get(reasoning_effort, 8192)
            if enable_thinking is None:
                enable_thinking = True

        # Gemma-4 default: its chat template enables thinking by default but
        # emits inline `thought` tokens that don't auto-stop, producing output
        # like "4thought\nThinking Process: …" for simple prompts. Default
        # enable_thinking=False for gemma-4 unless caller passed reasoning_effort.
        if enable_thinking is None and reasoning_effort is None:
            _mn = getattr(self, "model_name", None)
            if isinstance(_mn, str) and "gemma-4" in _mn.lower():
                enable_thinking = False

        # Response cache lookup (YUNSHU_RESPONSE_CACHE=1)
        _rc_hash = None
        # Only cache DETERMINISTIC requests. The HTTP middleware skips
        # temperature>0 && seed is None (it would freeze one random sample and replay it for
        # every identical request); the engine cache had NO such guard, so a sampled/creative
        # request got the same frozen completion every time. Mirror the middleware.
        _rc_deterministic = not (
            temperature is not None and temperature > 0 and seed is None
        )
        if not spec_decode and _rc_deterministic:
            try:
                from .gateway_optimizer import get_response_cache

                _rc = get_response_cache()
                if _rc.enabled:
                    _rc_hash = _rc.hash_request(
                        self.model_name,
                        prompt if isinstance(prompt, str) else str(prompt),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop=str(stop),
                        seed=seed,
                        enable_thinking=enable_thinking,
                        thinking_budget=thinking_budget,
                        json_schema=str(json_schema),
                        # Output-affecting params must be in the key or a
                        # logprobs / biased / xtc request can collide with a
                        # plain one and serve a wrong cached response.
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                        logit_bias=str(logit_bias),
                        xtc_probability=xtc_probability,
                        xtc_threshold=xtc_threshold,
                        # These ALSO change the output and were missing from the key —
                        # lora_adapter is the worst (LoRA is the product feature: a request for
                        # adapter Y collided with a cached adapter X response → wrong fine-tuned
                        # weights served as 200 OK).
                        lora_adapter=str(lora_adapter),
                        stop_token_ids=str(stop_token_ids),
                        min_tokens=min_tokens,
                        ignore_eos=ignore_eos,
                        suppress_tokens=str(suppress_tokens),
                    )
                    _rc_hit = await _rc.get(_rc_hash)
                    if _rc_hit is not None:
                        self._response_cache_hits += 1
                        return _rc_hit
                    self._response_cache_misses += 1
            except Exception:
                logger.debug("response cache lookup failed", exc_info=True)

        # ── Context window truncation for long prompts ──
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            try:
                if self._tokenizer and hasattr(self._tokenizer, "encode"):
                    text = self._apply_chat_template(prompt, enable_thinking)
                    token_count = len(self._tokenizer.encode(text))
                    max_ctx = getattr(self._model, "max_seq_len", None)
                    if max_ctx is None:
                        max_ctx = getattr(
                            getattr(self._model, "config", None), "max_seq_len", None
                        ) or getattr(
                            getattr(self._model, "args", None), "max_seq_len", None
                        )
                    # Thinking tokens also consume context window positions —
                    # subtract them from the available prompt budget.
                    _thinking_overhead = (
                        thinking_budget if (thinking_budget and enable_thinking) else 0
                    )
                    _generation_budget = max_tokens + _thinking_overhead
                    if max_ctx and token_count + _generation_budget > max_ctx:
                        from .context_window import ContextWindowManager

                        ctx_mgr = ContextWindowManager(
                            token_counter=lambda text: len(
                                self._tokenizer.encode(text)
                            ),
                        )
                        result = ctx_mgr.compute_truncation(
                            messages=prompt,
                            max_tokens=max_ctx - _generation_budget,
                            strategy="importance_aware",
                        )
                        prompt = result.messages
                        result.publish()
                        result.raise_if_cannot_fit()
                        logger.debug(
                            f"Context window truncated: {token_count} → "
                            f"{result.truncated_token_count} tokens (saved {result.tokens_saved})"
                        )
            except ContextBudgetError:
                raise
            except Exception:
                logger.warning("context window truncation skipped", exc_info=True)

        # Gemma-4 assistant-drafter spec decode (#175): n=1 serving via the
        # validated dual-load primitive (~2.5x, lossless). Eligibility-gated so
        # the output is identical to normal generation; only active when the
        # drafter was loaded (YUNSHU_GEMMA4_ASSISTANT).
        if _spec == "gemma4_assistant":
            try:
                return await self._generate_gemma4_assistant_spec(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    stop_token_ids=stop_token_ids,
                )
            except Exception:
                logger.warning(
                    "Gemma-4 assistant spec decode failed; falling back", exc_info=True
                )

        # Explicitly opted-in, unverified external LM draft route.
        if _spec == "eagle":
            return await self._generate_speculative(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                enable_thinking=enable_thinking,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                json_schema=json_schema,
                cancel_event=cancel_event,
                logits_processors=logits_processors,
                timeout_seconds=timeout_seconds or 300.0,
                lora_adapter=lora_adapter,
            )

        # N-gram speculative decoding (model-free, CPU-based proposal).
        # N-gram verify accepts a draft iff it equals the
        # target's ARGMAX but samples the bonus at the request temperature, so for
        # temperature>0 accepted tokens are forced to the greedy sequence (biased,
        # not lossless). Restrict to GREEDY requests; temp>0 → normal decode.
        #
        # Lossless at temp<=0 (verifier accepts only the model's own argmax), so output
        # is byte-identical to the plain fast path. But NOT free: OPT-IN, not default —
        # measured 2026-06-30, batch-verify costs K× compute per step, so on low-acceptance
        # output (normal prose / code) it's ~2.5× SLOWER on Qwen2.5-3B-4bit; it only wins on
        # repetitive/copy-heavy/agentic output. Reach it via per-request spec_decode=true or
        # YUNSHU_NGRAM_DEFAULT=1 (the adaptive controller does NOT yet back off enough to make
        # default-on safe). Engages only for greedy (temp<=0).
        if _spec == "ngram":
            return await self._generate_ngram_spec(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                json_schema=json_schema,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                cancel_event=cancel_event,
                logits_processors=logits_processors,
                timeout_seconds=timeout_seconds or 300.0,
                lora_adapter=lora_adapter,
            )

        # Fast path: direct generate_step on executor thread for full GPU utilization
        if not _use_engine_loop:
            result = await self._generate_fast(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                enable_thinking=enable_thinking,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                json_schema=json_schema,
                cancel_event=cancel_event,
                logits_processors=logits_processors,
                priority=priority,
                timeout_seconds=timeout_seconds or 300.0,
                lora_adapter=lora_adapter,
                kv_cache_breakpoints=kv_cache_breakpoints,
                min_tokens=min_tokens,
                ignore_eos=ignore_eos,
                suppress_tokens=suppress_tokens,
            )
            if _rc_hash is not None and result.finish_reason != "error":
                try:
                    from .gateway_optimizer import get_response_cache

                    await get_response_cache().put(_rc_hash, result)
                except Exception:
                    logger.debug("response cache store failed", exc_info=True)
            # prompt_logprobs — one extra prompt forward (eval/perplexity).
            # Opt-in, length-gated (full [seq,vocab] logits are large), and run on
            # the MLX executor so it doesn't block the event loop. Never fails the
            # request — best-effort attach.
            if prompt_logprobs is not None and result.finish_reason != "error":
                try:
                    _pl = await self._compute_prompt_logprobs_for(
                        prompt,
                        enable_thinking,
                        int(prompt_logprobs),
                    )
                    result.prompt_logprobs = _pl
                except Exception:
                    logger.debug("prompt_logprobs computation failed", exc_info=True)
            return result

        # Engine loop path: continuous batching with scheduler overhead
        result = await self._engine_core.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            stop=stop,
            stop_token_ids=stop_token_ids,
            seed=seed,
            json_schema=json_schema,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            priority=priority,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            reasoning_effort=reasoning_effort,
            logits_processors=logits_processors,
            cancel_event=cancel_event,
            lora_adapter=lora_adapter,
            kv_cache_breakpoints=kv_cache_breakpoints,
            # The engine-loop branch silently dropped min_tokens / ignore_eos /
            # suppress_tokens — the scheduler's _make_sampler honors them but they
            # were never threaded here, so an engine-loop request ignored all three (e.g.
            # min_tokens floor, ignore_eos throughput runs, suppress_tokens bans). The fast
            # path passes them (above); make the opt-in loop match.
            min_tokens=min_tokens,
            ignore_eos=ignore_eos,
            suppress_tokens=suppress_tokens,
        )

        if result is None:
            return GenerationOutput(
                finished=True, finish_reason="error", error="engine_core returned None"
            )

        # Map engine_core finish_reason to OpenAI-compatible finish_reason
        finish_reason = result.finish_reason
        if finish_reason == "memory_exceeded":
            finish_reason = "memory_limit"
        # Guard: finish_reason must never be None when finished=True
        if finish_reason is None:
            finish_reason = "stop"

        # Apply output parser to extract reasoning/tool_calls from raw text
        output_text = _clean_special_tokens(result.output_text)
        _reasoning_tok = 0
        try:
            from .output_parser import parse_output

            parsed = parse_output(output_text, self.model_name)
            if parsed.finish_reason:
                finish_reason = parsed.finish_reason
            # Use cleaned content (reasoning stripped) as the main text
            if parsed.reasoning and parsed.content != output_text:
                output_text = parsed.content
        except Exception:
            logger.debug("output_parser failed", exc_info=True)

        # Reasoning parser: model-specific reasoning extraction with token count
        # Supplements output_parser with per-model reasoning token counting.
        # Fall back to scheduler-computed reasoning_tokens when parser returns 0.
        _reasoning_tok = getattr(result, "reasoning_tokens", 0) or 0
        try:
            from .reasoning_parser import get_reasoning_parser

            rp = get_reasoning_parser(self.model_name)
            rp_out = rp.parse(output_text)
            if rp_out.reasoning and rp_out.reasoning_tokens > 0:
                _reasoning_tok = rp_out.reasoning_tokens
        except Exception:
            logger.debug("reasoning_parser failed", exc_info=True)

        # TTFT from engine_core (computed before request cleanup)
        _ttft_ms = getattr(result, "ttft_ms", 0.0)

        # Record TTFT in Prometheus (consistency with fast path)
        if _ttft_ms > 0:
            try:
                from yunshu_gateway.middleware.prometheus_exporter import (
                    get_prometheus_metrics,
                )

                pm = get_prometheus_metrics()
                pm.observe_histogram(
                    "ttft_seconds",
                    _ttft_ms / 1000.0,
                    labels={"model_id": self.model_label},
                )
            except Exception:
                logger.debug(
                    "engine loop TTFT prometheus recording failed", exc_info=True
                )

        engine_loop_result = GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finished=True,
            finish_reason=finish_reason,
            reasoning_tokens=_reasoning_tok,
            cached_tokens=getattr(result, "cached_tokens", 0),
            logprobs=getattr(result, "logprobs", None),
            ttft_ms=_ttft_ms,
            error=getattr(result, "error", None),
        )
        if _rc_hash is not None and engine_loop_result.finish_reason != "error":
            try:
                from .gateway_optimizer import get_response_cache

                await get_response_cache().put(_rc_hash, engine_loop_result)
            except Exception:
                logger.debug("response cache store failed", exc_info=True)
        return engine_loop_result

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        json_schema: dict | str | None = None,
        spec_decode: bool = False,
        use_engine_loop: bool | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        reasoning_effort: str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        top_n_sigma: float = 0.0,
        priority: int = 0,
        logprobs: bool | int = False,
        top_logprobs: int | None = None,
        logits_processors: list | None = None,
        cancel_event: asyncio.Event | None = None,
        timeout_seconds: float | None = None,
        images: list | None = None,
        lora_adapter: str | None = None,
        kv_cache_breakpoints: list[int] | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        suppress_tokens: list[int] | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Streaming text generation.

        Default uses fast path (direct generate_step on executor) for single
        requests. Set use_engine_loop=True for continuous batching path.
        If use_engine_loop is None, uses YUNSHU_ENGINE_LOOP env var.

        When cancel_event is provided (e.g. from gateway disconnect detection),
        it is checked alongside the engine's internal cancel event to allow
        cooperative cancellation from the HTTP layer.
        """
        if not self._loaded:
            await self.start()

        # publish per-request top-nσ for _build_temp_sampler (same event-loop task)
        _REQUEST_TOP_N_SIGMA.set(
            top_n_sigma if top_n_sigma and top_n_sigma > 0 else None
        )
        _use_engine_loop = self._should_use_engine_loop(use_engine_loop)
        # Streaming speculation follows the route table (see _spec_route): only
        # the experimental EAGLE / MTP routes stream; everything else is plain.
        _spec = self._spec_route(
            spec_decode=spec_decode,
            stream=True,
            temperature=temperature,
            logprobs=bool(logprobs),
            use_engine_loop=_use_engine_loop,
        )
        spec_decode = _spec is not None
        # Resolve reasoning_effort → thinking_budget if not explicitly set
        if thinking_budget is None and reasoning_effort is not None:
            thinking_budget = _REASONING_EFFORT_MAP.get(reasoning_effort, 8192)
            if enable_thinking is None:
                enable_thinking = True

        # Gemma-4 default: its chat template enables thinking by default but
        # emits inline `thought` tokens that don't auto-stop, producing output
        # like "4thought\nThinking Process: …" for simple prompts. Default
        # enable_thinking=False for gemma-4 unless caller passed reasoning_effort.
        if enable_thinking is None and reasoning_effort is None:
            _mn = getattr(self, "model_name", None)
            if isinstance(_mn, str) and "gemma-4" in _mn.lower():
                enable_thinking = False

        # Early return: max_tokens <= 0 produces no output (matches generate() guard)
        if max_tokens <= 0:
            yield GenerationOutput(
                text="",
                new_text="",
                prompt_tokens=0,
                completion_tokens=0,
                finished=True,
                finish_reason="length",
            )
            return

        # Memory guard preflight check
        guard_rejection = self._check_memory_guard(prompt, max_tokens)
        if guard_rejection is not None:
            yield guard_rejection
            return

        # Register with request tracker for cancellation support (all paths)
        import uuid as _uuid

        _stream_req_id = f"stream-{_uuid.uuid4().hex[:8]}"
        try:
            from .request_tracker import get_request_tracker

            if cancel_event is not None:
                # The gateway already registered this request (its event carries the
                # live RunStats and the client's request id); a second registration
                # would take over both. Cancellation is the gateway's event alone.
                _tracker = None
                _cancel_event = cancel_event
            else:
                _tracker = get_request_tracker()
                _active_gen = _tracker.register(_stream_req_id, self.model_name or "")
                _cancel_event = _active_gen.cancel_event
        except Exception:
            logger.debug("request tracker registration failed", exc_info=True)
            _cancel_event = None
            _tracker = None

        # If the gateway passes an external cancel_event, wrap both events
        # so that checking .is_set() on the wrapper detects either source.
        if cancel_event is not None and _tracker is not None:
            _internal = _cancel_event
            _external = cancel_event

            class _CompositeCancelEvent:
                """Proxy that returns True if either the internal or external event is set.

                Thread-safe: reads asyncio.Event._value directly (GIL-protected bool)
                instead of calling .is_set() which is not safe from executor threads.
                """

                def is_set(self):
                    if _internal is not None:
                        # asyncio.Event: read _value (GIL-protected bool)
                        if hasattr(_internal, "_value"):
                            if _internal._value:
                                return True
                        elif _internal.is_set():
                            return True
                    return _is_cancelled(_external)

            _cancel_event = _CompositeCancelEvent()

        # Fast path: bypass EngineCore for single-request streaming
        if not _use_engine_loop:
            try:
                async for output in self._stream_generate_fast(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                    stop=stop,
                    stop_token_ids=stop_token_ids,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    json_schema=json_schema,
                    cancel_event=_cancel_event,
                    logprobs=bool(logprobs),
                    top_logprobs=top_logprobs,
                    logits_processors=logits_processors,
                    priority=priority,
                    timeout_seconds=timeout_seconds or 300.0,
                    lora_adapter=lora_adapter,
                    kv_cache_breakpoints=kv_cache_breakpoints,
                    min_tokens=min_tokens,
                    ignore_eos=ignore_eos,
                    suppress_tokens=suppress_tokens,
                ):
                    yield output
            finally:
                if _tracker is not None:
                    try:
                        _tracker.unregister(_stream_req_id)
                    except Exception:
                        logger.debug("request tracker cleanup failed", exc_info=True)
            return

        # Engine loop path: continuous batching with scheduler
        await self._ensure_engine_core()
        _stream_t0 = time.perf_counter()
        request_id = await self._engine_core.add_request(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            stop=stop,
            stop_token_ids=stop_token_ids,
            seed=seed,
            json_schema=json_schema,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            priority=priority,
            logprobs=bool(logprobs),
            top_logprobs=top_logprobs,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            reasoning_effort=reasoning_effort,
            logits_processors=logits_processors,
            images=images,
            lora_adapter=lora_adapter,
        )

        finished_normally = False
        _first_token = True
        _stream_ttft_ms = 0.0
        try:
            async for output in self._engine_core.stream_outputs(
                request_id, cancel_event=_cancel_event
            ):
                # Check cancel event (gateway disconnect or internal cancel)
                if _cancel_event is not None and _cancel_event.is_set():
                    logger.debug(
                        f"Cancel event triggered during streaming: {request_id}"
                    )
                    # Yield terminal stop chunk so consumer sees finished=True
                    yield GenerationOutput(
                        text="",
                        new_text="",
                        prompt_tokens=getattr(output, "prompt_tokens", 0)
                        if hasattr(output, "prompt_tokens")
                        else 0,
                        completion_tokens=getattr(output, "completion_tokens", 0)
                        if hasattr(output, "completion_tokens")
                        else 0,
                        finished=True,
                        finish_reason="stop",
                        ttft_ms=_stream_ttft_ms,
                        cached_tokens=getattr(output, "cached_tokens", 0),
                        reasoning_tokens=getattr(output, "reasoning_tokens", 0),
                    )
                    break
                cleaned = _clean_special_tokens(output.new_text)
                finish_reason = output.finish_reason
                if finish_reason == "memory_exceeded":
                    finish_reason = "memory_limit"
                # Guard: finish_reason must never be None when finished=True
                if output.finished and finish_reason is None:
                    finish_reason = "stop"
                # Compute TTFT on first streamed output
                _ttft_ms = 0.0
                if _first_token:
                    _stream_ttft_ms = round(
                        (time.perf_counter() - _stream_t0) * 1000, 1
                    )
                    _ttft_ms = _stream_ttft_ms
                    _first_token = False
                    # Record TTFT in Prometheus (consistency with fast path)
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _stream_ttft_ms / 1000.0,
                            labels={"model_id": self.model_label},
                        )
                    except Exception:
                        logger.debug(
                            "engine loop streaming TTFT prometheus recording failed",
                            exc_info=True,
                        )
                gen_output = GenerationOutput(
                    text=_clean_special_tokens(output.output_text),
                    new_text=cleaned,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=finish_reason,
                    reasoning_tokens=getattr(output, "reasoning_tokens", 0),
                    cached_tokens=getattr(output, "cached_tokens", 0),
                    logprobs=getattr(output, "logprobs", None),
                    ttft_ms=_ttft_ms,
                    current_state=getattr(output, "current_state", None),
                    error=getattr(output, "error", None),
                    prefill_progress=getattr(output, "prefill_progress", None),
                )
                if output.finished:
                    finished_normally = True
                yield gen_output
        except GeneratorExit:
            logger.debug(f"Client disconnected during streaming: {request_id}")
        finally:
            if not finished_normally:
                try:
                    await self._engine_core.abort_request(request_id)
                except Exception:
                    logger.debug("stream abort failed", exc_info=True)
            if _tracker is not None:
                try:
                    _tracker.unregister(_stream_req_id)
                except Exception:
                    logger.debug("request tracker cleanup failed", exc_info=True)

    def _init_spec_decode(self) -> None:
        """Initialize speculative decoding if the model supports it.

        Text serving supports n-gram/suffix, Gemma-4 assistant, and the explicit
        unverified external LM experiment. Qwen native MTP/DFlash belongs to
        the VLM batch runner. No trained EAGLE checkpoint is required by the
        generic external LM route (despite its historical "eagle" name).
        """
        # Always initialize N-gram proposer (model-free, zero overhead when idle)
        from .ngram_proposer import NgramConfig, NgramProposer

        max_n, k, mode = 5, 5, "lps"
        # Spec proposer family (shares the lossless verify path + base loop):
        #   "ngram"  (default) — fixed-length n-gram suffix→continuation, O(1)
        #   "suffix" — Suffix Decoding (arXiv 2411.04975): LONGEST-suffix match,
        #              stronger on repetitive output (code/JSON/agentic loops).
        # Both are lossless by construction — verify_with_last_token only
        # accepts the model's own argmax, so the proposer affects speed, not
        # output. The slot below is the single "spec proposer" the routing
        # gates and per-request construction read.
        _proposer_kind = settings.get("YUNSHU_SPEC_PROPOSER")
        if _proposer_kind == "suffix":
            from .suffix_proposer import SuffixConfig, SuffixProposer

            self._ngram_proposer = SuffixProposer(
                SuffixConfig(max_draft=k, max_trie_depth=max(max_n, 16) * 4)
            )
        else:
            self._ngram_proposer = NgramProposer(
                NgramConfig(max_n=max_n, k=k, mode=mode)
            )
        # Correctness history: the earlier gross corruption (dropped/
        # duplicated tokens — "2, 4, 6"→"246", "…the average speed"→"…the
        # average average") was NOT the verifier: it was the prefill+base loop
        # driving the KV cache with generate_step's prefill-break + per-token
        # feed while the verify path used direct model() calls — the mixed
        # access skewed the cache offset. Both loops now use direct model()
        # forwards (consistent with verify_with_last_token), which is lossless:
        #   • speculation is exact — full-spec output == base-only (no-spec)
        #     output, byte-for-byte;
        #   • byte-identical to the fast path on short/medium greedy gen across
        #     gemma-4-e4b, Qwen2.5-3B-4bit (spec actually running) and
        #     Qwen3.5-2B (hybrid cache → spec correctly disabled).
        # A residual single-token divergence can appear DEEP in long greedy gen
        # (~1.4k chars in): the n-gram loop (direct model() + mlx sampler) and
        # the fast loop (generate_step + numpy sampler) break a near-tie argmax
        # differently. It is FP-level path difference — both are valid greedy
        # decodes, same class as the cross-process non-determinism this engine
        # already has — NOT corruption.
        #
        # DEFAULT OFF (measured 2026-06-30): n-gram spec is lossless but NOT free —
        # it batch-verifies K draft tokens per step, so when acceptance is low (normal
        # prose, code — the common case) it pays K× compute for ~1 token and is SLOWER.
        # Measured on Qwen2.5-3B-4bit greedy: ~2.5× slower on normal/code, ~1.2× faster
        # only on highly repetitive output; on Qwen3.5-2B-bf16 ~3% slower everywhere.
        # So it's opt-in (YUNSHU_NGRAM_DEFAULT=1) for repetitive/agentic workloads where
        # it wins, not a safe global default. Routing: see _spec_route (per-request
        # spec_decode=true or YUNSHU_NGRAM_DEFAULT=1; greedy, non-streaming only).
        self._ngram_greedy_default = settings.get_bool("YUNSHU_NGRAM_DEFAULT")
        logger.info(
            "Spec proposer initialized: kind=%s, max_n=%d, k=%d, mode=%s, greedy_default=%s",
            _proposer_kind,
            max_n,
            k,
            mode,
            self._ngram_greedy_default,
        )

        # Adaptive spec controller (requires N-gram proposer active). ON by default
        # whenever spec runs: it sets the draft length K from acceptance feedback and
        # backs off to K=0 on low acceptance, which BOUNDS spec's downside. Measured
        # 2026-06-30 (Qwen2.5-3B-4bit, greedy): without it, default-on spec was ~2.5×
        # SLOWER on low-acceptance output (normal prose / code); with it, that becomes
        # ~1.15× while the ~1.7× win on repetitive/agentic output is kept.
        #
        # It does NOT make spec strictly never-slower, so YUNSHU_NGRAM_DEFAULT stays 0
        # (opt-in): even the controller's K=0 step runs the spec loop's own plain branch
        # (a direct model() forward), which is ~15% slower than _generate_fast's
        # generate_step — a structural cost of being inside _generate_ngram_spec at all.
        # Eliminating it would mean routing idle steps back through generate_step (major
        # surgery, deferred). So this just makes the OPT-IN spec path safe + self-tuning.
        if self._ngram_proposer is not None:
            from .adaptive_spec import AdaptiveSpecController

            self._adaptive_spec = AdaptiveSpecController()

        # Gemma-4 assistant drafter (external EAGLE-style drafter, KV-shared with
        # target). Independent of target spec heads, so initialize before the
        # head-detection early return below.
        self._init_gemma4_assistant_spec()
        self._init_external_lm_spec()

    def _init_external_lm_spec(self) -> None:
        """Build the explicit external LM experiment; no EAGLE head is needed.

        A normal mlx-lm checkpoint suffices. Keep this reachable route behind
        both experimental settings; never load a second model for an unset
        unverified switch. Native text MTP backends have no serving dispatch.
        """
        if settings.get("YUNSHU_SPEC_UNVERIFIED") != "eagle":
            return
        draft_path = settings.get("YUNSHU_DRAFT_MODEL")
        if not draft_path:
            config_obj = getattr(self._model, "config", None) or getattr(
                self._model, "args", None
            )
            if config_obj is not None:
                if hasattr(config_obj, "to_dict"):
                    model_config = config_obj.to_dict()
                else:
                    model_config = vars(config_obj)
                draft_path = model_config.get("draft_model_path")
        if not draft_path:
            return
        try:
            from mlx_lm.utils import load as load_model

            from .speculative_decoder import SpecDecodingConfig, SpeculativeDecoder

            draft_model = load_model(draft_path)[0]
            self._spec_decoder = SpeculativeDecoder(
                self._model,
                draft_model,
                self._tokenizer,
                config=SpecDecodingConfig(draft_temperature=0.0),
                lookahead=self._lookahead_reasoning,
            )
            self._spec_enabled = True
            logger.info("Unverified external LM draft loaded from %s", draft_path)
        except Exception:
            logger.warning("External LM draft load failed", exc_info=True)

    def _init_gemma4_assistant_spec(self) -> None:
        """Build the Gemma-4 dual-load assistant drafter when configured.

        Gated by ``YUNSHU_GEMMA4_ASSISTANT`` (path to the assistant-drafter dir).
        The drafter is an external EAGLE-style proposer that shares the target's
        KV cache; it consumes the target's token embedding + last hidden state
        (not an autonomous LM), so it is driven via ``spec_decode_generate``
        rather than the scheduler's generic draft-model path. Validated ~1.27×
        (greedy-only). Graceful no-op when unset or on any load failure.
        """
        drafter_dir = settings.get("YUNSHU_GEMMA4_ASSISTANT")

        if not drafter_dir:
            return
        try:
            inner = getattr(self._model, "language_model", self._model)
            tm = getattr(inner, "model", inner)
            embed = tm.embed_tokens
            embed_scale = float(getattr(tm, "embed_scale", 1.0))
            # Read config.json directly: the loaded mlx_lm config object flattens
            # away nested keys (text_config.layer_types, num_kv_shared_layers) that
            # the drafter's KV-share resolution needs.
            import json as _json
            from pathlib import Path

            mp = Path(self.model_name)
            if not (mp / "config.json").exists():
                from mlx_lm.utils import hf_repo_to_path

                mp = Path(hf_repo_to_path(self.model_name))
            tcfg = _json.loads((mp / "config.json").read_text())
            from .gemma4_assistant import Gemma4AssistantProposer

            self._gemma4_assistant_proposer = Gemma4AssistantProposer.from_paths(
                drafter_dir, embed.weight, embed_scale, tcfg
            )
            logger.info(
                f"Gemma-4 assistant drafter loaded from {drafter_dir}: "
                f"spec decode primitive ACTIVE (sliding_kv={self._gemma4_assistant_proposer.sliding_kv_layer}, "
                f"full_kv={self._gemma4_assistant_proposer.full_kv_layer})"
            )
        except Exception as e:
            logger.warning(f"Gemma-4 assistant drafter init failed ({e}), skipping")
            self._gemma4_assistant_proposer = None


from .text_utils import cache_tokenizer_vocab as _cache_tokenizer_vocab
