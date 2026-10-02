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
from typing import Any as Any

from . import settings as settings
from .context_window import ContextBudgetError as ContextBudgetError
from .context_window import reject_overlong_prompt as reject_overlong_prompt
from .fast_path_stats import FastPathStats as FastPathStats
from .stream_bridge import StreamBridge as StreamBridge
from .stream_bridge import make_stream_queue as make_stream_queue
from .text_utils import StopHoldbackBuffer as StopHoldbackBuffer

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

        # Speculative decoding state (Phase 4)
        self._spec_decoder = None  # SpeculativeDecoder instance
        self._spec_enabled = False

        # Gemma-4 dual-load assistant drafter (EAGLE-style external drafter that
        # shares the target's KV; validated 2.08×). Gated by YUNSHU_GEMMA4_ASSISTANT.
        self._gemma4_assistant_proposer = None  # Gemma4AssistantProposer

        # MTP speculative decoding (built-in multi-token prediction heads)
        self._mtp_decoder = None  # MTPDecoder instance
        self._mtp_strategy = None  # MTPStrategy wrapper

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
        # Experimental MTP for text-only models (YUNSHU_SPEC_UNVERIFIED=mlxvlm_mtp;
        # YUNSHU_MTP is the VLM runner's MTP draft switch). mlx-vlm is the supported MTP
        # implementation (its speculative path is the only correct one — see mlxvlm_mtp.py).
        # This text-engine MTP backend is single-backend + honors only
        # temperature (drops top_p/json_schema/penalties, see _warn_mtp_dropped_params) +
        # non-streaming. Treat it as EXPERIMENTAL, not a shipped prod win; the real spec win
        # is the gemma-4 assistant drafter. When set and the model has native MTP weights,
        # the engine serves via that backend (single-backend swap, no dual-load): greedy →
        # MTP; sampling → plain gen on the same model.
        self._mlxvlm_mtp_enabled = (
            settings.get("YUNSHU_SPEC_UNVERIFIED") == "mlxvlm_mtp"
        )
        self._mlxvlm_mtp = None

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

        # mlx-vlm native MTP single-backend mode. When enabled
        # and the checkpoint has native MTP weights, load the mlx-vlm target +
        # MTP drafter on the executor thread and skip the mlx-lm fast-path setup
        # entirely (no dual-load). chat() delegates to this backend.
        if self._mlxvlm_mtp_enabled:
            try:
                from .mlxvlm_mtp import MLXVLMMtp, is_mtp_capable

                if is_mtp_capable(self.model_name):
                    backend = MLXVLMMtp(self.model_name)
                    await loop.run_in_executor(executor, backend.load)
                    self._mlxvlm_mtp = backend
                    self._tokenizer = backend.tokenizer
                    _cache_tokenizer_vocab(self._tokenizer)  # missed this twin
                    self._loaded = True
                    logger.info("mlx-vlm MTP backend active for %s", self.model_name)
                    return
                logger.info(
                    "YUNSHU_SPEC_UNVERIFIED=mlxvlm_mtp set but %s is not MTP-capable; "
                    "using standard path",
                    self.model_name,
                )
            except Exception:
                logger.warning(
                    "mlx-vlm MTP backend load failed; falling back to standard path",
                    exc_info=True,
                )

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

        # The home-grown Qwen3.5 MTP (n_confirmed_patch + mtp_patch +
        # mtp_decoder) lacked mlx-vlm's GatedDeltaNet intermediate-state capture
        # (garbage on 27B, ~0.9x on 9B); Qwen3.5-family MTP is served by the VLM
        # runner. Kept only as the experimental YUNSHU_SPEC_UNVERIFIED=mtp route.
        if settings.get("YUNSHU_SPEC_UNVERIFIED") == "mtp":
            try:
                from .n_confirmed_patch import apply_n_confirmed_patch

                if apply_n_confirmed_patch():
                    logger.info("[experimental] n_confirmed patch applied")
            except Exception:
                logger.debug("n_confirmed patch skipped", exc_info=True)
            try:
                from .mtp_patch import apply_mtp_patch

                if apply_mtp_patch():
                    logger.info("[experimental] MTP patch applied")
            except Exception:
                logger.warning("MTP patch skipped", exc_info=True)

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
        self._init_spec_decode()

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
        if s.spec_decode_enabled:
            self._spec_enabled = True

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

        Called after _ensure_engine_core() creates the scheduler and after
        _init_spec_decode() has detected spec heads and created decoders.

        Three paths:
          1. Cross-model: SpeculativeDecoder → scheduler.set_spec_decoder()
          2. MTP: MTPDecoder → scheduler.set_mtp_decoder()
          3. N-gram: already handled by SchedulerConfig.ngram_spec_enabled

        Also propagates N-gram proposer settings to the scheduler config
        if BatchedEngine has one but the scheduler doesn't.
        """
        if self._engine_core is None:
            return
        scheduler = self._engine_core.scheduler

        # Path 1: Cross-model speculative decoder (EAGLE-3 / external draft)
        if self._spec_decoder is not None:
            scheduler.set_spec_decoder(self._spec_decoder)
            logger.info("Wired cross-model spec decoder into scheduler batch path")

        # Path 2: MTP decoder (built-in multi-token prediction heads)
        if self._mtp_decoder is not None:
            scheduler.set_mtp_decoder(self._mtp_decoder)
            logger.info("Wired MTP decoder into scheduler batch path")

        # Path 3: Propagate N-gram proposer to scheduler if needed
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
        self._gemma4_assistant_proposer = None
        self._ngram_proposer = None
        self._adaptive_spec = None
        self._mtp_decoder = None
        self._mtp_strategy = None
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
    # | eagle / mtp      | only with YUNSHU_SPEC_UNVERIFIED=eagle|mtp, greedy,  | NOT verified lossless: EAGLE's acceptance  |
    # |                  | (eagle never streams: its streamer has no stop       |                                            |
    # |                  | hold-back, so multi-token stops would leak)          |                                            |
    # |                  | fast path, no logprobs (experiments)                 | ignores request temperature; both give     |
    # |                  |                                                      | empty output on non-trimmable (hybrid)     |
    # |                  |                                                      | caches; no measured win on Apple Silicon.  |
    # |                  |                                                      | Qwen3.5-family MTP lives in the VLM runner.|
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
    ) -> str | None:
        if use_engine_loop or logprobs:
            return None
        greedy = temperature is None or temperature <= 0.0
        experimental = settings.get("YUNSHU_SPEC_UNVERIFIED")
        if spec_decode and greedy and experimental == "eagle" and not stream:
            if getattr(self, "_spec_enabled", False) and getattr(
                self, "_spec_decoder", None
            ):
                return "eagle"
        if spec_decode and greedy and experimental == "mtp":
            if getattr(self, "_mtp_decoder", None) is not None:
                return "mtp"
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

        # Speculative decoding path (Phase 4: single-request EAGLE-3).
        # NB: all spec paths below are gated on `not logprobs` — none of them
        # populate per-token logprobs, so when logprobs are requested correctness
        # wins and we fall through to normal generation (which does).
        # The cross-model SpeculativeDecoder omits residual-
        # distribution resampling and tests acceptance at temperature 1.0
        # regardless of request temperature, so it is NOT lossless for
        # temperature>0 (output distribution is biased toward the greedy
        # sequence). Restrict it to GREEDY requests, where longest-exact-argmax
        # acceptance IS lossless; temp>0 falls through to correct normal decode.
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

        # MTP speculative decoding (built-in multi-token prediction heads)
        if _spec == "mtp":
            return await self._generate_mtp(
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
                cancel_event=cancel_event,
                json_schema=json_schema,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
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

        # Speculative decoding path (Phase 4)
        if _spec == "eagle":
            try:
                async for output in self._stream_generate_speculative(
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
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    stop=stop,
                    stop_token_ids=stop_token_ids,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    json_schema=json_schema,
                    cancel_event=_cancel_event,
                    logits_processors=logits_processors,
                    lora_adapter=lora_adapter,
                ):
                    yield output
            finally:
                if _tracker is not None:
                    try:
                        _tracker.unregister(_stream_req_id)
                    except Exception:
                        logger.debug("request tracker cleanup failed", exc_info=True)
            return

        # MTP speculative decoding streaming (built-in mlx-lm MTPDecoder path),
        # experimental route only (YUNSHU_SPEC_UNVERIFIED=mtp).
        if _spec == "mtp":
            try:
                async for output in self._stream_generate_mtp(
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
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    stop=stop,
                    stop_token_ids=stop_token_ids,
                    seed=seed,
                    cancel_event=_cancel_event,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    timeout_seconds=timeout_seconds or 300.0,
                    json_schema=json_schema,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    logits_processors=logits_processors,
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

    async def _warm_prompt_prefill(self) -> None:
        """Prefill KV cache with warm prompts on startup.

        Delegates to ModelWarmupManager.warm_prompt_prefill() which:
        - Reads prompts from YUNSHU_WARM_PROMPTS env var (||-separated)
        - Tokenizes and prefills each into KV prefix cache
        - Future requests with matching prefixes get instant cache hits

        Provides 1.3-2.25x TTFT improvement on first real request with
        matching prefix (vllm-mlx pattern).
        """
        from .model_optimizations import ModelWarmupManager

        mgr = ModelWarmupManager()

        # Quick check: skip entirely if no warm prompts configured
        if not mgr.resolve_warm_prompts():
            return

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        def _prefill_all():
            return mgr.warm_prompt_prefill(
                model=self._model,
                tokenizer=self._tokenizer,
                kv_prefix_cache=self._kv_prefix_cache,
                max_tokens=1,
                mem_pressure_threshold=self._mem_pressure_threshold,
            )

        try:
            result = await loop.run_in_executor(executor, _prefill_all)
            # Store result in engine stats
            self._warm_prompt_stats = {
                "prompts_loaded": result.prompts_loaded,
                "prompts_prefilled": result.prompts_prefilled,
                "prompts_skipped_cached": result.prompts_skipped_cached,
                "prompts_failed": result.prompts_failed,
                "total_tokens_prefilled": result.total_tokens_prefilled,
                "prefill_time_s": result.prefill_time_s,
                "source": result.source,
            }
            if result.prompts_prefilled > 0:
                logger.info(
                    f"Warm prompt prefill: {result.prompts_prefilled} prefilled, "
                    f"{result.prompts_skipped_cached} cached, "
                    f"{result.total_tokens_prefilled} tokens, "
                    f"{result.prefill_time_s:.3f}s"
                )
        except Exception as e:
            logger.warning(f"Warm prompt prefill failed: {e}")

    def _init_spec_decode(self) -> None:
        """Initialize speculative decoding if the model supports it.

        Called during start() after model loading. Checks for spec heads in the
        model config and creates a SpeculativeDecoder if detected.
        Also initializes N-gram proposer as a model-free fallback.
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

        from .speculative_decoder import auto_configure_speculative, detect_spec_heads

        model_config = {}
        config_obj = getattr(self._model, "config", None) or getattr(
            self._model, "args", None
        )
        if config_obj is not None:
            if hasattr(config_obj, "to_dict"):
                model_config = config_obj.to_dict()
            elif hasattr(config_obj, "__dict__"):
                model_config = {
                    k: v
                    for k, v in config_obj.__dict__.items()
                    if not k.startswith("_")
                }

        head_info = detect_spec_heads(model_config)

        # An EXTERNAL cross-model draft (YUNSHU_DRAFT_MODEL /
        # config draft_model_path) is independent of the target's native spec
        # heads — it works for ANY model. Load it BEFORE the no-native-heads
        # early return below, otherwise an explicitly-configured draft model was
        # silently ignored for models without native heads (e.g. plain Qwen2.5),
        # and a `spec_decode: true` request fell through to the n-gram path.
        draft_path = settings.get("YUNSHU_DRAFT_MODEL")
        if not draft_path and model_config:
            draft_path = model_config.get("draft_model_path", "")
        if draft_path:
            try:
                from mlx_lm.utils import load as load_model

                draft_model, _ = load_model(draft_path)
                from .speculative_decoder import SpeculativeDecoder

                self._spec_decoder = SpeculativeDecoder(
                    self._model,
                    draft_model,
                    self._tokenizer,
                    lookahead=self._lookahead_reasoning,
                )
                self._spec_enabled = True
                logger.info(
                    f"Draft model loaded from {draft_path}: "
                    f"speculative decoding ACTIVE (type={head_info.head_type})"
                )
            except Exception as e:
                logger.warning(
                    f"Draft model load failed ({e}), speculative decoding disabled"
                )

        if head_info.head_type == "none":
            logger.debug("No speculative decoding heads detected (native)")
            return  # external draft, if any, already loaded above

        spec_config = auto_configure_speculative(model_config)
        if spec_config.draft_length == 0:
            return

        logger.info(
            f"Speculative decoding available: type={head_info.head_type}, "
            f"draft_length={spec_config.draft_length}"
        )

        # The home-grown MTP decoder lacked mlx-vlm's GatedDeltaNet
        # intermediate-state capture (garbage on 27B, ~0.9x on 9B). Both text
        # MTP backends are experimental (YUNSHU_SPEC_UNVERIFIED=mtp|mlxvlm_mtp).
        if (
            head_info.head_type == "mtp"
            and settings.get("YUNSHU_SPEC_UNVERIFIED") != "mtp"
        ):
            logger.info(
                "Native MTP head detected on %s; text MTP is experimental "
                "(YUNSHU_SPEC_UNVERIFIED=mlxvlm_mtp or mtp).",
                self.model_name,
            )
        elif head_info.head_type == "mtp" and self._spec_decoder is None:
            try:
                # Load MTP head weights if available
                inner = getattr(self._model, "language_model", self._model)
                if not hasattr(inner, "mtp"):
                    try:
                        from .mtp_patch import load_model_with_mtp

                        model_name_or_path = model_config.get(
                            "_name_or_path", self.model_name
                        )
                        self._model = load_model_with_mtp(model_name_or_path)
                        logger.info("MTP head weights loaded from model directory")
                    except FileNotFoundError as e:
                        logger.info(
                            f"MTP weights not found ({e}), using backbone-only MTP"
                        )
                    except Exception as e:
                        logger.warning(
                            f"MTP weights load failed ({e}), using backbone-only MTP"
                        )

                from .mtp_decoder import MTPConfig, MTPDecoder

                mtp_config = MTPConfig(
                    max_tokens=256,
                    use_n_confirmed=True,
                )
                self._mtp_decoder = MTPDecoder(
                    self._model,
                    self._tokenizer,
                    mtp_config,
                )
                # (: the _mtp_strategy wrapper was dead — routing uses
                # _mtp_decoder directly; removed the unread MTPStrategy build.)
                self._spec_enabled = True
                logger.info(
                    f"MTP decoder initialized: heads={head_info.num_heads}, "
                    f"draft_length={head_info.draft_length}, "
                    f"n_confirmed=True, config={head_info.head_config}"
                )
            except Exception as e:
                logger.warning(f"MTP decoder init failed ({e})")

        # NOTE : Medusa, and the unified SpecStrategyFactory strategies
        # (GPUNgram/LLM/Suffix/DFlash/Composite), were built into engine state but
        # NEVER consulted by generate()/generate_stream() routing — only by the
        # now-deleted _get_spec_strategy() (zero serving callers). They were dead
        # code claiming to be working spec strategies. Removed to keep the spec
        # surface honest. The real, reachable spec paths are: N-gram (default),
        # MTP (model with prediction heads), cross-model (YUNSHU_DRAFT_MODEL), and
        # the Gemma-4 assistant drafter (YUNSHU_GEMMA4_ASSISTANT). Medusa/the
        # factory strategies remain available as research modules in their own
        # files but are not wired into serving.

        # Store config for on-demand decoder creation
        self._spec_config = spec_config
        self._spec_head_info = head_info
        self._spec_enabled = True

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
            self._spec_enabled = True
            logger.info(
                f"Gemma-4 assistant drafter loaded from {drafter_dir}: "
                f"spec decode primitive ACTIVE (sliding_kv={self._gemma4_assistant_proposer.sliding_kv_layer}, "
                f"full_kv={self._gemma4_assistant_proposer.full_kv_layer})"
            )
        except Exception as e:
            logger.warning(f"Gemma-4 assistant drafter init failed ({e}), skipping")
            self._gemma4_assistant_proposer = None

    def gemma4_spec_generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 256,
        k: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> list[int]:
        """Single-sequence generation via the Gemma-4 assistant drafter.

        Reachable serving path for the dual-load spec-decode primitive (validated
        2.08×): builds the target KV cache and drives ``spec_decode_generate``.
        ``temperature == 0`` → greedy (target's greedy sequence); ``> 0`` →
        distribution-correct speculative sampling. Requires
        ``YUNSHU_GEMMA4_ASSISTANT`` to have been set so the proposer was built at
        start(). Raises RuntimeError if the proposer is unavailable.
        """
        if self._gemma4_assistant_proposer is None:
            raise RuntimeError(
                "Gemma-4 assistant drafter not initialized "
                "(set YUNSHU_GEMMA4_ASSISTANT to the drafter dir before start())"
            )
        from mlx_lm.models.cache import make_prompt_cache

        inner = getattr(self._model, "language_model", self._model)
        tm = getattr(inner, "model", inner)
        eos: set[int] = set()
        eid = getattr(self._tokenizer, "eos_token_id", None)
        if isinstance(eid, (list, tuple)):
            eos.update(eid)
        elif eid is not None:
            eos.add(eid)
        cache = make_prompt_cache(self._model)
        # apply Gemma's final_logit_softcapping to the target
        # lm_head. Model.__call__ does tanh(logits/cap)*cap (cap=30), but the spec path
        # samples straight from embed_tokens.as_linear (raw logits) → softcap bypassed.
        # Greedy (temp=0) is unaffected (softcap is monotonic), but for temp>0 the raw
        # logit spread is wider, so the spec sampling distribution diverged from normal
        # generation. Wrapping restores parity; verify acceptance (greedy-exact) is
        # unchanged since argmax is preserved.
        _softcap = getattr(self._model, "_yunshu_final_softcap", None)
        _base_head = tm.embed_tokens.as_linear
        if _softcap:
            import mlx.core as _mx_sc

            _cap = float(_softcap)

            def _lm_head(_h, _f=_base_head, _c=_cap, _mx=_mx_sc):
                return _mx.tanh(_f(_h) / _c) * _c
        else:
            _lm_head = _base_head
        return self._gemma4_assistant_proposer.spec_decode_generate(
            tm,
            _lm_head,
            cache,
            prompt_ids,
            max_tokens=max_tokens,
            k=k,
            eos_ids=eos,
            temperature=temperature,
            seed=seed,
        )

    def _gemma4_spec_eligible(
        self,
        *,
        logprobs,
        json_schema,
        logits_processors,
        logit_bias,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        xtc_probability,
        lora_adapter=None,
    ) -> bool:
        """Whether an n=1 request can be served by the Gemma-4 spec primitive.

        Only routes when the spec output would be IDENTICAL to normal generation:
        the primitive supports greedy / pure-temperature sampling but not
        top_p/top_k/min_p/penalties/grammar/logprobs/logit_bias/xtc. Gating those
        out guarantees no silent behavior change. Active only when the assistant
        drafter was loaded (``YUNSHU_GEMMA4_ASSISTANT``).

        ALSO gate out lora_adapter. `_generate_gemma4_
        assistant_spec` runs the base-weight drafter primitive and does NOT apply
        any adapter, so routing a LoRA request here would silently generate from
        the BASE model — violating the "identical to normal generation" contract.
        LoRA requests must fall through to `_generate_fast` (which applies the
        adapter atomically on the executor via the keystone).
        """
        return (
            getattr(self, "_gemma4_assistant_proposer", None) is not None
            and not logprobs
            and json_schema is None
            and not logits_processors
            and not logit_bias
            and not lora_adapter
            and (top_p is None or top_p >= 1.0)
            and (not top_k or top_k <= 0)
            and (not min_p or min_p <= 0.0)
            and repetition_penalty == 1.0
            and frequency_penalty == 0.0
            and presence_penalty == 0.0
            and (not xtc_probability or xtc_probability <= 0.0)
        )

    async def _generate_gemma4_assistant_spec(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        stop_token_ids: list[int] | None = None,
        k: int = 4,
    ) -> GenerationOutput:
        """n=1 serving via the Gemma-4 assistant drafter (validated ~2.5x, lossless).

        Builds prompt_ids, runs the proven ``spec_decode_generate`` primitive on
        the single MLX executor thread, and returns a GenerationOutput. The
        primitive is eos-terminated; stop-STRING truncation is applied by the
        gateway post-hoc (same contract as the fast path). Eligibility is gated by
        ``_gemma4_spec_eligible`` so this is only reached when the result is
        identical to normal generation.
        """
        tokenizer = self._tokenizer
        if isinstance(prompt, str):
            text = prompt
            input_ids = tokenizer.encode(text)
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # : route through the engine's _apply_chat_template + _encode_prompt
            # (NOT raw tokenizer calls) so this spec path matches the default fast path
            # exactly — role remap (developer→system, function→tool), the gemma-4 family
            # message adapter (system-message merge), dangling-<think> close, tool_calls
            # arguments str→dict normalization, AND the double-BOS guard. Bypassing them
            # diverged tool/multi-role prompts from normal generation (violating this
            # route's "identical to normal generation" contract).
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(tokenizer, text)
        else:
            text = str(prompt)
            input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        eos_ids: set[int] = set()
        eid = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)
        if stop_token_ids:
            eos_ids.update(stop_token_ids)

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        def _run():
            gen_ids = self.gemma4_spec_generate(
                input_ids,
                max_tokens=max_tokens,
                k=k,
                temperature=temperature or 0.0,
                seed=seed,
            )
            finished_by_eos = bool(gen_ids) and gen_ids[-1] in eos_ids
            visible_ids = gen_ids[:-1] if finished_by_eos else gen_ids
            # Detokenize via the per-request detokenizer, skipping special tokens
            # (e.g. Gemma-4's reserved id 100 -> "<|channel>") so the rendered text
            # is identical to normal generation. The raw detokenizer does NOT skip
            # them, so filter all_special_ids first. completion_tokens still counts
            # every generated token. Runs on the single executor thread.
            _special = set(getattr(tokenizer, "all_special_ids", None) or [])
            detok = tokenizer.detokenizer
            detok.reset()
            for tok in visible_ids:
                if tok in _special:
                    continue
                detok.add_token(tok)
            detok.finalize()
            return visible_ids, detok.text, finished_by_eos

        visible_ids, out_text, finished_by_eos = await loop.run_in_executor(
            executor, _run
        )
        ttft_ms = (time.monotonic() - t0) * 1000.0
        out_text = _clean_special_tokens(out_text)
        # (self-audit): record ServerMetrics. This is the ONE reachable
        # non-streaming spec route (YUNSHU_GEMMA4_ASSISTANT); the gateway stopped
        # double-recording (37568f5), so without this the request is UNCOUNTED —
        # same defect class as the VLM non-streaming metrics loss.
        try:
            from .server_metrics import get_server_metrics

            get_server_metrics().record_request_complete(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(visible_ids),
                prefill_duration=ttft_ms / 1000.0,
                model_id=self.model_name,
            )
        except Exception:
            logger.debug(
                "ServerMetrics record failed (gemma4 assistant spec)", exc_info=True
            )
        return GenerationOutput(
            text=out_text,
            new_text=out_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(visible_ids),
            finished=True,
            finish_reason="stop" if finished_by_eos else "length",
            ttft_ms=ttft_ms,
            cached_tokens=0,
        )

    async def _generate_speculative(
        self,
        prompt: str | list[dict],
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
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        thinking_budget: int | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        json_schema: dict | str | None = None,
        cancel_event: asyncio.Event | None = None,
        logits_processors: list | None = None,
        timeout_seconds: float = 300.0,
        lora_adapter: str | None = None,
    ) -> GenerationOutput:
        """Generate using speculative decoding (single-request EAGLE-3 path).

        This path bypasses the continuous batching scheduler and runs the
        SpeculativeDecoder directly. Best for single-request scenarios where
        the draft model can propose K tokens for the target to verify.
        """
        if self._spec_decoder is None:
            # Fall back to standard generation if no decoder
            return await self._generate_fast(
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
                timeout_seconds=timeout_seconds,
                lora_adapter=lora_adapter,
            )

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        # Tokenize prompt — handle messages-format (list of dicts) like _generate_fast.
        # route through _apply_chat_template + _encode_prompt so this matches the
        # fast path AND applies the double-BOS guard (raw apply_chat_template+encode prepends
        # BOS twice for Gemma/Llama-3/Mistral, corrupting the first-token distribution).
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(self._tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = self._tokenizer.encode(text)
        import mlx.core as mx

        if seed is not None:
            mx.random.seed(seed)

        input_array = mx.array(input_ids).reshape(1, -1)

        # Build EOS + stop token sets
        eos_ids = set()
        if hasattr(self._tokenizer, "eos_token_id"):
            eid = self._tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
        if stop_token_ids:
            eos_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                try:
                    ids = self._tokenizer.encode(s)
                    if len(ids) == 1:
                        eos_ids.add(ids[0])
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        # Run speculative generation on the MLX executor thread
        # Use incremental detokenizer for correct multi-byte UTF-8
        detokenizer = self._tokenizer.detokenizer
        detokenizer.reset()

        # Wire grammar constraint into spec decoder for structured output.
        # route regex/choice/cfg correctly (was always JsonSchemaConstraint →
        # silently dropped non-JSON grammars).
        _spec_constraint = None
        if json_schema is not None:
            try:
                _spec_constraint = _build_grammar_constraint(
                    json_schema, self._tokenizer
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for spec decode", exc_info=True
                )
        _prev_constraint = self._spec_decoder.constraint
        self._spec_decoder.constraint = _spec_constraint

        # (LoRA concurrency keystone): the adapter lifecycle runs inside
        # _run_spec_with_lora on the executor thread (serialized with generation), not here
        # on the event loop. _lora_applied stays False so the legacy event-loop release
        # blocks below are inert no-ops.
        _lora_applied = False

        def _run_spec():
            token_ids = self._spec_decoder.generate(
                input_ids=input_array,
                max_tokens=max_tokens,
                temperature=temperature,
                cancel_event=cancel_event,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
            )
            # Apply stop token truncation (exclude stop token from output)
            for i, tid in enumerate(token_ids):
                if tid in eos_ids:
                    # Add tokens before the stop token to detokenizer
                    for j in range(i):
                        detokenizer.add_token(token_ids[j])
                    return token_ids[:i], True
            # No stop token found — add all tokens to detokenizer
            for tid in token_ids:
                detokenizer.add_token(tid)
            return token_ids, False

        def _run_spec_with_lora():
            # adapter acquire+apply / release+restore on the executor thread.
            _applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _applied:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            try:
                return _run_spec()
            finally:
                if _applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(
                            "LoRA release failed (spec executor)", exc_info=True
                        )

        _spec_gen_t0 = time.perf_counter()
        try:
            token_ids, hit_stop = await asyncio.wait_for(
                loop.run_in_executor(executor, _run_spec_with_lora),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            logger.warning(f"Speculative generation timed out after {timeout_seconds}s")
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                logger.debug(
                    "GPU cache cleanup failed after spec decode timeout", exc_info=True
                )
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after spec decode timeout", exc_info=True
                    )
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error=f"Speculative generation timed out after {timeout_seconds}s",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except MemoryError:
            logger.warning(
                "OOM during speculative generation — returning memory_limit finish reason"
            )
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                logger.debug(
                    "GPU cache cleanup failed after spec decode OOM", exc_info=True
                )
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after spec decode OOM", exc_info=True
                    )
            return GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error="OOM during speculative generation",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except RuntimeError as e:
            if "memory" in str(e).lower() or "out of" in str(e).lower():
                logger.warning(f"MLX OOM during speculative generation: {e}")
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed after spec decode OOM (RuntimeError)",
                        exc_info=True,
                    )
                self._spec_decoder.constraint = _prev_constraint
                if (
                    _lora_applied
                    and hasattr(self, "_lora_manager")
                    and self._lora_manager is not None
                ):
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.warning(
                            "LoRA release failed after spec decode OOM (RuntimeError)",
                            exc_info=True,
                        )
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=len(input_ids),
                    completion_tokens=0,
                    error=str(e),
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after spec decode RuntimeError",
                        exc_info=True,
                    )
            # Return error output for non-OOM RuntimeError instead of
            # propagating to caller (which expects GenerationOutput).
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error=f"RuntimeError during speculative generation: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during speculative generation: {e}", exc_info=True
            )
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after spec decode unexpected error",
                        exc_info=True,
                    )
            # Return error output instead of propagating exception to caller.
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error=f"Speculative generation failed: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        _spec_ttft_s = time.perf_counter() - _spec_gen_t0
        detokenizer.finalize()

        text = _clean_special_tokens(detokenizer.text)

        # Build logprobs from individual tokens — speculative decode black-box
        # generate() does not expose logits, so we cannot compute real logprobs.
        # Return None instead of fake 0.0 to avoid misleading consumers.
        _logprobs = None
        if logprobs and token_ids:
            _logprobs = None  # Real logprobs unavailable from spec decode black-box

        # Record TTFT in Prometheus for spec decode path
        if _spec_ttft_s > 0:
            try:
                from yunshu_gateway.middleware.prometheus_exporter import (
                    get_prometheus_metrics,
                )

                pm = get_prometheus_metrics()
                pm.observe_histogram(
                    "ttft_seconds", _spec_ttft_s, labels={"model_id": self.model_label}
                )
            except Exception:
                logger.debug(
                    "TTFT prometheus recording failed in spec decode path",
                    exc_info=True,
                )

        # Determine finish_reason with cancel awareness
        _cancelled = _is_cancelled(cancel_event)
        if _cancelled or hit_stop or len(token_ids) < max_tokens:
            _finish_reason = "stop"
        else:
            _finish_reason = "length"

        # Reasoning parser: extract thinking tokens from spec decode output.
        # Speculative decode uses a black-box generate() that does not track
        # thinking tokens, so we parse the output text for reasoning content.
        _spec_reasoning_tok = 0
        if text:
            try:
                from .reasoning_parser import get_reasoning_parser

                rp = get_reasoning_parser(self.model_name)
                rp_out = rp.parse(text)
                if rp_out.reasoning and rp_out.reasoning_tokens > 0:
                    _spec_reasoning_tok = rp_out.reasoning_tokens
                    if rp_out.content != text:
                        text = rp_out.content
            except Exception:
                logger.debug(
                    "reasoning_parser failed in spec decode path", exc_info=True
                )

        self._spec_decoder.constraint = _prev_constraint
        if (
            _lora_applied
            and hasattr(self, "_lora_manager")
            and self._lora_manager is not None
        ):
            try:
                self._lora_manager.release_adapter(lora_adapter)
            except Exception:
                logger.warning(
                    "LoRA release failed after spec decode normal completion",
                    exc_info=True,
                )
        return GenerationOutput(
            text=text,
            new_text=text,
            prompt_tokens=len(input_ids),
            completion_tokens=len(token_ids),
            finished=True,
            finish_reason=_finish_reason,
            reasoning_tokens=_spec_reasoning_tok,
            cached_tokens=0,
            logprobs=_logprobs,
            ttft_ms=round(_spec_ttft_s * 1000, 1),
        )

    async def _stream_generate_speculative(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        json_schema: dict | str | None = None,
        cancel_event: asyncio.Event | None = None,
        logits_processors: list | None = None,
        lora_adapter: str | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using speculative decoding (single-request path).

        Yields chunks as they are verified by the target model.
        Each yield contains the accepted tokens from one verify step.

        UNREACHABLE in the streaming flow (honesty annotation, mirrors
        _stream_generate_mtp): stream_generate() forces ``spec_decode = False`` at
        the top (batched_engine.py ~4381 — fast-path spec routes are not lossless and give
        no speedup on Apple Silicon), so the only caller (the `if spec_decode and ...`
        block ~4456) never fires. The documented "spec-stream multi-token stop-prefix
        leak" therefore is NOT a live bug — it lives only here in dead code. The LIVE
        streaming path is _stream_generate_fast, which routes deltas through
        StopHoldbackBuffer so no stop prefix leaks. Retrofitting hold-back into
        this complex verify-loop would be risky churn on an unreachable path; if spec
        streaming is ever re-enabled, wrap the per-step emit through StopHoldbackBuffer
        (the _stream_generate_fast _hb pattern) BEFORE shipping.
        """
        if self._spec_decoder is None:
            # Fall back to fast path streaming (avoid recursive dispatch)
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
                cancel_event=cancel_event,
                logprobs=bool(logprobs),
                top_logprobs=top_logprobs,
                logits_processors=logits_processors,
                lora_adapter=lora_adapter,
            ):
                yield output
            return

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        # Tokenize prompt — handle messages-format (list of dicts) like _generate_fast.
        # route through _apply_chat_template + _encode_prompt so this matches the
        # fast path AND applies the double-BOS guard (raw apply_chat_template+encode prepends
        # BOS twice for Gemma/Llama-3/Mistral, corrupting the first-token distribution).
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(self._tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = self._tokenizer.encode(text)
        import mlx.core as mx

        if seed is not None:
            mx.random.seed(seed)

        input_array = mx.array(input_ids).reshape(1, -1)

        # Get EOS IDs + stop tokens
        eos_ids = set()
        stop_suffixes = []
        if hasattr(self._tokenizer, "eos_token_id"):
            eid = self._tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
        if stop_token_ids:
            eos_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                if not s:
                    continue
                try:
                    ids = self._tokenizer.encode(s)
                    if len(ids) == 1:
                        eos_ids.add(ids[0])
                    # : always string-match (bare-stop token id
                    # rarely matches the space-prefixed emitted token).
                    stop_suffixes.append(s)
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        _eids = getattr(self._tokenizer, "eos_token_ids", None)
        if _eids is not None:  # may be a bare int (Qwen3.6-27B), not iterable
            eos_ids.update(_eids if isinstance(_eids, (list, tuple, set)) else (_eids,))

        # Run speculative steps on executor thread, yielding after each step
        from mlx_lm.models.cache import make_prompt_cache

        target_cache = make_prompt_cache(self._spec_decoder.target)
        draft_cache = make_prompt_cache(self._spec_decoder.draft)

        # Wire grammar constraint into spec decoder for streaming structured output.
        # Save the previous constraint to restore after this request completes,
        # preventing cross-request constraint mixing on the shared decoder instance.
        # route regex/choice/cfg correctly (was always JsonSchemaConstraint →
        # silently dropped non-JSON grammars).
        _spec_constraint = None
        if json_schema is not None:
            try:
                _spec_constraint = _build_grammar_constraint(
                    json_schema, self._tokenizer
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for spec streaming", exc_info=True
                )
        _prev_constraint = self._spec_decoder.constraint
        self._spec_decoder.constraint = _spec_constraint

        # LoRA acquire+apply / release MUST happen on the executor thread (inside
        # _prefill / the release closure below), never here on the event loop:
        # acquire_adapter loads+activates the adapter by MUTATING the shared model,
        # and doing that off the executor races a concurrent request's generate_step
        # (wrong-adapter bleed). _lora_state carries the applied flag back so the
        # finally can release on the executor. (Same keystone as _generate_fast.)
        _lora_state = {"applied": False}

        # Thinking budget enforcement — detect <think/</think via single-token IDs
        _spec_think_start_token = None
        _spec_think_end_token = None
        if thinking_budget is not None or enable_thinking:
            # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
            _spec_think_start_token, _spec_think_end_token = _resolve_think_token_ids(
                self._tokenizer
            )
        _spec_in_thinking = False
        _spec_thinking_tokens_used = 0

        # Prefill both models
        def _prefill():
            # Acquire+apply the LoRA adapter HERE on the executor thread, before any
            # forward pass, serialized with every other request's generate_step.
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _lora_state["applied"] = self._lora_manager.acquire_adapter(
                        lora_adapter
                    )
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _lora_state["applied"]:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            self._spec_decoder.target(input_array, cache=target_cache)
            self._spec_decoder.draft(input_array, cache=draft_cache)

            # Roll back both caches by 1 position so the last prompt token is
            # NOT in the cache. This prevents double-feeding:
            # - generate_draft will feed current_ids (last prompt token) into
            # the draft cache — correct since it was rolled back.
            # - verify_draft will feed [current_ids, d0, ...] into the target
            # cache — correct since current_ids was rolled back.
            # Without this rollback, both methods would double-feed the last
            # prompt token (it's already in both caches from prefill).
            try:
                from mlx_lm.models.cache import trim_prompt_cache

                trim_prompt_cache(target_cache, 1)
                trim_prompt_cache(draft_cache, 1)
            except Exception:
                logger.debug(
                    "trim_prompt_cache failed, falling back to per-layer trim",
                    exc_info=True,
                )
                for c in target_cache:
                    if hasattr(c, "trim"):
                        c.trim(1)
                for c in draft_cache:
                    if hasattr(c, "trim"):
                        c.trim(1)

        # After prefill + rollback, both caches have prompt[:-1].
        # current_ids is the last prompt token, which will be fed into both
        # caches by generate_draft and verify_draft respectively — no double-feed.
        current_ids = input_array[:, -1:]  # [1, 1] last prompt token

        generated_tokens: list[int] = []
        prompt_tokens = len(input_ids)
        detokenizer = self._tokenizer.detokenizer
        detokenizer.reset()

        try:
            await loop.run_in_executor(executor, _prefill)
        except Exception as e:
            logger.error(f"Spec decode streaming prefill failed: {e}", exc_info=True)
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_state["applied"]
                and getattr(self, "_lora_manager", None) is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after spec decode prefill error",
                        exc_info=True,
                    )
            raise
        _spec_gen_t0 = time.perf_counter()  # TTFT timing starts after prefill
        try:
            _spec_ttft_ms_val = 0.0
            _spec_ttft_recorded = False
            while len(generated_tokens) < max_tokens:
                if _is_cancelled(cancel_event):
                    logger.debug("Cancel event triggered during spec decode streaming")
                    # Yield terminal stop chunk so consumer sees finished=True
                    if generated_tokens:
                        detokenizer.finalize()
                        _final_text = _clean_special_tokens(detokenizer.text)
                    else:
                        _final_text = ""
                    yield GenerationOutput(
                        text=_final_text,
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=len(generated_tokens),
                        finished=True,
                        finish_reason="stop",
                        ttft_ms=_spec_ttft_ms_val,
                        reasoning_tokens=0,
                        cached_tokens=0,
                        logprobs=None,
                    )
                    break

                # Snapshot draft cache for rollback on rejection
                draft_snap = self._spec_decoder._snapshot_cache(draft_cache)

                # Cap draft length to remaining thinking budget
                _effective_spec_K = None
                if thinking_budget is not None and _spec_in_thinking:
                    remaining = thinking_budget - _spec_thinking_tokens_used
                    if remaining <= 0:
                        if _spec_think_end_token is not None:
                            _spec_in_thinking = False
                            generated_tokens.append(_spec_think_end_token)
                            detokenizer.add_token(_spec_think_end_token)
                        break
                    _effective_spec_K = max(1, remaining - 1)

                _spec_target_trimmed = (
                    False  # Set by _spec_step if SP-PEN trims target cache
                )

                def _spec_step():
                    # First iteration: last prompt token (already in cache from prefill,
                    # so forward pass advances the cache by 1 and returns logits).
                    # Subsequent iterations: last accepted/bonus token.
                    draft_result = self._spec_decoder.generate_draft(
                        current_ids, draft_cache, max_draft_tokens=_effective_spec_K
                    )
                    # verify_draft: includes current_ids as alignment token so logits
                    # are correctly positioned. Returns accepted tokens + bonus.
                    # NOTE: verify_draft feeds [current_ids, draft_tokens] to target,
                    # populating target_cache with K+1 tokens. We must NOT re-feed
                    # accepted tokens to target (that would double-populate the cache).
                    verify_result = self._spec_decoder.verify_draft(
                        draft_result,
                        current_ids,
                        target_cache,
                    )

                    # SP-PEN: Apply penalty/bias to bonus token logits.
                    # After verify_draft, target_cache has K+1 entries but only the
                    # first (accepted_count + 1) are valid. Trim the rejected entries
                    # so the cache is consistent, then do a single forward pass for
                    # the bonus position to get fresh logits with penalties applied.
                    _has_penalties = (
                        repetition_penalty != 1.0
                        or frequency_penalty != 0.0
                        or presence_penalty != 0.0
                        or (logit_bias is not None and len(logit_bias) > 0)
                    )
                    if _has_penalties:
                        import mlx.core as _sp_mx

                        K_local = len(draft_result.token_ids)
                        ac = verify_result.accepted_count
                        _already_trimmed = K_local > 0 and ac < K_local
                        if _already_trimmed:
                            from mlx_lm.models.cache import trim_prompt_cache

                            _trim_n = K_local - ac
                            try:
                                trim_prompt_cache(target_cache, _trim_n)
                            except Exception:
                                logger.debug(
                                    "trim_prompt_cache (SP-PEN partial) failed, falling back",
                                    exc_info=True,
                                )
                                for _c in target_cache:
                                    if hasattr(_c, "trim"):
                                        _c.trim(_trim_n)
                            nonlocal _spec_target_trimmed
                            _spec_target_trimmed = True
                        # Feed last accepted token (or current_ids if none accepted) to get bonus logits
                        _bonus_input = (
                            mx.array([[verify_result.accepted_ids[-1]]])
                            if verify_result.accepted_ids
                            else current_ids
                        )
                        _bonus_out = self._spec_decoder.target(
                            _bonus_input, cache=target_cache
                        )
                        _bonus_logits = (
                            _bonus_out.logits
                            if hasattr(_bonus_out, "logits")
                            else _bonus_out
                        )
                        _bonus_logits = _bonus_logits[0, -1, :]
                        # Undo the bonus forward — next iteration's verify_draft will feed
                        # current_ids (= bonus) into target_cache, and we must not have
                        # the SP-PEN probe already in the cache or it double-populates.
                        try:
                            from mlx_lm.models.cache import trim_prompt_cache

                            trim_prompt_cache(target_cache, 1)
                        except Exception:
                            logger.debug(
                                "trim_prompt_cache (SP-PEN bonus undo) failed, falling back",
                                exc_info=True,
                            )
                            for _c in target_cache:
                                if hasattr(_c, "trim"):
                                    _c.trim(1)
                        # Build token history: prompt + all generated so far + accepted drafts
                        _token_hist = (
                            list(input_ids)
                            + generated_tokens
                            + verify_result.accepted_ids
                        )
                        _bonus_logits = _apply_spec_bonus_penalties(
                            _bonus_logits,
                            _token_hist,
                            len(input_ids),
                            repetition_penalty=repetition_penalty,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            logit_bias=logit_bias,
                        )
                        # Re-sample bonus token from penalized logits using configured sampler
                        from mlx_lm.sample_utils import make_sampler as _make_sp_sampler

                        # mlx-lm make_sampler's first param is `temp`,
                        # not `temperature` — the old kwarg raised TypeError and failed the
                        # whole request (no fallback at the _spec_step call site) whenever
                        # cross-model spec decode ran with any penalty/bias set.
                        _sp_sampler = _make_sp_sampler(
                            temp=temperature,
                            top_p=top_p,
                            top_k=top_k if top_k and top_k > 0 else -1,
                            min_p=min_p if min_p and min_p > 0 else 0.0,
                        )
                        _bonus_id = int(
                            _sp_sampler(
                                _sp_mx.expand_dims(_bonus_logits, axis=(0, 1))
                            ).item()
                        )
                        # Overwrite bonus token in verify_result (simple reconstruction)
                        verify_result = type(verify_result)(
                            accepted_count=verify_result.accepted_count,
                            accepted_ids=verify_result.accepted_ids,
                            rejected_at=verify_result.rejected_at,
                            bonus_token_id=_bonus_id,
                            target_logprobs=verify_result.target_logprobs,
                        )

                    return draft_result, verify_result

                draft_result, verify_result = await loop.run_in_executor(
                    executor, _spec_step
                )

                K = len(draft_result.token_ids)
                accepted_count = verify_result.accepted_count
                new_tokens = verify_result.accepted_ids[:]
                if (
                    verify_result.bonus_token_id is not None
                    and verify_result.bonus_token_id >= 0
                ):
                    new_tokens.append(verify_result.bonus_token_id)

                self._spec_decoder._stats["total_draft_tokens"] += K
                self._spec_decoder._stats["total_accepted_tokens"] += accepted_count
                self._spec_decoder._stats["total_bonus_tokens"] += 1
                self._spec_decoder._stats["total_steps"] += 1
                if self._lookahead_reasoning is not None:
                    self._lookahead_reasoning.record_accept(accepted_count)

                hit_eos = False
                _hit_suffix = False
                _yielded_token_count = (
                    0  # Track how many tokens were actually yielded before break
                )
                for token_id in new_tokens:
                    if token_id in eos_ids:
                        hit_eos = True
                        break
                    # Thinking budget tracking — detect <think/</think via token IDs
                    if _spec_think_start_token is not None:
                        if (
                            not _spec_in_thinking
                            and token_id == _spec_think_start_token
                        ):
                            _spec_in_thinking = True
                        elif _spec_in_thinking:
                            _spec_thinking_tokens_used += 1
                            if token_id == _spec_think_end_token:
                                _spec_in_thinking = False
                    generated_tokens.append(token_id)
                    detokenizer.add_token(token_id)
                    _yielded_token_count += 1
                    # Thinking budget enforcement: force </think when budget exceeded
                    if (
                        thinking_budget is not None
                        and _spec_in_thinking
                        and _spec_thinking_tokens_used >= thinking_budget
                        and _spec_think_end_token is not None
                    ):
                        _spec_in_thinking = False
                        # Force-insert </think token
                        generated_tokens.append(_spec_think_end_token)
                        detokenizer.add_token(_spec_think_end_token)
                        _yielded_token_count += 1
                        # Treat as stop — no more tokens after budget hit
                        hit_eos = True
                        break
                    if stop_suffixes and any(
                        detokenizer.text.endswith(s) for s in stop_suffixes
                    ):
                        _hit_suffix = True
                        # Remove the suffix-triggering token — it should not
                        # appear in the output, matching the non-spec pattern.
                        generated_tokens.pop()
                        _yielded_token_count -= 1
                        # Reset detokenizer to state before the suffix token was
                        # added. Simply popping from detokenizer.tokens is not
                        # sufficient because NaiveStreamingDetokenizer computes
                        # .text from _current_tokens (an internal list), not from
                        # the .tokens attribute. Re-decode all remaining tokens
                        # to produce clean text without the suffix.
                        _kept_tokens = (
                            list(detokenizer.tokens[:-1]) if detokenizer.tokens else []
                        )
                        detokenizer.reset()
                        for _t in _kept_tokens:
                            detokenizer.add_token(_t)
                        break

                # Compute TTFT before first yield
                if not _spec_ttft_recorded:
                    _spec_ttft_recorded = True
                    _spec_ttft_s = time.perf_counter() - _spec_gen_t0
                    _spec_ttft_ms_val = round(_spec_ttft_s * 1000, 1)
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _spec_ttft_s,
                            labels={"model_id": self.model_label},
                        )
                    except Exception:
                        logger.debug(
                            "spec streaming TTFT prometheus recording failed",
                            exc_info=True,
                        )

                # Yield accepted text via incremental detokenizer
                chunk_text = _clean_special_tokens(detokenizer.last_segment)
                finish_reason = None
                if hit_eos or _hit_suffix:
                    finish_reason = "stop"
                elif len(generated_tokens) >= max_tokens:
                    finish_reason = "length"

                # Trim stop suffix from chunk text when matched
                if _hit_suffix and stop_suffixes:
                    for s in stop_suffixes:
                        if chunk_text.endswith(s):
                            chunk_text = chunk_text[: -len(s)]
                            break

                # Build logprobs from target model verification
                # Only include logprobs for tokens that were actually yielded
                # (before hit_eos or _hit_suffix broke the loop). Without this
                # guard, logprobs included EOS and post-break tokens that were
                # never added to generated_tokens / detokenizer.
                _chunk_logprobs = None
                if logprobs and new_tokens:
                    _chunk_logprobs = []
                    _lp_count = min(_yielded_token_count, len(new_tokens))
                    for i in range(_lp_count):
                        tid = new_tokens[i]
                        tok_text = _clean_special_tokens(self._tokenizer.decode([tid]))
                        lp = (
                            verify_result.target_logprobs[i]
                            if i < len(verify_result.target_logprobs)
                            else 0.0
                        )
                        _chunk_logprobs.append(
                            {
                                "token": tok_text,
                                "logprob": float(lp),
                                "top_logprobs": [
                                    {"token": tok_text, "logprob": float(lp)}
                                ],
                            }
                        )

                yield GenerationOutput(
                    text=_clean_special_tokens(detokenizer.text),
                    new_text=chunk_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(generated_tokens),
                    finished=finish_reason is not None,
                    finish_reason=finish_reason,
                    reasoning_tokens=_spec_thinking_tokens_used
                    if _spec_think_start_token is not None
                    else 0,
                    cached_tokens=0,
                    logprobs=_chunk_logprobs,
                    ttft_ms=_spec_ttft_ms_val,
                )

                if finish_reason is not None:
                    detokenizer.finalize()
                    break

                # Update caches for next iteration:
                # - Target cache: verify_draft already fed [last_tok, d0..dK-1].
                # If all accepted, target has exactly the right state (last_tok + K drafts).
                # If partially accepted, we need to trim rejected tokens from target cache.
                # - Draft cache: if all accepted, draft already has K tokens from generate_draft.
                # If partially accepted, restore snapshot and refeed accepted + bonus.
                def _update_caches():
                    nonlocal draft_snap

                    if accepted_count < K:
                        if not _spec_target_trimmed:
                            # Partial acceptance: trim target cache to remove rejected entries.
                            # verify_draft fed K+1 tokens (last_tok + K drafts).
                            # We want to keep: last_tok + accepted_count drafts = accepted_count + 1
                            # Trim: (K+1) - (accepted_count + 1) = K - accepted_count entries.
                            from mlx_lm.models.cache import trim_prompt_cache

                            trim_count = K - accepted_count
                            try:
                                trim_prompt_cache(target_cache, trim_count)
                            except Exception:
                                logger.debug(
                                    "trim_prompt_cache (partial acceptance) failed, falling back",
                                    exc_info=True,
                                )
                                for c in target_cache:
                                    if hasattr(c, "trim"):
                                        c.trim(trim_count)

                        # Restore draft cache to pre-draft state and refeed accepted tokens.
                        # This MUST run even when SP-PEN already trimmed the target cache,
                        # otherwise the draft cache accumulates incorrect state from rejected
                        # tokens, corrupting subsequent draft generation.
                        # Do NOT refeed the bonus token here — generate_draft will feed
                        # current_ids (= bonus token) at the start of the next iteration,
                        # so including it now would cause a double-feed.
                        self._spec_decoder._restore_cache(draft_cache, draft_snap)
                        for tok in verify_result.accepted_ids:
                            self._spec_decoder.draft(
                                mx.array([[tok]]), cache=draft_cache
                            )

                    # Feed bonus token to both caches (target already has it from verify_draft
                    # when all accepted; when partial, we trimmed and need to re-add).
                    # For draft: bonus token needs to be fed in both cases.
                    # For target: when all accepted, bonus is the last token from verify_draft
                    # logits[K] — already in cache. When partial, we trimmed and re-added
                    # accepted+bonus above, so target is current.

                    # Update current_ids for next iteration
                    return mx.array([[verify_result.bonus_token_id]])

                current_ids = await loop.run_in_executor(executor, _update_caches)
        except GeneratorExit:
            logger.debug("Client disconnected during spec decode streaming")
        except Exception as e:
            logger.error(f"Spec decode streaming error: {e}", exc_info=True)
            # Yield a terminal error output so the consumer sees finished=True
            # instead of a broken stream (exception without terminal output).
            try:
                detokenizer.finalize()
                _err_final_text = _clean_special_tokens(detokenizer.text)
            except Exception:
                _err_final_text = ""
            yield GenerationOutput(
                text=_err_final_text,
                new_text="",
                prompt_tokens=prompt_tokens,
                completion_tokens=len(generated_tokens),
                finished=True,
                finish_reason="error",
                ttft_ms=_spec_ttft_ms_val,
                reasoning_tokens=0,
                cached_tokens=0,
                logprobs=None,
            )
            raise
        finally:
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_state["applied"]
                and getattr(self, "_lora_manager", None) is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed in spec decode streaming finally",
                        exc_info=True,
                    )

    def _new_request_proposer(self):
        """Fresh per-request spec proposer mirroring the configured global one.

        A per-request instance avoids reset()/propose() races on a shared object
        under concurrency. Branches on the proposer family (n-gram vs suffix)
        selected in _init_spec_decode; both expose the same propose(token_ids)
        contract and feed the same lossless verifier.
        """
        from .suffix_proposer import SuffixProposer

        g = self._ngram_proposer
        if isinstance(g, SuffixProposer):
            from .suffix_proposer import SuffixConfig

            c = g.config
            return SuffixProposer(
                SuffixConfig(
                    min_suffix_length=c.min_suffix_length,
                    max_window=c.max_window,
                    max_draft=c.max_draft,
                    max_model_len=c.max_model_len,
                    max_trie_depth=c.max_trie_depth,
                )
            )
        from .ngram_proposer import NgramConfig, NgramProposer

        return NgramProposer(
            NgramConfig(max_n=g.config.max_n, k=g.config.k, mode=g.config.mode)
        )

    async def _generate_ngram_spec(
        self,
        prompt: str | list[dict],
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
        logprobs: bool = False,
        top_logprobs: int | None = None,
        json_schema: dict | str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        cancel_event: asyncio.Event | None = None,
        logits_processors: list | None = None,
        timeout_seconds: float = 300.0,
        lora_adapter: str | None = None,
    ) -> GenerationOutput:
        """Generate using N-gram speculative decoding (model-free).

        Uses the N-gram proposer to predict K draft tokens, then verifies
        each by running the target model forward. Accepts matching tokens,
        resamples on mismatch.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        from .mlx_executor import get_mlx_executor

        tokenizer = self._tokenizer
        model = self._model

        # SAFETY GUARD: speculative verification rolls the KV cache back
        # to discard rejected drafts, which REQUIRES a trimmable cache. Qwen3.5's
        # hybrid attention uses 18 non-trimmable ArraysCache (linear/gated-attention
        # recurrent state) + 6 KVCache, so can_trim is False — spec there can't roll
        # back rejected drafts (degenerate ".txt.txt..." repetition), and the
        # recurrent state must be driven by ONE continuous generate_step, so we
        # delegate the WHOLE request to the plain fast path. mlx-lm's
        # speculative_generate_step likewise RAISES on a non-trimmable cache.
        try:
            from mlx_lm.models.cache import (
                can_trim_prompt_cache as _can_trim,
            )
            from mlx_lm.models.cache import (
                make_prompt_cache as _mk_cache,
            )

            _spec_cache_supported = _can_trim(_mk_cache(model))
        except Exception:
            _spec_cache_supported = False
        if not _spec_cache_supported:
            logger.info(
                "N-gram spec disabled: model cache is not trimmable — "
                "using the plain fast path."
            )
            return await self._generate_fast(
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
                timeout_seconds=timeout_seconds,
                lora_adapter=lora_adapter,
            )
        # Fresh per-request proposer (n-gram or suffix) — avoids reset()/propose()
        # races on a shared instance under concurrency.
        proposer = self._new_request_proposer()

        # Handle messages-format prompts (list of dicts) — apply chat template.
        # route through _apply_chat_template + _encode_prompt (NOT raw tokenizer
        # calls) so this spec path matches the default fast path — role remap, family
        # adapters, AND the double-BOS guard. The raw apply_chat_template(tokenize=False)
        # emits a string already opening with the literal bos_token; a following
        # tokenizer.encode (add_special_tokens=True) prepended BOS a SECOND time for
        # Gemma/Llama-3/Mistral → [BOS, BOS, …], corrupting the first-token distribution.
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        # Build stop token sets
        stop_ids: set[int] = set()
        stop_suffixes = []
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                stop_ids.update(eid)
            elif eid is not None:
                stop_ids.add(eid)
        _eids = getattr(tokenizer, "eos_token_ids", None)
        if _eids is not None:  # may be a bare int (Qwen3.6-27B), not iterable
            stop_ids.update(
                _eids if isinstance(_eids, (list, tuple, set)) else (_eids,)
            )
        # Model (generation_)config multi-eos — e.g. Gemma-4 turn-end 106 the
        # tokenizer omits, so the model never stops and rambles.
        stop_ids.update(_read_config_eos_ids(self.model_name))
        if stop_token_ids:
            stop_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                try:
                    ids = tokenizer.encode(s)
                    if len(ids) == 1:
                        stop_ids.add(ids[0])
                    elif len(ids) > 1:
                        stop_suffixes.append(s)
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        # route temp>0 off mlx-lm's PRNG-trapped make_sampler (its
        # categorical_sampling @mx.compile cache traps the global PRNG state, so
        # sequential/concurrent temp>0 spec requests collapse + seed is a no-op).
        if temperature is not None and temperature > 1e-6:
            sampler = _build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                seed=seed,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        else:
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )

        # Grammar constraint: pre-validate draft tokens against allowed set
        _grammar_constraint = None
        if json_schema is not None:
            try:
                sampler = _build_constrained_sampler(sampler, json_schema, tokenizer)
                _grammar_constraint = (
                    sampler.constraint if hasattr(sampler, "constraint") else None
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for n-gram spec", exc_info=True
                )

        def _grammar_filter_drafts(
            draft_ids: list[int],
            generated_ids: list[int],
        ) -> list[int]:
            """Filter draft tokens that violate grammar constraints.

            Returns the longest prefix of draft_ids where every token is
            in the grammar's allowed set at its position. This avoids
            wasting a batched forward pass on drafts that can't be accepted.
            """
            if _grammar_constraint is None:
                return draft_ids
            # rollback() is no-arg ONLY for JsonSchemaConstraint; Regex/
            # Choice/Cfg constraints' rollback(saved) REQUIRES the dict checkpoint()
            # returns, so the bare rollback() here raised TypeError for a {"type":
            # "regex"|"choice"|"cfg"} grammar on the n-gram spec path. Capture the
            # checkpoint and roll back signature-robustly.
            _saved = _grammar_constraint.checkpoint()

            def _gc_rollback():
                try:
                    _grammar_constraint.rollback(_saved)
                except TypeError:
                    try:
                        _grammar_constraint.rollback()
                    except Exception:
                        logger.debug(
                            "grammar rollback failed in n-gram filter", exc_info=True
                        )
                except Exception:
                    logger.debug(
                        "grammar rollback failed in n-gram filter", exc_info=True
                    )

            allowed = _grammar_constraint.get_allowed_tokens(tokenizer, generated_ids)
            if not allowed:
                _gc_rollback()
                return draft_ids
            allowed_set = set(allowed)
            filtered = []
            for tid in draft_ids:
                if tid in allowed_set:
                    filtered.append(tid)
                    try:
                        tok_text = tokenizer.decode([tid])
                        _grammar_constraint.advance(tok_text)
                    except Exception:
                        logger.debug("grammar constraint advance failed", exc_info=True)
                        break
                    allowed = _grammar_constraint.get_allowed_tokens(
                        tokenizer, generated_ids + filtered
                    )
                    if allowed:
                        allowed_set = set(allowed)
                    else:
                        break
                else:
                    break
            _gc_rollback()
            return filtered

        # Inflight prefix sharing: defined before _run so cleanup is accessible
        # in exception handlers. Use timestamp instead of id(_run) since _run
        # is not yet defined at this point.
        _ng_inflight_req_id = f"ng-{int(time.monotonic() * 1e6)}"

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_ng_inflight_req_id)
            except Exception:
                logger.debug(
                    "inflight prefix unregister failed in n-gram spec", exc_info=True
                )

        def _run():
            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            tokens = []
            ttft_s = 0.0
            _stopped_by_stop_id = False
            _stopped_by_suffix = False

            # Prefill with KV prefix cache
            prefix_cache = self._kv_prefix_cache
            # bypass the KV prefix cache when a LoRA adapter is active — it is keyed
            # on token ids + model name but NOT the adapter, so reusing/storing KV here would
            # decode on a different adapter's KV (the keystone, un-propagated to this
            # n-gram spec path; the two live fast paths bypass it at 3571 / 5230). Setting it
            # to None disables BOTH the get and the post-generation add below. Latent today
            # (a LoRA request fails _gemma4_spec_eligible → spec_decode forced off → this path
            # isn't entered with an adapter), but fail-safe if that gate ever changes.
            if lora_adapter is not None:
                prefix_cache = None
            if prefix_cache is not None:
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
            # Also evict from paged KV manager when enabled
            if self._kv_manager is not None:
                try:
                    self._kv_manager.memory_pressure_evict(
                        self._mem_pressure_threshold / 100.0
                    )
                except Exception:
                    logger.debug("paged KV pressure eviction failed", exc_info=True)
            try:
                cached_kv, _, matched = (
                    prefix_cache.get(ids)
                    if prefix_cache is not None
                    else (None, None, 0)
                )
            except Exception:
                logger.warning(
                    "KV prefix cache get failed in spec path — falling back to full prefill",
                    exc_info=True,
                )
                cached_kv, _, matched = None, None, 0
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids
            # generate_step requires at least one prompt token to start
            # decoding. If the prefix cache returned a full match, re-feed
            # the last cached token so we have a starting input.
            if len(ids_to_prefill) == 0 and len(ids) > 0:
                ids_to_prefill = ids[-1:]

            # Inflight prefix sharing: register for concurrent KV block sharing
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().register(
                    _ng_inflight_req_id,
                    [int(t) for t in ids],
                    cache,
                    self.model_name or "",
                )
            except Exception:
                logger.debug(
                    "inflight prefix register failed in n-gram spec", exc_info=True
                )

            try:
                gen_t0 = time.perf_counter()
                _timeout_deadline = gen_t0 + timeout_seconds
                _timeout_check_interval = 32
                detokenizer = tokenizer.detokenizer
                detokenizer.reset()
                all_token_ids = list(
                    input_ids
                )  # Track full history for N-gram matching

                with _wired_limit_ctx(model):
                    # Step 1: Prefill via a DIRECT model() forward — NOT
                    # generate_step. generate_step's prefill-break + per-token-feed
                    # pattern corrupts the KV cache when interleaved with the verify
                    # path's direct model() calls: the cache offset skews and tokens
                    # get dropped or duplicated (verified — "2, 4, 6"→"246", "the
                    # average speed"→"the average average"). Driving the WHOLE path
                    # with direct model() forwards (consistent with
                    # verify_with_last_token) is lossless. The last produced token
                    # stays OUT of the cache (it is the next token to feed), matching
                    # the verify invariant.
                    _pf_logits = model(ids_to_prefill[None], cache=cache)
                    if hasattr(_pf_logits, "logits"):
                        _pf_logits = _pf_logits.logits

                    ttft_s = time.perf_counter() - gen_t0

                    # First token via the SAME sampler (honors temperature/top_p/
                    # top_k/grammar), applied to the last prefill position.
                    first_token = int(
                        sampler(_pf_logits[:, -1, :]).reshape(-1)[0].item()
                    )
                    tokens.append(first_token)
                    all_token_ids.append(first_token)

                    # align with mlx-lm's speculative_generate_step — the
                    # last produced token stays OUT of the cache (it is the next
                    # token to feed). generate_step already yields first_token
                    # WITHOUT adding its KV, which is exactly the invariant
                    # verify_with_last_token now expects, so we do NOT feed it here.

                    # Step 2: Decode loop with N-gram lookahead
                    remaining = max_tokens - 1
                    while remaining > 0:
                        if _is_cancelled(cancel_event):
                            break
                        # Stop check: break outer loop if stop token was hit in
                        # a previous iteration's accepted/bonus tokens.
                        if _stopped_by_stop_id or _stopped_by_suffix:
                            break
                        # Request-level timeout: check every N tokens to bound
                        # generation time. Without this, a pathological N-gram
                        # proposal/accept cycle can loop indefinitely.
                        if len(tokens) % _timeout_check_interval == 0:
                            if time.perf_counter() > _timeout_deadline:
                                logger.warning(
                                    f"N-gram spec generation timed out after "
                                    f"{timeout_seconds}s ({len(tokens)} tokens)"
                                )
                                break
                        # Propose K draft tokens via N-gram. When the adaptive
                        # controller is active it sets K — and K==0 means "back off to
                        # plain decode this step" (→ n_draft==0 path below, zero spec
                        # overhead). MUST special-case 0: `_k or len(...)` would wrongly
                        # fall through to a full-length draft on the backoff.
                        if self._adaptive_spec is not None:
                            _adaptive_k = self._adaptive_spec.get_draft_length()
                            draft_ids = (
                                proposer.propose(all_token_ids)[:_adaptive_k]
                                if _adaptive_k > 0
                                else []
                            )
                        else:
                            draft_ids = proposer.propose(all_token_ids)[
                                : len(all_token_ids)
                            ]
                        # Grammar-aware draft filtering: reject drafts that violate constraints
                        draft_ids = _grammar_filter_drafts(draft_ids, all_token_ids)
                        n_draft = min(len(draft_ids), remaining)

                        if n_draft == 0:
                            # No N-gram proposal — generate one token via a DIRECT
                            # model() forward (NOT generate_step — see the prefill
                            # note). tokens[-1] is NOT in the cache (the verify
                            # invariant), so this forward feeds it and predicts the
                            # next token, leaving the result out of the cache.
                            _bl = model(mx.array([tokens[-1]])[None], cache=cache)
                            if hasattr(_bl, "logits"):
                                _bl = _bl.logits
                            token_id = int(sampler(_bl[:, -1, :]).reshape(-1)[0].item())
                            tokens.append(token_id)
                            all_token_ids.append(token_id)
                            remaining -= 1
                            if token_id in stop_ids:
                                tokens.pop()
                                _stopped_by_stop_id = True
                            else:
                                detokenizer.add_token(token_id)
                                if stop_suffixes and any(
                                    detokenizer.text.endswith(s) for s in stop_suffixes
                                ):
                                    tokens.pop()  # Exclude suffix-triggering token
                                    _stopped_by_suffix = True
                            continue

                        # C10: Batch verify all K draft tokens via SpecDraftVerifier
                        self._ngram_stats["proposals"] += 1
                        self._ngram_stats["total_draft"] += n_draft

                        # Use SpecDraftVerifier for proper batch verification:
                        # 1. One forward pass populates KV cache for all K positions
                        # 2. Consecutive prefix match finds acceptance boundary
                        # 3. KV cache trimmed to remove rejected entries
                        # 4. Bonus token emitted from rejection point
                        result = self._spec_draft_verifier.verify_with_last_token(
                            model=model,
                            last_token_id=tokens[-1],
                            draft_ids=draft_ids[:n_draft],
                            prompt_cache=cache,
                            sampler=sampler,
                        )

                        # Emit accepted tokens
                        _stopped = False
                        for tid in result.accepted_tokens:
                            tokens.append(tid)
                            all_token_ids.append(tid)
                            remaining -= 1
                            if tid in stop_ids:
                                tokens.pop()
                                _stopped = True
                                _stopped_by_stop_id = True
                                break
                            detokenizer.add_token(tid)
                            if stop_suffixes and any(
                                detokenizer.text.endswith(s) for s in stop_suffixes
                            ):
                                tokens.pop()  # Exclude suffix-triggering token from count
                                _stopped = True
                                _stopped_by_suffix = True
                                break
                            # Advance sampler's grammar constraint to stay in sync
                            if _grammar_constraint is not None:
                                try:
                                    tok_text = tokenizer.decode([tid])
                                    _grammar_constraint.advance(tok_text)
                                except Exception:
                                    logger.debug(
                                        "Grammar constraint advance failed for token %d",
                                        tid,
                                        exc_info=True,
                                    )

                        # Emit bonus token (model's own prediction at rejection/last point)
                        if (
                            not _stopped
                            and result.bonus_token is not None
                            and remaining > 0
                        ):
                            bonus = result.bonus_token
                            tokens.append(bonus)
                            all_token_ids.append(bonus)
                            remaining -= 1
                            if bonus in stop_ids:
                                tokens.pop()
                                _stopped_by_stop_id = True
                            else:
                                detokenizer.add_token(bonus)
                                # Suffix check ONLY when bonus is not a stop_id.
                                # Previously this ran unconditionally, so when bonus was
                                # a stop_id the detokenizer text was stale and a coincidental
                                # suffix match from prior tokens would cause a spurious
                                # tokens.pop() (double-pop) and incorrect _stopped_by_suffix.
                                if stop_suffixes and any(
                                    detokenizer.text.endswith(s) for s in stop_suffixes
                                ):
                                    tokens.pop()  # Exclude suffix-triggering token from count
                                    _stopped_by_suffix = True
                                    _stopped = True
                            # Advance sampler's grammar constraint for bonus token
                            if _grammar_constraint is not None:
                                try:
                                    tok_text = tokenizer.decode([bonus])
                                    _grammar_constraint.advance(tok_text)
                                except Exception:
                                    logger.debug(
                                        "grammar advance failed for n-gram bonus token",
                                        exc_info=True,
                                    )
                            # do NOT feed the bonus token into the cache.
                            # Aligning with mlx-lm, the bonus is the next token to
                            # feed — it stays OUT of the cache and the next
                            # verify_with_last_token feeds [bonus, drafts] directly
                            # (no pre-trim). The previous "feed the bonus" + verify
                            # "trim 1" pair left the bonus's stale K/V in the buffer
                            # at the rolled-back position, which the next forward
                            # attended to → repeated/garbled output (".txt.txt…").

                        self._ngram_stats["accepted"] += result.accepted_count

                        # Feed back to adaptive spec controller
                        if self._adaptive_spec is not None:
                            self._adaptive_spec.record_step(
                                n_draft, result.accepted_count
                            )

                # Cache KV state
                if self._kv_quant_bits is not None:
                    _maybe_quantize_kv_cache(
                        cache,
                        self._kv_quant_start,
                        self._kv_quant_group_size,
                        self._kv_quant_bits,
                    )
                if prefix_cache is not None:
                    prefix_cache.add(mx.array(input_ids), cache)

                # Finalize detokenizer to flush partial UTF-8 bytes before
                # assembling output. Without this, multi-byte characters
                # at token boundaries can be truncated, causing incorrect
                # suffix detection via detokenizer.text.endswith() above.
                try:
                    detokenizer.finalize()
                except Exception:
                    logger.debug(
                        "detokenizer finalize failed in n-gram spec", exc_info=True
                    )

                # Use detokenizer text when suffix matching is active (same
                # pattern as _generate_fast) because tokenizer.decode(tokens)
                # may contain a partial suffix that leaked across boundaries.
                if _stopped_by_suffix and stop_suffixes:
                    output_text = detokenizer.text
                    for s in stop_suffixes:
                        if output_text.endswith(s):
                            output_text = output_text[: -len(s)]
                            break
                    output_text = _clean_special_tokens(output_text)
                else:
                    output_text = tokenizer.decode(tokens, skip_special_tokens=True)
                mx.synchronize()
                return (
                    tokens,
                    output_text,
                    [],
                    ttft_s,
                    matched,
                    _stopped_by_suffix,
                    _stopped_by_stop_id,
                )
            finally:
                _unregister_inflight()

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        # (self-audit fix B): keep _lora_applied permanently False so the
        # legacy event-loop release blocks below are inert no-ops; the adapter
        # lifecycle now runs INSIDE _run_with_lora on the executor thread, serialized
        # with generate_step (max_workers=1) — the same keystone as _generate_fast.
        # Acquiring/releasing on the event loop mutated the shared model while a
        # different request's generation read it (cross-thread race).
        _lora_applied = False

        def _run_with_lora():
            _applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _applied:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            try:
                return _run()
            finally:
                if _applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(
                            "LoRA release failed (n-gram spec executor)", exc_info=True
                        )

        try:
            (
                tokens,
                output_text,
                _,
                ttft_s,
                cached_tokens,
                _stopped_by_suffix,
                _stopped_by_stop_id,
            ) = await loop.run_in_executor(executor, _run_with_lora)
        except MemoryError:
            logger.warning(
                "OOM during N-gram spec generation — returning memory_limit finish reason"
            )
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                logger.debug(
                    "GPU cache cleanup failed after n-gram spec OOM", exc_info=True
                )
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after n-gram spec OOM", exc_info=True
                    )
            return GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error="OOM during N-gram spec generation",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except RuntimeError as e:
            if "memory" in str(e).lower() or "out of" in str(e).lower():
                logger.warning(f"MLX OOM during N-gram spec generation: {e}")
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed after n-gram spec OOM (RuntimeError)",
                        exc_info=True,
                    )
                if (
                    _lora_applied
                    and hasattr(self, "_lora_manager")
                    and self._lora_manager is not None
                ):
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.warning(
                            "LoRA release failed after n-gram spec OOM (RuntimeError)",
                            exc_info=True,
                        )
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=str(e),
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after n-gram spec RuntimeError",
                        exc_info=True,
                    )
            # Return error output for non-OOM RuntimeError instead of
            # propagating to caller (which expects GenerationOutput).
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"RuntimeError during N-gram spec generation: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during N-gram spec generation: {e}", exc_info=True
            )
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after n-gram spec unexpected error",
                        exc_info=True,
                    )
            # Return error output instead of propagating exception to caller.
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"N-gram spec generation failed: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )

        # Determine finish_reason with cancel awareness.
        # Stop tokens are popped from `tokens`, so check the flags instead.
        _cancelled = _is_cancelled(cancel_event)
        if _cancelled or _stopped_by_suffix or _stopped_by_stop_id:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        output_text = _clean_special_tokens(output_text)

        # Trim stop suffix from output text when matched during generation
        if _stopped_by_suffix and stop_suffixes:
            for s in stop_suffixes:
                if output_text.endswith(s):
                    output_text = output_text[: -len(s)]
                    break

        # Build logprobs from generated tokens — n-gram spec decode does not
        # expose per-token logits from the verify step, so we cannot compute
        # real logprobs. Return None instead of fake 0.0 to avoid misleading.
        _ngram_logprobs = None
        if logprobs and tokens:
            _ngram_logprobs = None  # Real logprobs unavailable from n-gram spec path

        # Record TTFT in Prometheus for n-gram spec path
        if ttft_s > 0:
            try:
                from yunshu_gateway.middleware.prometheus_exporter import (
                    get_prometheus_metrics,
                )

                pm = get_prometheus_metrics()
                pm.observe_histogram(
                    "ttft_seconds", ttft_s, labels={"model_id": self.model_label}
                )
            except Exception:
                logger.debug(
                    "TTFT prometheus recording failed in n-gram spec path",
                    exc_info=True,
                )

        # Channel-style reasoning recovery (Gemma-4 <|channel>…<channel|>): the
        # greedy default routes here (n-gram spec), so the same recovery the fast
        # path does must run here too, else gemma reasoning leaks into content.
        output_text, _ch_reason = _recover_channel_reasoning(
            tokens, tokenizer, output_text
        )

        # Reasoning parser: extract thinking tokens from n-gram spec output.
        # N-gram spec decode does not track thinking tokens internally,
        # so we parse the output text for reasoning content.
        _ng_reasoning_tok = len(_ch_reason)
        if output_text:
            try:
                from .reasoning_parser import get_reasoning_parser

                rp = get_reasoning_parser(self.model_name)
                rp_out = rp.parse(output_text)
                if rp_out.reasoning and rp_out.reasoning_tokens > 0:
                    _ng_reasoning_tok = rp_out.reasoning_tokens
                    if rp_out.content != output_text:
                        output_text = rp_out.content
            except Exception:
                logger.debug(
                    "reasoning_parser failed in n-gram spec path", exc_info=True
                )

        if (
            _lora_applied
            and hasattr(self, "_lora_manager")
            and self._lora_manager is not None
        ):
            try:
                self._lora_manager.release_adapter(lora_adapter)
            except Exception:
                logger.warning(
                    "LoRA release failed after n-gram spec normal completion",
                    exc_info=True,
                )
        return GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(tokens),
            finished=True,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
            ttft_ms=round(ttft_s * 1000, 1),
            reasoning_tokens=_ng_reasoning_tok,
            logprobs=_ngram_logprobs,
        )

    async def _stream_generate_ngram_spec(
        self,
        prompt: str | list[dict],
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
        logprobs: bool = False,
        top_logprobs: int | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        cancel_event: asyncio.Event | None = None,
        timeout_seconds: float = 300.0,
        logits_processors: list | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        lora_adapter: str | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using N-gram speculative decoding (queue-based).

        UNREACHABLE in the streaming flow (honesty annotation): stream_generate()
        forces ``spec_decode = False`` (~4381) so the n-gram branch (~4537) routes to
        _stream_generate_fast instead of here (this impl
        early-terminates and is bandwidth-bound = no speedup). The documented spec-stream
        multi-token stop-prefix leak (the per-segment `suffix_hit` blanking below holds
        back only the CURRENT segment, not a prefix split across earlier tokens) is thus a
        DEAD-code defect, not a live bug — the live path _stream_generate_fast uses
        StopHoldbackBuffer. If re-enabled, route the consumer's new_text through
        StopHoldbackBuffer (the _hb pattern) before yielding.
        """
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        from .mlx_executor import get_mlx_executor

        tokenizer = self._tokenizer
        model = self._model

        # SAFETY GUARD (streaming): see _generate_ngram_spec. Spec verify
        # needs a trimmable cache; Qwen3.5's hybrid attention uses non-trimmable
        # ArraysCache (recurrent state), so delegate the whole stream to the plain
        # streaming fast path rather than corrupting into ".txt.txt..." repetition.
        try:
            from mlx_lm.models.cache import (
                can_trim_prompt_cache as _can_trim,
            )
            from mlx_lm.models.cache import (
                make_prompt_cache as _mk_cache,
            )

            _spec_cache_supported = _can_trim(_mk_cache(model))
        except Exception:
            _spec_cache_supported = False
        if not _spec_cache_supported:
            logger.info(
                "N-gram spec disabled (streaming): cache not trimmable — "
                "using the plain streaming fast path."
            )
            async for _chunk in self._stream_generate_fast(
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
                cancel_event=cancel_event,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                logits_processors=logits_processors,
            ):
                yield _chunk
            return
        # Fresh per-request proposer (n-gram or suffix) — avoids reset()/propose()
        # races on a shared instance under concurrency.
        proposer = self._new_request_proposer()

        # Handle messages-format prompts (list of dicts) — apply chat template.
        # route through _apply_chat_template + _encode_prompt (NOT raw tokenizer
        # calls) so this spec path matches the default fast path — role remap, family
        # adapters, AND the double-BOS guard. The raw apply_chat_template(tokenize=False)
        # emits a string already opening with the literal bos_token; a following
        # tokenizer.encode (add_special_tokens=True) prepended BOS a SECOND time for
        # Gemma/Llama-3/Mistral → [BOS, BOS, …], corrupting the first-token distribution.
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        stop_ids: set[int] = set()
        stop_suffixes = []
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                stop_ids.update(eid)
            elif eid is not None:
                stop_ids.add(eid)
        _eids = getattr(tokenizer, "eos_token_ids", None)
        if _eids is not None:  # may be a bare int (Qwen3.6-27B), not iterable
            stop_ids.update(
                _eids if isinstance(_eids, (list, tuple, set)) else (_eids,)
            )
        # Model (generation_)config multi-eos — e.g. Gemma-4 turn-end 106 the
        # tokenizer omits, so the model never stops and rambles.
        stop_ids.update(_read_config_eos_ids(self.model_name))
        if stop_token_ids:
            stop_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                try:
                    ids = tokenizer.encode(s)
                    if len(ids) == 1:
                        stop_ids.add(ids[0])
                    elif len(ids) > 1:
                        stop_suffixes.append(s)
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        # route temp>0 off mlx-lm's PRNG-trapped make_sampler (see the
        # non-streaming sibling). Greedy (temp==0) stays on argmax make_sampler.
        if temperature is not None and temperature > 1e-6:
            sampler = _build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                seed=seed,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        else:
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )

        # Grammar constraint: pre-validate draft tokens against allowed set
        _stream_grammar_constraint = None
        if json_schema is not None:
            try:
                sampler = _build_constrained_sampler(sampler, json_schema, tokenizer)
                _stream_grammar_constraint = (
                    sampler.constraint if hasattr(sampler, "constraint") else None
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for streaming n-gram spec",
                    exc_info=True,
                )

        # LoRA adapter: the acquire+apply / release+restore lifecycle MUST run on the
        # executor thread (inside _run), NOT here on the event loop — acquiring here
        # mutates the shared model off-thread while another request's generate_step
        # reads it (cross-thread wrong-adapter bleed). Kept False so the legacy
        # event-loop release below is an inert no-op; _run owns the lifecycle. Mirrors
        # the _generate_fast / n-gram-non-streaming keystone .
        _ng_s_lora_applied = False

        def _stream_grammar_filter_drafts(
            draft_ids: list[int],
            generated_ids: list[int],
        ) -> list[int]:
            """Filter draft tokens that violate grammar constraints (streaming path)."""
            if _stream_grammar_constraint is None:
                return draft_ids
            _stream_grammar_constraint.checkpoint()
            allowed = _stream_grammar_constraint.get_allowed_tokens(
                tokenizer, generated_ids
            )
            if not allowed:
                _stream_grammar_constraint.rollback()
                return draft_ids
            allowed_set = set(allowed)
            filtered = []
            for tid in draft_ids:
                if tid in allowed_set:
                    filtered.append(tid)
                    try:
                        tok_text = tokenizer.decode([tid])
                        _stream_grammar_constraint.advance(tok_text)
                    except Exception:
                        logger.debug(
                            "grammar constraint advance failed in streaming filter",
                            exc_info=True,
                        )
                        break
                    allowed = _stream_grammar_constraint.get_allowed_tokens(
                        tokenizer, generated_ids + filtered
                    )
                    if allowed:
                        allowed_set = set(allowed)
                    else:
                        break
                else:
                    break
            _stream_grammar_constraint.rollback()
            return filtered

        _sentinel = object()
        _q: asyncio.Queue = make_stream_queue(512)
        loop = asyncio.get_running_loop()
        from .streaming_optimizer import StreamingBackpressureController

        _backpressure = StreamingBackpressureController(max_queue_size=100)

        _bridge = StreamBridge(
            loop,
            _q,
            lambda it: it is _sentinel or isinstance(it, BaseException),
            on_overflow=lambda: _ng_timeout_cancel.set(),
            overflow_error="N-gram streaming queue overflow — output truncated",
        )

        def _put(item):
            return _bridge.put(item)

        # Inflight prefix sharing for streaming n-gram spec
        _ng_s_inflight_req_id = f"ng-s-{int(time.monotonic() * 1e6)}"

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_ng_s_inflight_req_id)
            except Exception:
                logger.debug(
                    "inflight prefix unregister failed in streaming n-gram spec",
                    exc_info=True,
                )

        _ng_timeout_cancel = threading.Event()
        _ng_gen_t0 = time.perf_counter()

        def _run():
            # Acquire+apply the LoRA adapter HERE, on the executor thread, serialized
            # with generate_step (max_workers=1) — never on the event loop (which would
            # mutate the shared model while another request reads it).
            _applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _applied:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            try:
                _run_inner()
            except Exception as e:
                logger.error(f"N-gram streaming generation failed: {e}", exc_info=True)
                try:
                    import mlx.core as _cleanup_mx

                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed in n-gram streaming error handler",
                        exc_info=True,
                    )
                _put(e)
            finally:
                if _applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(
                            "LoRA release failed (streaming n-gram spec executor)",
                            exc_info=True,
                        )
                _unregister_inflight()
                _put(_sentinel)

        def _run_inner():
            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            tokens = []
            all_token_ids = list(input_ids)

            prefix_cache = self._kv_prefix_cache
            if prefix_cache is not None:
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
            # Also evict from paged KV manager when enabled
            if self._kv_manager is not None:
                try:
                    self._kv_manager.memory_pressure_evict(
                        self._mem_pressure_threshold / 100.0
                    )
                except Exception:
                    logger.debug("paged KV pressure eviction failed", exc_info=True)
            try:
                cached_kv, _, matched = (
                    prefix_cache.get(ids)
                    if prefix_cache is not None
                    else (None, None, 0)
                )
            except Exception:
                logger.warning(
                    "KV prefix cache get failed in spec path — falling back to full prefill",
                    exc_info=True,
                )
                cached_kv, _, matched = None, None, 0
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids
            if len(ids_to_prefill) == 0 and len(ids) > 0:
                ids_to_prefill = ids[-1:]

            # Inflight prefix sharing: register for concurrent KV block sharing
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().register(
                    _ng_s_inflight_req_id,
                    [int(t) for t in ids],
                    cache,
                    self.model_name or "",
                )
            except Exception:
                logger.debug(
                    "inflight prefix register failed in streaming n-gram spec",
                    exc_info=True,
                )

            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            n_tok = 0

            # Thinking budget enforcement — detect <think/</think via single-token IDs
            _ng_think_start_token = None
            _ng_think_end_token = None
            if thinking_budget is not None or enable_thinking:
                # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
                _ng_think_start_token, _ng_think_end_token = _resolve_think_token_ids(
                    tokenizer
                )
            _ng_in_thinking = False
            _ng_thinking_tokens_used = 0

            def _ng_track_thinking(tid):
                """Track thinking state and enforce budget. Returns True if budget exceeded."""
                nonlocal _ng_in_thinking, _ng_thinking_tokens_used
                if _ng_think_start_token is None:
                    return False
                if not _ng_in_thinking and tid == _ng_think_start_token:
                    _ng_in_thinking = True
                elif _ng_in_thinking:
                    _ng_thinking_tokens_used += 1
                    if tid == _ng_think_end_token:
                        _ng_in_thinking = False
                        return False
                    if (
                        thinking_budget is not None
                        and _ng_thinking_tokens_used >= thinking_budget
                        and _ng_think_end_token is not None
                    ):
                        _ng_in_thinking = False
                        # Force-insert </think token
                        tokens.append(_ng_think_end_token)
                        all_token_ids.append(_ng_think_end_token)
                        n_tok_cur = n_tok + 1
                        detokenizer.add_token(_ng_think_end_token)
                        _end_text = detokenizer.last_segment
                        if _end_text:
                            _put((_end_text, n_tok_cur, None, _ng_think_end_token))
                        # Stop token to end generation
                        _put(("", n_tok_cur, "stop", _ng_think_end_token))
                        return True
                return False

            with _wired_limit_ctx(model):
                # Prefill + first token
                for token, _logits in generate_step(
                    ids_to_prefill,
                    model,
                    max_tokens=1,
                    sampler=sampler,
                    prompt_cache=cache,
                ):
                    first_token = int(token)
                    tokens.append(first_token)
                    all_token_ids.append(first_token)
                    n_tok += 1
                    if first_token in stop_ids:
                        # First token is stop — don't add to detokenizer, signal stop
                        n_tok -= 1  # Exclude stop token from completion count
                        _put(("", n_tok, "stop", first_token))
                        if prefix_cache is not None:
                            prefix_cache.add(ids, cache)
                        mx.synchronize()
                        _unregister_inflight()
                        return
                    detokenizer.add_token(first_token)
                    if _ng_track_thinking(first_token):
                        if prefix_cache is not None:
                            prefix_cache.add(ids, cache)
                        mx.synchronize()
                        _unregister_inflight()
                        return
                    _put((detokenizer.last_segment, n_tok, None, first_token))
                remaining = max_tokens - 1
                while remaining > 0:
                    if _is_cancelled(cancel_event):
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        if _remaining:
                            _put((_remaining, n_tok, None, 0))
                        _put(("", n_tok, "stop", 0))
                        return
                    if _ng_timeout_cancel.is_set():
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        if _remaining:
                            _put((_remaining, n_tok, None, 0))
                        _put(("", n_tok, "timeout", 0))
                        return
                    _adaptive_k = (
                        self._adaptive_spec.get_draft_length()
                        if self._adaptive_spec
                        else None
                    )
                    draft_ids = proposer.propose(all_token_ids)
                    if _adaptive_k is not None:
                        draft_ids = draft_ids[:_adaptive_k]
                    # Grammar-aware draft filtering: reject drafts that violate constraints
                    draft_ids = _stream_grammar_filter_drafts(draft_ids, all_token_ids)
                    n_draft = min(len(draft_ids), remaining)

                    if n_draft == 0:
                        step_input = mx.array(
                            [tokens[-1]]
                        )  # 1D — generate_step adds the batch dim
                        for token, _logits in generate_step(
                            step_input,
                            model,
                            max_tokens=1,
                            sampler=sampler,
                            prompt_cache=cache,
                        ):
                            token_id = int(token)
                            tokens.append(token_id)
                            all_token_ids.append(token_id)
                            remaining -= 1
                            n_tok += 1
                            stop_hit = token_id in stop_ids
                            if stop_hit:
                                n_tok -= 1  # Exclude stop token from count
                                # Stop token — don't add to detokenizer
                                _put(("", n_tok, "stop", token_id))
                                if prefix_cache is not None:
                                    prefix_cache.add(ids, cache)
                                mx.synchronize()
                                _unregister_inflight()
                                return
                            detokenizer.add_token(token_id)
                            if _ng_track_thinking(token_id):
                                if prefix_cache is not None:
                                    prefix_cache.add(ids, cache)
                                mx.synchronize()
                                _unregister_inflight()
                                return
                            suffix_hit = False
                            if stop_suffixes:
                                suffix_hit = any(
                                    detokenizer.text.endswith(s) for s in stop_suffixes
                                )
                            _text = "" if suffix_hit else detokenizer.last_segment
                            if suffix_hit:
                                n_tok -= 1  # Exclude suffix-triggering token from count
                            _put(
                                (_text, n_tok, "stop" if suffix_hit else None, token_id)
                            )
                            if suffix_hit:
                                detokenizer.finalize()
                                _remaining = detokenizer.last_segment
                                if stop_suffixes and _remaining:
                                    for s in stop_suffixes:
                                        if _remaining.endswith(s):
                                            _remaining = _remaining[: -len(s)]
                                            break
                                if _remaining:
                                    _put((_remaining, n_tok, None, token_id))
                                if prefix_cache is not None:
                                    prefix_cache.add(ids, cache)
                                mx.synchronize()
                                _unregister_inflight()
                                return
                        continue

                    self._ngram_stats["proposals"] += 1
                    self._ngram_stats["total_draft"] += n_draft
                    accepted = 0
                    stopped = False

                    # C10: Batch verify all K draft tokens in one forward pass
                    draft_arr = mx.array(draft_ids[:n_draft]).reshape(1, -1)
                    batch_logits = model(draft_arr, cache=cache)
                    if hasattr(batch_logits, "logits"):
                        batch_logits = batch_logits.logits

                    # Sequential verify
                    # NG-PEN: Apply penalty/bias to the bonus/correction logits
                    # at the first rejection position (the first "new" token).
                    _has_ng_pen_cpu = (
                        repetition_penalty != 1.0
                        or frequency_penalty != 0.0
                        or presence_penalty != 0.0
                        or (logit_bias is not None and len(logit_bias) > 0)
                    )
                    for i in range(n_draft):
                        # Apply penalties at the first rejection position only
                        if _has_ng_pen_cpu and i == accepted:
                            _ng_bonus_logits_cpu = batch_logits[0, i, :]
                            _ng_token_hist_cpu = all_token_ids
                            _ng_bonus_logits_cpu = _apply_spec_bonus_penalties(
                                _ng_bonus_logits_cpu,
                                _ng_token_hist_cpu,
                                len(input_ids),
                                repetition_penalty=repetition_penalty,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                logit_bias=logit_bias,
                            )
                            batch_logits[0, i, :] = _ng_bonus_logits_cpu
                        model_pick = int(mx.argmax(batch_logits[0, i], axis=-1).item())
                        draft_id = draft_ids[i]
                        is_accept = model_pick == draft_id
                        accepted_id = draft_id if is_accept else model_pick
                        tokens.append(accepted_id)
                        all_token_ids.append(accepted_id)
                        if is_accept:
                            accepted += 1
                        remaining -= 1
                        n_tok += 1
                        stop_hit = accepted_id in stop_ids
                        if stop_hit:
                            n_tok -= 1  # Exclude stop token from count
                            _put(("", n_tok, "stop", accepted_id))
                            stopped = True
                        else:
                            detokenizer.add_token(accepted_id)
                            if _ng_track_thinking(accepted_id):
                                stopped = True
                                break
                            suffix_hit = False
                            if stop_suffixes:
                                suffix_hit = any(
                                    detokenizer.text.endswith(s) for s in stop_suffixes
                                )
                            _text = "" if suffix_hit else detokenizer.last_segment
                            if suffix_hit:
                                n_tok -= 1  # Exclude suffix-triggering token from count
                            _put(
                                (
                                    _text,
                                    n_tok,
                                    "stop" if suffix_hit else None,
                                    accepted_id,
                                )
                            )
                            if suffix_hit:
                                stopped = True
                        # Advance grammar constraint for accepted/bonus token
                        if _stream_grammar_constraint is not None and not stop_hit:
                            try:
                                tok_text = tokenizer.decode([accepted_id])
                                _stream_grammar_constraint.advance(tok_text)
                            except Exception:
                                logger.debug(
                                    "grammar advance failed for n-gram streaming CPU token",
                                    exc_info=True,
                                )
                        if not is_accept:
                            stopped = True
                        if stopped:
                            break

                    # CPU fallback: trim KV cache for rejected tokens
                    if accepted < n_draft:
                        try:
                            from mlx_lm.models.cache import trim_prompt_cache

                            trim_prompt_cache(cache, n_draft - accepted)
                        except Exception:
                            logger.debug(
                                "trim_prompt_cache (n-gram CPU fallback) failed, falling back",
                                exc_info=True,
                            )
                            for c in cache:
                                if hasattr(c, "trim"):
                                    c.trim(n_draft - accepted)
                        # Feed the correction token to populate its KV entry
                        if not stopped and tokens:
                            _correction = tokens[-1]
                            _ = model(mx.array([[_correction]]), cache=cache)

                    self._ngram_stats["accepted"] += accepted
                    if self._adaptive_spec is not None:
                        self._adaptive_spec.record_step(n_draft, accepted)
                    if stopped:
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        # Trim stop suffix from remaining text — the suffix may
                        # span multiple tokens, so detokenizer.text still
                        # contains it even after finalize().
                        if stop_suffixes and _remaining:
                            for s in stop_suffixes:
                                if _remaining.endswith(s):
                                    _remaining = _remaining[: -len(s)]
                                    break
                        if _remaining:
                            _put((_remaining, n_tok, None, 0))
                        if prefix_cache is not None:
                            prefix_cache.add(ids, cache)
                        mx.synchronize()
                        return

            if prefix_cache is not None:
                prefix_cache.add(ids, cache)
            detokenizer.finalize()
            remaining = detokenizer.last_segment
            if remaining:
                _put((remaining, n_tok, None, 0))
            _put(("", n_tok, "length", 0))
            mx.synchronize()

        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run)

        accumulated = ""
        n_tok = 0
        _ng_ttft_recorded = False
        _ng_ttft_ms_val = 0.0
        _ng_gen_t0 = time.perf_counter()
        # Consumer-side thinking state mirrors GPU-side tracking for reporting
        _ng_consumer_think_start = None
        _ng_consumer_think_end = None
        _ng_consumer_in_thinking = False
        _ng_consumer_thinking_tokens = 0
        if thinking_budget is not None or enable_thinking:
            # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
            _ng_consumer_think_start, _ng_consumer_think_end = _resolve_think_token_ids(
                tokenizer
            )
        _ng_fp_lock = getattr(self, "_fast_path_lock", None)
        if _ng_fp_lock is not None:
            with _ng_fp_lock:
                self._active_fast_path_count += 1
        try:
            while True:
                # Check cancel_event from consumer side (mirrors MTP streaming path)
                if _is_cancelled(cancel_event):
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="stop",
                        ttft_ms=_ng_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=timeout_seconds)
                except TimeoutError:
                    logger.warning(
                        f"N-gram streaming timeout: no token for {timeout_seconds}s"
                    )
                    _ng_timeout_cancel.set()  # Signal GPU loop to stop
                    # Yield terminal output so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="error",
                        error=f"N-gram streaming timeout: no token for {timeout_seconds}s",
                        ttft_ms=_ng_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                if item is _sentinel:
                    break
                if isinstance(item, BaseException):
                    logger.warning(f"N-gram streaming error: {item}")
                    # Yield terminal error output so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="error",
                        error=str(item),
                        ttft_ms=_ng_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                new_text, tok_count, _fr_val, token_id = item
                # Backward compat: _fr_val may be bool or str or None
                if isinstance(_fr_val, bool):
                    finish_reason = "stop" if _fr_val else None
                    done = _fr_val
                else:
                    finish_reason = _fr_val
                    done = _fr_val is not None
                accumulated += new_text
                if len(accumulated) > _MAX_STREAMING_TEXT_BUFFER:
                    logger.error(
                        "Streaming text buffer exceeded 1MB limit (%d bytes) — truncating",
                        len(accumulated),
                    )
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated),
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=tok_count,
                        finished=True,
                        finish_reason="length",
                        error="Streaming text buffer exceeded 1MB limit",
                        cached_tokens=0,
                        reasoning_tokens=_ng_consumer_thinking_tokens
                        if _ng_consumer_think_start is not None
                        else 0,
                        ttft_ms=_ng_ttft_ms_val,
                    )
                    break
                n_tok = tok_count
                # Consumer-side thinking state tracking for reasoning_tokens reporting
                if _ng_consumer_think_start is not None and isinstance(token_id, int):
                    if (
                        not _ng_consumer_in_thinking
                        and token_id == _ng_consumer_think_start
                    ):
                        _ng_consumer_in_thinking = True
                    elif _ng_consumer_in_thinking:
                        _ng_consumer_thinking_tokens += 1
                        if token_id == _ng_consumer_think_end:
                            _ng_consumer_in_thinking = False

                # Streaming backpressure: slow down if client can't keep up
                if _backpressure.check_backpressure(_q.qsize()):
                    _delay = _backpressure.get_delay_ms(_q.qsize())
                    if _delay > 0:
                        await asyncio.sleep(_delay / 1000)

                # Record TTFT on first token
                if not _ng_ttft_recorded and n_tok == 1:
                    _ng_ttft_recorded = True
                    _ng_ttft_s = time.perf_counter() - _ng_gen_t0
                    _ng_ttft_ms_val = round(_ng_ttft_s * 1000, 1)
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _ng_ttft_s,
                            labels={"model_id": self.model_label},
                        )
                    except Exception:
                        logger.debug(
                            "N-gram streaming TTFT prometheus recording failed",
                            exc_info=True,
                        )

                # Build logprobs for this token — n-gram spec decode does not
                # expose per-token logits in the queue-based streaming path.
                # When logprobs=True, return an empty list with a warning so
                # consumers get a valid (but empty) structure instead of None.
                _chunk_logprobs = None
                if logprobs:
                    _chunk_logprobs: list = []
                    if n_tok <= 1:
                        logger.debug(
                            "N-gram streaming path: real logprobs unavailable "
                            "(per-token logits not exposed via queue). "
                            "Returning empty logprobs list."
                        )

                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                    reasoning_tokens=_ng_consumer_thinking_tokens
                    if _ng_consumer_think_start is not None
                    else 0,
                    cached_tokens=0,
                    logprobs=_chunk_logprobs,
                    ttft_ms=_ng_ttft_ms_val,
                )
                if done:
                    break
        finally:
            # Decrement active fast path count (prevents model eviction mid-generation)
            # Release LoRA adapter ref acquired at start of streaming n-gram spec
            if (
                _ng_s_lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.debug(
                        "LoRA release failed in streaming n-gram spec path",
                        exc_info=True,
                    )
            _ng_fp_lock = getattr(self, "_fast_path_lock", None)
            if _ng_fp_lock is not None:
                with _ng_fp_lock:
                    self._active_fast_path_count -= 1
            if not future.done():
                future.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await future
            # Drain queue to unblock any pending call_soon_threadsafe from
            # the executor thread, preventing GPU work from continuing after
            # the consumer has stopped iterating.
            while not _q.empty():
                try:
                    _q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def _generate_mtp(
        self,
        prompt: str | list[dict],
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
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        thinking_budget: int | None = None,
        cancel_event: asyncio.Event | None = None,
        json_schema: dict | str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logits_processors: list | None = None,
        timeout_seconds: float = 300.0,
        lora_adapter: str | None = None,
    ) -> GenerationOutput:
        """Generate using MTP speculative decoding (built-in prediction heads).

        Uses the model's own MTP heads to propose draft tokens, then verifies
        via the backbone forward with n_confirmed=1 for zero-cost reject.
        Best for Qwen3.5 and other models with GatedDeltaNet SSM layers.
        """
        from .mlx_executor import get_mlx_executor

        # Build grammar constraint from json_schema if provided.
        # The constraint filters MTP draft logits so draft tokens respect
        # structured output requirements.
        _mtp_constraint = None
        if json_schema is not None:
            try:
                from mlx_lm.sample_utils import make_sampler as _make_s

                _tmp_sampler = _make_s(temp=0.0)
                _constrained_sampler = _build_constrained_sampler(
                    _tmp_sampler,
                    json_schema,
                    self._tokenizer,
                )
                _mtp_constraint = getattr(
                    _constrained_sampler,
                    "constraint",
                    None,
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for MTP, continuing without",
                    exc_info=True,
                )
        import mlx.core as mx

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        tokenizer = self._tokenizer
        mtp_decoder = self._mtp_decoder

        # double-BOS guard (see non-streaming siblings) — route through
        # _apply_chat_template + _encode_prompt instead of raw apply_chat_template+encode.
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        # Build EOS + stop token sets
        eos_ids: set[int] = set()
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
        # Model (generation_)config multi-eos — e.g. Gemma-4 turn-end 106.
        eos_ids.update(_read_config_eos_ids(self.model_name))
        if stop_token_ids:
            eos_ids.update(stop_token_ids)
        stop_suffixes = []
        if stop:
            for s in stop:
                try:
                    ids = tokenizer.encode(s)
                    if len(ids) == 1:
                        eos_ids.add(ids[0])
                    elif len(ids) > 1:
                        stop_suffixes.append(s)
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        if seed is not None:
            mx.random.seed(seed)

        # Build sampler for MTP path — applied to bonus tokens and rejection
        # corrections while the draft/verify comparison stays greedy.
        from mlx_lm.sample_utils import make_sampler

        # temp>0 → per-request sampler (no mlx-lm PRNG-trap collapse, seed
        # honored); temp==0 with filters → make_sampler (argmax, unchanged); fully
        # greedy-unconstrained → None.
        if temperature is not None and temperature > 1e-6:
            _mtp_sampler = _build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                seed=seed,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        elif top_p < 1.0 or top_k > 0 or min_p > 0 or xtc_probability > 0:
            _mtp_sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        else:
            _mtp_sampler = None

        # Use incremental detokenizer for correct multi-byte UTF-8
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()

        def _run():
            return mtp_decoder.generate(
                input_ids,
                max_tokens=max_tokens,
                cancel_event=cancel_event,
                sampler=_mtp_sampler,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                constraint=_mtp_constraint,
            )

        # (self-audit fix B): keep _lora_applied permanently False so the
        # _lora_release() helper below is an inert no-op; the adapter lifecycle now
        # runs INSIDE _run_with_lora on the executor thread, serialized with
        # generate_step (max_workers=1) — the same keystone as _generate_fast.
        # Acquiring/releasing on the event loop mutated the shared model while a
        # different request's generation read it (cross-thread race).
        _lora_applied = False

        def _lora_release():
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning("LoRA release failed in MTP path", exc_info=True)

        def _run_with_lora():
            _applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _applied:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            try:
                return _run()
            finally:
                if _applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(
                            "LoRA release failed (MTP executor)", exc_info=True
                        )

        _mtp_gen_t0 = time.perf_counter()
        try:
            token_ids = await asyncio.wait_for(
                loop.run_in_executor(executor, _run_with_lora),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            logger.warning(f"MTP generation timed out after {timeout_seconds}s")
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                logger.debug(
                    "GPU cache cleanup failed after MTP timeout", exc_info=True
                )
            _lora_release()
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"MTP generation timed out after {timeout_seconds}s",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except MemoryError:
            logger.warning(
                "OOM during MTP generation — returning memory_limit finish reason"
            )
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                logger.debug("GPU cache cleanup failed after MTP OOM", exc_info=True)
            _lora_release()
            return GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error="OOM during MTP generation",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except RuntimeError as e:
            if "memory" in str(e).lower() or "out of" in str(e).lower():
                logger.warning(f"MLX OOM during MTP generation: {e}")
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed after MTP OOM (RuntimeError)",
                        exc_info=True,
                    )
                _lora_release()
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=str(e),
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            _lora_release()
            # Return error output for non-OOM RuntimeError instead of
            # propagating to caller (which expects GenerationOutput).
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"RuntimeError during MTP generation: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except Exception as e:
            logger.error(f"Unexpected error during MTP generation: {e}", exc_info=True)
            _lora_release()
            # Return error output instead of propagating exception to caller.
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"MTP generation failed: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        _mtp_ttft_s = time.perf_counter() - _mtp_gen_t0

        # Thinking budget enforcement (MTP decoder does not support it natively).
        # Detect <think/</think token IDs and truncate thinking content when the
        # budget is exceeded. This is a post-processing approximation — the MTP
        # decoder already generated all tokens, but we cap the output to respect
        # the budget, matching the behavior of _generate_fast.
        _mtp_thinking_tokens_used = 0
        _mtp_think_end_token = None
        _mtp_in_thinking = False
        _mtp_think_budget_truncate_idx = None
        if (thinking_budget is not None or enable_thinking) and token_ids:
            # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
            _mtp_think_start_token, _mtp_think_end_token = _resolve_think_token_ids(
                tokenizer
            )
            if _mtp_think_end_token is not None:
                for _i, _tid in enumerate(token_ids):
                    if not _mtp_in_thinking and _tid == _mtp_think_start_token:
                        _mtp_in_thinking = True
                    elif _mtp_in_thinking:
                        if _tid == _mtp_think_end_token:
                            _mtp_in_thinking = False
                        else:
                            _mtp_thinking_tokens_used += 1
                            if (
                                thinking_budget is not None
                                and _mtp_thinking_tokens_used >= thinking_budget
                            ):
                                _mtp_think_budget_truncate_idx = _i
                                break

        if _mtp_think_budget_truncate_idx is not None:
            token_ids = token_ids[:_mtp_think_budget_truncate_idx]
            if _mtp_think_end_token is not None:
                token_ids.append(_mtp_think_end_token)
                # Count the forced think_end_token in reasoning tokens so
                # reasoning_tokens + content_tokens == completion_tokens.
                _mtp_thinking_tokens_used += 1

        # Truncate at stop tokens (exclude stop token from output).
        # Use a temporary detokenizer to probe for suffix matches WITHOUT
        # corrupting the main detokenizer's state when a suffix is hit.
        # Previously, add_token(tid) was called before the suffix check,
        # so the suffix-triggering token leaked into detokenizer.text even
        # though token_ids was correctly truncated — causing the suffix
        # to appear in the output despite the trimming below.
        hit_stop = False
        hit_suffix = False
        _mtp_completion_count = len(token_ids)  # Track actual completion count
        for i, tid in enumerate(token_ids):
            if tid in eos_ids:
                token_ids = token_ids[:i]
                _mtp_completion_count = i
                hit_stop = True
                break
            # Probe suffix match using the main detokenizer, but be
            # prepared to roll back if it triggers.
            detokenizer.add_token(tid)
            if stop_suffixes and any(
                detokenizer.text.endswith(s) for s in stop_suffixes
            ):
                # Roll back: remove the suffix-triggering token from the
                # detokenizer so its state matches the truncated token_ids.
                # NaiveStreamingDetokenizer supports .tokens attribute.
                if hasattr(detokenizer, "tokens") and detokenizer.tokens:
                    detokenizer.tokens.pop()
                # Re-initialize detokenizer state from remaining tokens
                # to ensure .text is consistent (simply popping .tokens
                # does not update the internal byte buffer).
                _kept = (
                    list(detokenizer.tokens)
                    if hasattr(detokenizer, "tokens")
                    else token_ids[:i]
                )
                detokenizer.reset()
                for _t in _kept:
                    detokenizer.add_token(_t)
                _mtp_completion_count = i
                token_ids = token_ids[:i]
                hit_suffix = True
                break
        else:
            # No stop/suffix hit — completion count is all tokens
            _mtp_completion_count = len(token_ids)
        detokenizer.finalize()
        output_text = _clean_special_tokens(detokenizer.text)

        # Determine finish_reason with cancel awareness
        _cancelled = _is_cancelled(cancel_event)
        finish_reason = "stop" if _cancelled or hit_stop or hit_suffix else "length"

        # Record MTP stats + TTFT in Prometheus
        try:
            from yunshu_gateway.middleware.prometheus_exporter import (
                get_prometheus_metrics,
            )

            pm = get_prometheus_metrics()
            s = mtp_decoder.stats
            if s.total_cycles > 0:
                _ml = {"model_id": self.model_label}
                pm.set_gauge(
                    "mtp_acceptance_rate", s.accepts / s.total_cycles, labels=_ml
                )
                pm.set_counter("mtp_total_cycles", s.total_cycles, labels=_ml)
            pm.observe_histogram(
                "ttft_seconds", _mtp_ttft_s, labels={"model_id": self.model_label}
            )
        except Exception:
            logger.debug("MTP metrics export failed", exc_info=True)

        # Build logprobs from MTP output tokens — MTP decoder uses greedy
        # decoding internally and does not expose per-token logits.
        # Return None instead of fake 0.0 to avoid misleading consumers.
        _mtp_logprobs = None
        if logprobs and token_ids:
            _mtp_logprobs = None  # Real logprobs unavailable from MTP path

        # Reasoning parser: extract thinking tokens from MTP output text when
        # the tokenizer supports <think/</think single-token encoding.
        _mtp_reasoning_tok = _mtp_thinking_tokens_used
        if _mtp_reasoning_tok == 0 and output_text:
            try:
                from .reasoning_parser import get_reasoning_parser

                rp = get_reasoning_parser(self.model_name)
                rp_out = rp.parse(output_text)
                if rp_out.reasoning and rp_out.reasoning_tokens > 0:
                    _mtp_reasoning_tok = rp_out.reasoning_tokens
                    if rp_out.content != output_text:
                        output_text = rp_out.content
            except Exception:
                logger.debug("reasoning_parser failed in MTP path", exc_info=True)

        _lora_release()
        return GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=_mtp_completion_count,
            finished=True,
            finish_reason=finish_reason,
            reasoning_tokens=_mtp_reasoning_tok,
            cached_tokens=0,
            logprobs=_mtp_logprobs,
            ttft_ms=round(_mtp_ttft_s * 1000, 1),
        )

    async def _stream_generate_mtp(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        cancel_event: asyncio.Event | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        timeout_seconds: float = 300.0,
        json_schema: dict | str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logits_processors: list | None = None,
        lora_adapter: str | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using MTP speculative decoding (queue-based).

        Runs MTPDecoder on the executor thread and yields accepted tokens
        as they are verified by the backbone forward pass.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        from .mlx_executor import get_mlx_executor

        # Build grammar constraint from json_schema if provided.
        # The constraint is used to filter MTP draft logits so draft tokens
        # respect structured output requirements.
        _mtp_grammar_constraint = None
        if json_schema is not None:
            try:
                from mlx_lm.sample_utils import make_sampler as _make_s

                _tmp_sampler = _make_s(temp=0.0)
                _constrained_sampler = _build_constrained_sampler(
                    _tmp_sampler,
                    json_schema,
                    self._tokenizer,
                )
                _mtp_grammar_constraint = getattr(
                    _constrained_sampler,
                    "constraint",
                    None,
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for MTP streaming, "
                    "continuing without constraint filtering",
                    exc_info=True,
                )

        tokenizer = self._tokenizer
        model = self._model
        mtp_decoder = self._mtp_decoder

        # double-BOS guard (see non-streaming siblings) — route through
        # _apply_chat_template + _encode_prompt instead of raw apply_chat_template+encode.
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        eos_ids: set[int] = set()
        stop_suffixes: list[str] = []
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
        _eids = getattr(tokenizer, "eos_token_ids", None)
        if _eids is not None:  # may be a bare int (Qwen3.6-27B), not iterable
            eos_ids.update(_eids if isinstance(_eids, (list, tuple, set)) else (_eids,))
        # Model (generation_)config multi-eos — e.g. Gemma-4 turn-end 106.
        eos_ids.update(_read_config_eos_ids(self.model_name))
        if stop_token_ids:
            eos_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                try:
                    ids = tokenizer.encode(s)
                    if len(ids) == 1:
                        eos_ids.add(ids[0])
                    elif len(ids) > 1:
                        stop_suffixes.append(s)
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        if seed is not None:
            mx.random.seed(seed)

        # Build sampler for MTP streaming path — applied to bonus tokens,
        # rejection corrections, and first token (NOT draft/verify comparison).
        from mlx_lm.sample_utils import make_sampler

        # temp>0 → per-request sampler (no mlx-lm PRNG-trap collapse, seed
        # honored); temp==0 with filters → make_sampler (argmax, unchanged); fully
        # greedy-unconstrained → None.
        if temperature is not None and temperature > 1e-6:
            _mtp_sampler = _build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                seed=seed,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        elif top_p < 1.0 or top_k > 0 or min_p > 0 or xtc_probability > 0:
            _mtp_sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        else:
            _mtp_sampler = None

        # Inflight prefix sharing: register for concurrent KV block sharing
        _inflight_req_id = f"mtp-s-{id(self)}-{int(time.monotonic() * 1e6)}"
        try:
            from .inflight_prefix_sharing import get_inflight_tracker

            get_inflight_tracker().register(
                _inflight_req_id,
                token_ids=input_ids,
                kv_cache=None,
            )
        except Exception:
            logger.debug("MTP inflight prefix register failed", exc_info=True)

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_inflight_req_id)
            except Exception:
                logger.debug("MTP inflight prefix unregister failed", exc_info=True)

        _lora_applied = False
        if (
            lora_adapter
            and hasattr(self, "_lora_manager")
            and self._lora_manager is not None
        ):
            try:
                _lora_applied = self._lora_manager.acquire_adapter(lora_adapter)
            except Exception as _le:  # fail loud, don't serve base
                raise RuntimeError(
                    f"LoRA adapter '{lora_adapter}' could not be applied"
                ) from _le
            if not _lora_applied:
                raise RuntimeError(
                    f"LoRA adapter '{lora_adapter}' could not be applied"
                )

        _sentinel = object()
        _q: asyncio.Queue = make_stream_queue(512)
        loop = asyncio.get_running_loop()
        from .streaming_optimizer import StreamingBackpressureController

        _mtp_timeout_cancel = threading.Event()
        _mtp_gen_t0 = time.perf_counter()
        _backpressure = StreamingBackpressureController(max_queue_size=100)

        _bridge = StreamBridge(
            loop,
            _q,
            lambda it: it is _sentinel or isinstance(it, BaseException),
            on_overflow=lambda: _mtp_timeout_cancel.set(),
            overflow_error="MTP streaming queue overflow — output truncated",
        )

        def _put(item):
            return _bridge.put(item)

        def _run():
            try:
                ids = mx.array(input_ids)
                cache = make_prompt_cache(model)
                detokenizer = tokenizer.detokenizer
                detokenizer.reset()
                # Multi-token stop hold-back : the MTP path emitted each
                # accepted draft/bonus/correction token's text immediately and only
                # trimmed the stop on the COMPLETING token, leaking the earlier
                # tokens of a multi-token stop . Route
                # accepted-token text through the buffer; empty stop set → passthrough.
                _hb = StopHoldbackBuffer(stop_suffixes)

                # Thinking budget enforcement — detect <think/</think via single-token IDs
                _mtp_think_start_token = None
                _mtp_think_end_token = None
                if thinking_budget is not None or enable_thinking:
                    # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
                    _mtp_think_start_token, _mtp_think_end_token = (
                        _resolve_think_token_ids(tokenizer)
                    )
                _mtp_in_thinking = False
                _mtp_thinking_tokens_used = 0

                def _mtp_track_thinking(tid):
                    """Track thinking state and enforce budget. Returns True if budget exceeded."""
                    nonlocal _mtp_in_thinking, _mtp_thinking_tokens_used
                    if _mtp_think_start_token is None:
                        return False
                    if not _mtp_in_thinking and tid == _mtp_think_start_token:
                        _mtp_in_thinking = True
                    elif _mtp_in_thinking:
                        _mtp_thinking_tokens_used += 1
                        if tid == _mtp_think_end_token:
                            _mtp_in_thinking = False
                            return False
                        if (
                            thinking_budget is not None
                            and _mtp_thinking_tokens_used >= thinking_budget
                            and _mtp_think_end_token is not None
                        ):
                            _mtp_in_thinking = False
                            # Force-insert </think token
                            generated.append(_mtp_think_end_token)
                            detokenizer.add_token(_mtp_think_end_token)
                            _end_text = detokenizer.last_segment
                            _n = len(generated)
                            if _end_text:
                                _put((_end_text, _n, None, _mtp_think_end_token))
                            _put(("", _n, "stop", _mtp_think_end_token))
                            return True
                    return False

                # Prefill
                out, hidden = model(ids.reshape(1, -1), cache=cache, return_hidden=True)
                mx.synchronize()
                # Apply sampler to first token if available
                if _mtp_sampler is not None:
                    first = int(_mtp_sampler(out[0, -1:, :]).item())
                else:
                    first = int(mx.argmax(out[0, -1, :]).item())

                generated = [first]
                primary = first
                primary_h = hidden[:, -1:, :]

                if first in eos_ids:
                    # First token is stop — don't add to detokenizer, just signal stop
                    detokenizer.finalize()
                    _put(("", 0, "stop", first))
                    return

                # Yield first token via incremental detokenizer (through the
                # hold-back buffer so a stop prefix in the first token is withheld).
                detokenizer.add_token(first)
                _mtp_track_thinking(first)
                _first_emit = _hb.feed(_clean_special_tokens(detokenizer.last_segment))
                if _first_emit:
                    _put((_first_emit, 1, None, first))

                from .n_confirmed_patch import clear_rollback, restore_rollback

                # Emit an accepted token's cleaned text through the stop hold-back
                # buffer. Returns True if a multi-token string stop completed
                # (caller must set _early_stop and break). On an EOS token, call
                # _mtp_flush_held() instead to release genuine held text.
                def _mtp_emit(tok):
                    _seg = _clean_special_tokens(detokenizer.last_segment)
                    if stop_suffixes and any(
                        detokenizer.text.endswith(s) for s in stop_suffixes
                    ):
                        # include feed()'s pre-stop return (else content fused with
                        # the stop token is lost). Parity with the VLM/text paths (this MTP
                        # streaming path is currently unreachable, but fixed for correctness).
                        _tail = _hb.feed(_seg) + _hb.take_stopped()
                        if _tail:
                            _put((_tail, len(generated) - 1, None, tok))
                        _put(("", len(generated) - 1, "stop", tok))
                        return True
                    _emit = _hb.feed(_seg)
                    if _emit:
                        _put((_emit, len(generated), None, tok))
                    return False

                def _mtp_flush_held(tok):
                    # Release any text held back as a stop-prefix — it's genuine
                    # output when generation ends on an EOS/stop *token*.
                    _f = _hb.flush()
                    if _f:
                        _put((_f, len(generated) - 1, None, tok))

                _early_stop = (
                    False  # Set True when while loop breaks due to stop/cancel
                )

                while len(generated) < max_tokens:
                    # Check cancel_event
                    if _is_cancelled(cancel_event):
                        # Emit stop chunk before breaking so consumer sees finished=True
                        detokenizer.finalize()
                        _remaining = _hb.feed(detokenizer.last_segment) + _hb.flush()
                        if _remaining:
                            _put((_remaining, len(generated), None, None))
                        _put(("", len(generated), "stop", None))
                        _early_stop = True
                        break
                    # Check timeout-driven cancel from consumer
                    if _mtp_timeout_cancel.is_set():
                        detokenizer.finalize()
                        _remaining = _hb.feed(detokenizer.last_segment) + _hb.flush()
                        if _remaining:
                            _put((_remaining, len(generated), None, None))
                        _put(("", len(generated), "timeout", None))
                        _early_stop = True
                        break

                    # MTP draft — always greedy, with optional grammar constraint masking
                    # Checkpoint grammar constraint before draft
                    if _mtp_grammar_constraint is not None and hasattr(
                        _mtp_grammar_constraint, "checkpoint"
                    ):
                        try:
                            _mtp_grammar_constraint.checkpoint()
                        except Exception:
                            logger.debug("MTP grammar checkpoint failed", exc_info=True)
                    draft = mtp_decoder._mtp_draft(
                        primary_h,
                        primary,
                        constraint=_mtp_grammar_constraint,
                        generated_ids=generated,
                    )

                    # Verify: backbone forward [primary, draft] with n_confirmed=1
                    verify_out, verify_h = model(
                        mx.array([[primary, draft]]),
                        cache=cache,
                        return_hidden=True,
                        n_confirmed=1,
                    )
                    mx.synchronize()
                    # v0 MUST be greedy for spec decode acceptance check
                    v0 = int(mx.argmax(verify_out[0, 0, :]).item())
                    # MTP-PEN: Apply penalty/bias to bonus token logits (v1).
                    # Penalties are applied to the bonus token only — draft tokens
                    # are already committed via acceptance check.
                    _has_mtp_pen = (
                        repetition_penalty != 1.0
                        or frequency_penalty != 0.0
                        or presence_penalty != 0.0
                        or (logit_bias is not None and len(logit_bias) > 0)
                    )
                    _mtp_bonus_logits = verify_out[0, 1, :]
                    if _has_mtp_pen:
                        _mtp_token_hist = list(input_ids) + generated
                        _mtp_bonus_logits = _apply_spec_bonus_penalties(
                            _mtp_bonus_logits,
                            _mtp_token_hist,
                            len(input_ids),
                            repetition_penalty=repetition_penalty,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            logit_bias=logit_bias,
                        )
                        verify_out[0, 1, :] = _mtp_bonus_logits
                    # v1 (bonus) can use sampler for non-greedy output
                    if _mtp_sampler is not None:
                        v1 = int(_mtp_sampler(verify_out[0, 1:2, :]).item())
                    else:
                        v1 = int(mx.argmax(verify_out[0, 1, :]).item())

                    if v0 == draft:
                        # Accept
                        clear_rollback(cache)
                        # Discard grammar constraint checkpoint (all accepted)
                        if _mtp_grammar_constraint is not None:
                            try:
                                if hasattr(
                                    _mtp_grammar_constraint, "discard_checkpoint"
                                ):
                                    _mtp_grammar_constraint.discard_checkpoint()
                            except Exception:
                                logger.debug(
                                    "MTP grammar discard_checkpoint failed",
                                    exc_info=True,
                                )
                        generated.append(draft)
                        if draft in eos_ids:
                            # Stop token — flush held text, exclude EOS from count
                            _mtp_flush_held(draft)
                            _put(("", len(generated) - 1, "stop", draft))
                            _early_stop = True
                            break

                        detokenizer.add_token(draft)
                        if _mtp_track_thinking(draft):
                            _early_stop = True
                            break
                        if _mtp_emit(draft):
                            _early_stop = True
                            break

                        # Advance grammar constraint with accepted draft token
                        if _mtp_grammar_constraint is not None and draft not in eos_ids:
                            try:
                                _mtp_grammar_constraint.advance(
                                    tokenizer.decode([draft])
                                )
                            except Exception:
                                logger.debug(
                                    "MTP grammar advance (draft accepted) failed",
                                    exc_info=True,
                                )

                        # Bonus token
                        generated.append(v1)
                        if v1 in eos_ids:
                            # Stop token — flush held text, exclude EOS from count
                            _mtp_flush_held(v1)
                            _put(("", len(generated) - 1, "stop", v1))
                            _early_stop = True
                            break

                        detokenizer.add_token(v1)
                        if _mtp_track_thinking(v1):
                            _early_stop = True
                            break
                        if _mtp_emit(v1):
                            _early_stop = True
                            break
                        # Advance grammar constraint with bonus token
                        if _mtp_grammar_constraint is not None and v1 not in eos_ids:
                            try:
                                _mtp_grammar_constraint.advance(tokenizer.decode([v1]))
                            except Exception:
                                logger.debug(
                                    "MTP grammar advance (bonus) failed", exc_info=True
                                )
                        primary = v1
                        primary_h = verify_h[:, -1:, :]
                    else:
                        # Reject: restore rollback (zero-cost)
                        restore_rollback(cache)
                        # Rollback grammar constraint to pre-draft state
                        if _mtp_grammar_constraint is not None and hasattr(
                            _mtp_grammar_constraint, "rollback"
                        ):
                            try:
                                _mtp_grammar_constraint.rollback()
                            except Exception:
                                logger.debug(
                                    "MTP grammar rollback failed", exc_info=True
                                )
                        # MTP-PEN: Apply penalty/bias to rejection correction logits (v0).
                        # The correction token is the first new token after the rejection.
                        if _has_mtp_pen:
                            _mtp_corr_logits = verify_out[0, 0, :]
                            _mtp_token_hist_corr = list(input_ids) + generated
                            _mtp_corr_logits = _apply_spec_bonus_penalties(
                                _mtp_corr_logits,
                                _mtp_token_hist_corr,
                                len(input_ids),
                                repetition_penalty=repetition_penalty,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                logit_bias=logit_bias,
                            )
                            verify_out[0, 0, :] = _mtp_corr_logits
                        # Apply sampler to rejection correction token
                        if _mtp_sampler is not None:
                            v0 = int(_mtp_sampler(verify_out[0, 0:1, :]).item())
                        generated.append(v0)
                        if v0 in eos_ids:
                            # Stop token — flush held text, exclude EOS from count
                            _mtp_flush_held(v0)
                            _put(("", len(generated) - 1, "stop", v0))
                            _early_stop = True
                            break

                        detokenizer.add_token(v0)
                        if _mtp_track_thinking(v0):
                            _early_stop = True
                            break
                        if _mtp_emit(v0):
                            _early_stop = True
                            break
                        # Advance grammar constraint with correction token
                        if _mtp_grammar_constraint is not None and v0 not in eos_ids:
                            try:
                                _mtp_grammar_constraint.advance(tokenizer.decode([v0]))
                            except Exception:
                                logger.debug(
                                    "MTP grammar advance (correction) failed",
                                    exc_info=True,
                                )
                        primary = v0
                        # Re-feed correction token through rolled-back cache to
                        # get a hidden state consistent with the new primary token.
                        # Using verify_h[:, 0:1, :] here is WRONG because verify_h
                        # was computed before rollback — the cache state has changed.
                        _out_corr, _hid_corr = model(
                            mx.array([[v0]]),
                            cache=cache,
                            return_hidden=True,
                        )
                        mx.synchronize()
                        primary_h = _hid_corr[:, -1:, :]

                # Emit "length" finish chunk only when max_tokens exhausted naturally.
                # If the loop broke early (stop/cancel), a terminal chunk was already
                # emitted inside the loop — skip the spurious second one.
                if not _early_stop:
                    detokenizer.finalize()
                    _remaining = _hb.feed(detokenizer.last_segment) + _hb.flush()
                    if _remaining:
                        _put((_remaining, len(generated), None, None))
                    _put(("", len(generated), "length", None))
            except Exception as e:
                logger.error(f"MTP streaming generation failed: {e}", exc_info=True)
                # Finalize detokenizer to flush partial UTF-8 bytes before
                # reporting the error — without this, any bytes buffered in
                # the detokenizer's internal state are silently lost.
                try:
                    detokenizer.finalize()
                    _final_segment = detokenizer.last_segment
                    if _final_segment:
                        _put((_final_segment, len(generated), None, None))
                except Exception:
                    logger.debug(
                        "detokenizer finalize in MTP error handler failed",
                        exc_info=True,
                    )
                try:
                    mx.synchronize()
                    mx.clear_cache()
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed in MTP streaming error handler",
                        exc_info=True,
                    )
                _put(e)
            finally:
                _put(_sentinel)

        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run)

        accumulated = ""
        n_tok = 0
        _mtp_ttft_recorded = False
        _mtp_ttft_ms_val = 0.0
        _mtp_gen_t0 = time.perf_counter()
        # Consumer-side thinking state mirrors GPU-side tracking for reporting
        _mtp_consumer_think_start = None
        _mtp_consumer_think_end = None
        _mtp_consumer_in_thinking = False
        _mtp_consumer_thinking_tokens = 0
        if thinking_budget is not None or enable_thinking:
            # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
            _mtp_consumer_think_start, _mtp_consumer_think_end = (
                _resolve_think_token_ids(tokenizer)
            )
        _mtp_fp_lock = getattr(self, "_fast_path_lock", None)
        if _mtp_fp_lock is not None:
            with _mtp_fp_lock:
                self._active_fast_path_count += 1
        try:
            while True:
                # Check cancel_event from consumer side
                if _is_cancelled(cancel_event):
                    # Yield terminal stop chunk so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="stop",
                        ttft_ms=_mtp_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=timeout_seconds)
                except TimeoutError:
                    logger.warning(
                        f"MTP streaming timeout: no token for {timeout_seconds}s"
                    )
                    _mtp_timeout_cancel.set()  # Signal GPU loop to stop
                    # Yield terminal output so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="error",
                        error=f"MTP streaming timeout: no token for {timeout_seconds}s",
                        ttft_ms=_mtp_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                if item is _sentinel:
                    break
                if isinstance(item, BaseException):
                    logger.warning(f"MTP streaming error: {item}")
                    # Yield terminal error output so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="error",
                        error=str(item),
                        ttft_ms=_mtp_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                new_text, tok_count, _fr_val, token_id = item
                # Backward compat: _fr_val may be bool or str or None
                if isinstance(_fr_val, bool):
                    finish_reason = "stop" if _fr_val else None
                    done = _fr_val
                else:
                    finish_reason = _fr_val
                    done = _fr_val is not None
                accumulated += new_text
                if len(accumulated) > _MAX_STREAMING_TEXT_BUFFER:
                    logger.error(
                        "Streaming text buffer exceeded 1MB limit (%d bytes) — truncating",
                        len(accumulated),
                    )
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated),
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=tok_count,
                        finished=True,
                        finish_reason="length",
                        error="Streaming text buffer exceeded 1MB limit",
                        cached_tokens=0,
                        reasoning_tokens=_mtp_consumer_thinking_tokens
                        if _mtp_consumer_think_start is not None
                        else 0,
                        ttft_ms=_mtp_ttft_ms_val,
                    )
                    break
                n_tok = tok_count
                # Consumer-side thinking state tracking for reasoning_tokens reporting
                if _mtp_consumer_think_start is not None and isinstance(token_id, int):
                    if (
                        not _mtp_consumer_in_thinking
                        and token_id == _mtp_consumer_think_start
                    ):
                        _mtp_consumer_in_thinking = True
                    elif _mtp_consumer_in_thinking:
                        _mtp_consumer_thinking_tokens += 1
                        if token_id == _mtp_consumer_think_end:
                            _mtp_consumer_in_thinking = False

                # Streaming backpressure: slow down if client can't keep up
                if _backpressure.check_backpressure(_q.qsize()):
                    _delay = _backpressure.get_delay_ms(_q.qsize())
                    if _delay > 0:
                        await asyncio.sleep(_delay / 1000)

                # Record TTFT on first token
                if not _mtp_ttft_recorded and n_tok == 1:
                    _mtp_ttft_recorded = True
                    _mtp_ttft_s = time.perf_counter() - _mtp_gen_t0
                    _mtp_ttft_ms_val = round(_mtp_ttft_s * 1000, 1)
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _mtp_ttft_s,
                            labels={"model_id": self.model_label},
                        )
                    except Exception:
                        logger.debug(
                            "MTP streaming TTFT prometheus recording failed",
                            exc_info=True,
                        )

                # Build logprobs for this token — MTP uses greedy decoding
                # internally and does not expose per-token logits in the
                # queue-based streaming path.
                # When logprobs=True, return an empty list with a warning so
                # consumers get a valid (but empty) structure instead of None.
                _chunk_logprobs = None
                if logprobs:
                    _chunk_logprobs: list = []
                    if n_tok <= 1:
                        logger.debug(
                            "MTP streaming path: real logprobs unavailable "
                            "(per-token logits not exposed via queue). "
                            "Returning empty logprobs list."
                        )

                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                    reasoning_tokens=_mtp_consumer_thinking_tokens
                    if _mtp_consumer_think_start is not None
                    else 0,
                    cached_tokens=0,
                    logprobs=_chunk_logprobs,
                    ttft_ms=_mtp_ttft_ms_val,
                )
                if done:
                    break
        finally:
            _unregister_inflight()
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed in MTP streaming finally", exc_info=True
                    )
            # Decrement active fast path count (prevents model eviction mid-generation)
            _mtp_fp_lock = getattr(self, "_fast_path_lock", None)
            if _mtp_fp_lock is not None:
                with _mtp_fp_lock:
                    self._active_fast_path_count -= 1
            if not future.done():
                future.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await future
            # Drain queue to unblock any pending call_soon_threadsafe from
            # the executor thread, preventing GPU work from continuing after
            # the consumer has stopped iterating.
            while not _q.empty():
                try:
                    _q.get_nowait()
                except asyncio.QueueEmpty:
                    break


from .text_utils import cache_tokenizer_vocab as _cache_tokenizer_vocab
