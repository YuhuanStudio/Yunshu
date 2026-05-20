from __future__ import annotations
"""Yunshu BatchedEngine — user-facing continuous batching engine (oMLX pattern).

Studied from oMLX's engine/batched.py, written from scratch:
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

import asyncio
import logging
import os
import platform
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


def _is_cancelled(event: Any) -> bool:
    """Thread-safe cancel check. Works from the MLX executor thread.

    asyncio.Event.is_set() reads ._value (GIL-protected bool), which is
    safe from any thread in CPython.  Using the explicit attribute avoids
    the thread-safety warning from calling asyncio APIs off the event loop.
    """
    if event is None:
        return False
    if isinstance(event, asyncio.Event):
        return event._value
    return event.is_set()

_REASONING_EFFORT_MAP = {"low": 2048, "medium": 8192, "high": 32768}


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
    """Output from generation (oMLX GenerationOutput pattern)."""
    text: str = ""
    new_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finished: bool = False
    finish_reason: Optional[str] = None
    cached_tokens: int = 0
    logprobs: list[dict] | None = None
    ttft_ms: float = 0.0
    reasoning_tokens: int = 0
    current_state: Optional[str] = None  # "reasoning" or "normal" — matches RequestOutput
    error: Optional[str] = None  # Error message if generation failed
    prefill_progress: tuple[int, int] | None = None  # (processed, total) during chunked prefill


_PROGRESSIVE_QUANT_INTERVAL = 256


def _create_prompt_cache_with_quant(model, kv_quant_bits: int | None = None, kv_quant_group_size: int = 64):
    """KV-5: Create a prompt cache with optional immediate KV quantization.

    When kv_quant_bits is set (4 or 8), wraps each KV cache layer in a
    QuantizedKVCache so quantization is applied from the first token rather
    than only after progressive quantization kicks in. This provides maximum
    memory savings for long-context generation.

    When kv_quant_bits is None, falls back to standard make_prompt_cache.
    """
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)

    if kv_quant_bits is not None and kv_quant_bits < 16:
        try:
            for i, c in enumerate(cache):
                if hasattr(c, 'to_quantized'):
                    cache[i] = c.to_quantized(group_size=kv_quant_group_size, bits=kv_quant_bits)
        except (ImportError, Exception) as e:
            logger.debug(f"QuantizedKVCache wrapping failed, using standard cache: {e}")

    return cache


def _wrap_custom_logits_processor(proc):
    """SAMP-2: Wrap a user-provided logits processor to adapt its signature.

    User-provided processors follow the vLLM convention:
        (token_ids: list[int], logits: mx.array) -> mx.array

    But mlx-lm's generate_step passes (tokens: mx.array, logits: mx.array).
    This wrapper converts mx.array tokens → list[int] before calling the user processor.
    """
    def _wrapped(tokens_mx, logits):
        token_ids = [int(t) for t in tokens_mx]
        return proc(token_ids, logits)
    return _wrapped


def _maybe_quantize_kv_cache(
    prompt_cache: list,
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int,
) -> None:
    """Quantize KV cache layers that have exceeded the start threshold.

    Follows mlx-lm's maybe_quantize_kv_cache pattern from generate.py:299.
    Uses MLX's native cache.to_quantized() for hardware-efficient 4/8-bit
    KV storage, reducing memory footprint by 2-4x for long sequences.
    """
    if kv_bits is None:
        return
    for i, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and hasattr(c, "offset"):
            if c.offset >= quantized_kv_start:
                prompt_cache[i] = c.to_quantized(
                    group_size=kv_group_size, bits=kv_bits
                )


def _progressive_quantize_kv_cache(
    prompt_cache: list,
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int,
    current_token_count: int,
    interval: int = _PROGRESSIVE_QUANT_INTERVAL,
) -> None:
    """Progressively quantize KV cache during generation (C6 pattern).

    Called every N tokens during the generate loop to keep memory
    usage flat instead of peaking at full precision. Quantizes only
    newly eligible layers since last quantization pass.
    """
    if kv_bits is None or current_token_count % interval != 0:
        return
    _maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits)


def _store_thinking_segment(ids, thinking_tokens: list[int], thinking_store, kv_cache=None) -> None:
    """Store a thinking segment KV for future reuse."""
    try:
        import hashlib as _hl
        conv_id = _hl.sha256(str([int(t) for t in ids[:16]]).encode()).hexdigest()[:16]
        # Snapshot to avoid sharing mutable reference with prefix_cache
        _kv_snapshot = [c for c in kv_cache] if kv_cache else None
        thinking_store.store(
            conversation_id=conv_id,
            thinking_tokens=thinking_tokens,
            context_tokens=[int(t) for t in ids],
            kv_data=_kv_snapshot,
        )
    except Exception:
        logger.debug("thinking segment store failed", exc_info=True)


def _build_constrained_sampler(sampler, json_schema, tokenizer):
    """Build a constrained sampler from a grammar specification.

    Handles:
    - JSON schema dict → JsonSchemaConstraint
    - "json_object" string → generic JSON constraint
    - {"type": "regex", "pattern": "..."} → RegexConstraint
    - {"type": "choice", "choices": [...]} → ChoiceConstraint
    - {"type": "cfg", "grammar": "..."} → LarkGrammarConstraint

    When YUNSHU_GRAMMAR_BITMASK=1 is set, uses the bitmask engine instead
    of the allowlist-based ConstrainedSampler (xgrammar-style approach).
    """
    # Determine grammar type and payload
    grammar_type = None
    grammar = None

    if isinstance(json_schema, dict) and json_schema.get("type") in ("regex", "choice", "cfg"):
        grammar_type = json_schema["type"]
        if grammar_type == "regex":
            grammar = json_schema.get("pattern", "")
        elif grammar_type == "choice":
            grammar = json_schema.get("choices", [])
        elif grammar_type == "cfg":
            grammar = json_schema.get("grammar", "")
        else:
            return sampler
    else:
        grammar_type = "json_schema"
        grammar = json_schema

    # ── Bitmask path (YUNSHU_GRAMMAR_BITMASK=1) ──
    try:
        from .grammar_bitmask import is_bitmask_enabled, build_bitmask_engine, BitmaskConstrainedSampler
        if is_bitmask_enabled():
            try:
                engine = build_bitmask_engine(grammar_type, grammar)
                return BitmaskConstrainedSampler(sampler, engine, tokenizer)
            except Exception:
                logger.debug("bitmask engine setup failed, falling back to allowlist", exc_info=True)
    except ImportError:
        logger.debug("grammar_bitmask module not available, using allowlist path", exc_info=True)

    # ── Standard allowlist path ──
    if grammar_type in ("regex", "choice", "cfg"):
        from .grammar_constraint import ConstraintFactory
        try:
            constraint = ConstraintFactory.create(grammar_type, grammar, tokenizer)
            from .json_schema import ConstrainedSampler
            return ConstrainedSampler(sampler, constraint, tokenizer)
        except Exception:
            logger.debug("grammar constraint setup failed, returning unconstrained sampler", exc_info=True)
            return sampler

    # Standard JSON schema path
    from .json_schema import JsonSchemaConstraint, ConstrainedSampler
    if isinstance(json_schema, str):
        if json_schema == "json_object":
            # Generic JSON object mode — no specific schema
            constraint = JsonSchemaConstraint(None)
        else:
            import json as _json
            schema = _json.loads(json_schema)
            constraint = JsonSchemaConstraint(schema)
    else:
        constraint = JsonSchemaConstraint(json_schema)
    return ConstrainedSampler(sampler, constraint, tokenizer)


class BatchedEngine:
    """User-facing continuous batching engine (oMLX BatchedEngine pattern).

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

        self._model = None
        self._tokenizer = None
        self._engine_core = None
        self._loaded = False
        self._starting = False  # Guard against concurrent start() calls
        self._kv_manager = None  # Set from EngineCore._kv_manager after start

        # Fast-path active request tracking (prevents model eviction mid-generation)
        self._active_fast_path_count = 0
        self._fast_path_lock = threading.Lock()

        # ── Wave 42: Wired production modules ──
        # Model preprocessor registry (auto-detects model family for multimodal input)
        from .model_preprocessor import PreprocessorRegistry
        self._preprocessor_registry = PreprocessorRegistry()

        # Speculative decoding state (Phase 4)
        self._spec_decoder = None  # SpeculativeDecoder instance
        self._spec_enabled = False

        # MTP speculative decoding (built-in multi-token prediction heads)
        self._mtp_decoder = None  # MTPDecoder instance
        self._mtp_strategy = None  # MTPStrategy wrapper

        # N-gram proposer for model-free speculative decoding
        self._ngram_proposer = None  # NgramProposer, created on demand
        self._ngram_stats = {"proposals": 0, "accepted": 0, "total_draft": 0}

        # Response cache hit/miss counters (YUNSHU_RESPONSE_CACHE=1)
        self._response_cache_hits = 0
        self._response_cache_misses = 0

        # GPU-accelerated rejection sampling (opt-in via YUNSHU_GPU_REJECTION=1)
        from .gpu_rejection import GPURejectionSampler, should_enable_gpu_rejection
        self._gpu_rejection_sampler = GPURejectionSampler()
        self._gpu_rejection_enabled = should_enable_gpu_rejection()

        # Spec draft verifier: production-grade draft verification with
        # KV cache trimming and bonus token emission
        from .spec_draft_verifier import SpecDraftVerifier
        self._spec_draft_verifier = SpecDraftVerifier(track_stats=True)

        # Adaptive speculative decode controller (opt-in via YUNSHU_ADAPTIVE_SPEC=1)
        self._adaptive_spec = None

        # Lookahead reasoning: boosts spec decode during <think/> blocks
        from .speculative_decoder import LookaheadReasoning
        self._lookahead_reasoning = LookaheadReasoning()

        # Medusa speculative decoding (multi-head prediction on hidden state)
        self._medusa_proposer = None  # MedusaProposer instance
        self._medusa_strategy = None  # MedusaStrategy wrapper

        # SpecPrefill config (opt-in via YUNSHU_SPEC_PREFILL env var)
        self._spec_prefill_enabled = False
        self._spec_prefill_threshold = 8192
        self._spec_prefill_keep_rate = 0.20
        self._spec_prefill_draft_model = None

        # KV prefix cache for multi-turn speedup
        from .kv_prefix_cache import KVPrefixCache
        self._kv_prefix_cache = KVPrefixCache(max_entries=64, min_prefix_length=32)

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
        import os
        if os.environ.get("YUNSHU_SSD_CACHE", "").strip() in ("1", "true", "yes"):
            ssd_dir = os.environ.get("YUNSHU_SSD_CACHE_DIR", "~/.cache/yunshu/kv-ssd")
            ssd_max_gb = int(float(os.environ.get("YUNSHU_SSD_CACHE_MAX_GB", "10")))
            self._kv_prefix_cache.enable_ssd_cache(
                cache_dir=ssd_dir,
                max_size_bytes=ssd_max_gb * 1024 ** 3,
                model_name=model_name,
            )

        # KV cache quantization config (mlx-lm pattern: to_quantized)
        # Enable via YUNSHU_KV_QUANT_BITS=4 or 8 (MLX only supports these values)
        _qbits = os.environ.get("YUNSHU_KV_QUANT_BITS")
        if _qbits:
            _qbits_int = int(_qbits)
            if _qbits_int not in (2, 3, 4, 8):
                logger.warning(
                    "CONFIG: YUNSHU_KV_QUANT_BITS=%s is not a supported value "
                    "(MLX supports 2, 3, 4, 8). Ignoring.",
                    _qbits,
                )
                self._kv_quant_bits: int | None = None
            else:
                self._kv_quant_bits = _qbits_int
        else:
            self._kv_quant_bits = None
        self._kv_quant_group_size: int = int(
            os.environ.get("YUNSHU_KV_QUANT_GROUP_SIZE", "64")
        )
        self._kv_quant_start: int = int(
            os.environ.get("YUNSHU_KV_QUANT_START", "0")
        )

        # KV Transfer client — distributed KV cache transfer between nodes.
        # In single-node mode, used for KV serialization/persistence.
        # Enable via YUNSHU_KV_TRANSFER=1 for distributed prefill/decode.
        self._kv_transfer_client = None
        self._kv_transfer_stats = {
            "blocks_transferred": 0,
            "bytes_transferred": 0,
            "transfer_failures": 0,
        }
        if os.environ.get("YUNSHU_KV_TRANSFER", "").strip() in ("1", "true", "yes"):
            from .kv_transfer import KVTransferClient, KVTransferConfig
            _kv_transfer_cfg = KVTransferConfig.from_env()
            self._kv_transfer_client = KVTransferClient(_kv_transfer_cfg)
            logger.info(
                "KV transfer client initialized (remote=%s:%d)",
                _kv_transfer_cfg.remote_host, _kv_transfer_cfg.remote_port,
            )

        # Memory pressure eviction config (vllm-mlx pattern)
        # Stored as a percentage (0-100) for consistency with
        # KVPrefixCache.evict_under_pressure().  Converted to a 0-1
        # fraction when calling KVCacheManager.memory_pressure_evict().
        self._mem_pressure_threshold = float(
            os.environ.get("YUNSHU_MEM_PRESSURE_THRESHOLD", "85.0")
        )
        # If the value looks like a fraction (<=1.0), convert to percentage
        if 0 < self._mem_pressure_threshold <= 1.0:
            self._mem_pressure_threshold *= 100.0

        # DeltaNet state inversion for KV cache eviction recovery
        # Enable via YUNSHU_DELTANET_INVERSION=1 — when KV blocks are evicted
        # from the prefix cache, analytically invert the SSM recurrence so
        # the evicted context can be partially recovered (75x less overhead
        # than checkpoint/restore). Works with GatedDeltaNet models (Qwen3.5).
        self._deltanet_inverter = None
        self._deltanet_inversion_enabled = os.environ.get(
            "YUNSHU_DELTANET_INVERSION", ""
        ).strip() in ("1", "true", "yes")
        self._deltanet_inversion_stats = {
            "evictions_captured": 0,
            "inversions_attempted": 0,
            "inversions_succeeded": 0,
            "states_stored": 0,
        }

        # Per-model settings (loaded from model_settings.json + env vars)
        self._settings = None

        # Thinking Segment KV Substore — reasoning token KV cache reuse
        # Enable via YUNSHU_THINKING_CACHE=1
        self._thinking_store = None
        self._total_reasoning_tokens = 0
        if os.environ.get("YUNSHU_THINKING_CACHE", "").strip() in ("1", "true", "yes"):
            from yunshu_kv.thinking_segment import ThinkingSegmentSubstore, ThinkingSegmentConfig
            _thinking_cfg = ThinkingSegmentConfig(
                max_segments_per_conversation=int(
                    os.environ.get("YUNSHU_THINKING_MAX_PER_CONV", "10")
                ),
                max_total_segments=int(
                    os.environ.get("YUNSHU_THINKING_MAX_TOTAL", "1000")
                ),
                min_tokens_to_cache=int(
                    os.environ.get("YUNSHU_THINKING_MIN_TOKENS", "32")
                ),
                ttl_seconds=float(
                    os.environ.get("YUNSHU_THINKING_TTL", "3600")
                ),
                enable_ssd=os.environ.get(
                    "YUNSHU_THINKING_SSD", ""
                ).strip() in ("1", "true", "yes"),
                ssd_cache_dir=os.environ.get("YUNSHU_THINKING_SSD_DIR", ""),
                enable_compression=os.environ.get(
                    "YUNSHU_THINKING_COMPRESS", ""
                ).strip() in ("1", "true", "yes"),
            )
            self._thinking_store = ThinkingSegmentSubstore(_thinking_cfg)

        # LoRA adapter manager (vLLM pattern)
        self._lora_manager = None

        # mx.compile() for Metal kernel caching (SGLang CUDA Graphs equivalent)
        self._compiled = False
        _mx_compile_env = os.environ.get("YUNSHU_MX_COMPILE", "").strip().lower()
        if _mx_compile_env in ("1", "true", "yes"):
            self._use_compile = True
        elif _mx_compile_env in ("0", "false", "no"):
            self._use_compile = False
        else:
            # Auto-enable mx.compile on Apple Silicon — it's always safe
            self._use_compile = platform.system() == "Darwin"

        # Metal kernel manager for custom GPU kernels (paged attention, GEMV, KIVI)
        # Enable via YUNSHU_METAL_KERNELS=1 — provides Metal-accelerated attention,
        # quantized GEMV, and KIVI 2-bit KV cache compression kernels.
        self._metal_kernel_manager = None
        _metal_env = os.environ.get("YUNSHU_METAL_KERNELS", "").strip().lower()
        if _metal_env in ("1", "true", "yes"):
            self._metal_kernels_enabled = True
        elif _metal_env in ("0", "false", "no"):
            self._metal_kernels_enabled = False
        else:
            # Auto-enable Metal kernels on Apple Silicon when metallib is built
            self._metal_kernels_enabled = (
                platform.system() == "Darwin"
                and os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "metal", "default.metallib"))
            )

        # Engine loop default (continuous batching mode)
        # Enable via YUNSHU_ENGINE_LOOP=1 for multi-user concurrent serving
        self._engine_loop_default = os.environ.get(
            "YUNSHU_ENGINE_LOOP", ""
        ).strip() in ("1", "true", "yes")

        # Streaming optimizer pipeline components (C18 pattern)
        # Enable via YUNSHU_STREAMING_PIPELINE=1 for pipelined GPU/CPU overlap
        self._streaming_pipeline_enabled = os.environ.get(
            "YUNSHU_STREAMING_PIPELINE", ""
        ).strip() in ("1", "true", "yes")

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
        to decide whether ``start()`` needs to be called.  The legacy
        Engine class has the same property; BatchedEngine was missing it,
        which caused an ``AttributeError`` in single-engine fallback mode.
        """
        return self._loaded and (
            self._engine_core is not None and self._engine_core.is_running
        )

    async def start(self) -> None:
        """Load model and start EngineCore (oMLX BatchedEngine.start pattern)."""
        if self._loaded:
            return

        # Guard against concurrent start() calls from multiple coroutines.
        # Without this, two concurrent generate() calls that both see
        # _loaded=False can race and both load the model simultaneously,
        # doubling memory usage and causing model ref leaks.
        if getattr(self, '_starting', False):
            # Another coroutine is already starting — wait for it
            import asyncio
            while getattr(self, '_starting', False):
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
            # Check for quantization override from env or settings
            qconfig = os.environ.get("YUNSHU_QUANT_CONFIG")
            if qconfig:
                kwargs["quantization"] = qconfig
            return load_model(self.model_name, **kwargs)

        self._model, self._tokenizer = await loop.run_in_executor(executor, _load)

        try:
            await self._finish_start(loop, executor)
            self._loaded = True
        except Exception:
            # Partial init: clean up model that was loaded but subsystems failed
            logger.error("BatchedEngine start failed after model load, cleaning up", exc_info=True)
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
            from .model_optimizations import RoPEScalingOptimizer, AttentionOptimizer, MoEEfficiencyOptimizer
            rope_opt = RoPEScalingOptimizer()
            rope_opt.configure(self._model)
            attn_opt = AttentionOptimizer()
            attn_opt.detect_attention_type(self._model)
            moe_opt = MoEEfficiencyOptimizer()
            # Auto-detect MoE config from model
            moe_num_experts = 0
            moe_top_k = 0
            if hasattr(self._model, 'config'):
                cfg = self._model.config
                moe_num_experts = getattr(cfg, 'num_experts', getattr(cfg, 'num_local_experts', 0))
                moe_top_k = getattr(cfg, 'num_experts_per_tok', getattr(cfg, 'num_selected_experts', 0))
            if moe_num_experts > 0 and moe_top_k > 0:
                moe_opt.configure(self._model, moe_num_experts, moe_top_k)
            logger.info(
                f"Model optimizations detected: RoPE={rope_opt.get_scaling_config().scaling_type}, "
                f"Attention={attn_opt.get_stats().get('attention_type', 'unknown')}, "
                f"MoE={moe_opt.get_stats().get('num_experts', 0)} experts"
            )
        except Exception:
            logger.warning("Model optimization detection skipped", exc_info=True)

        # Apply n_confirmed patch for GatedDeltaNet SSM layers (Qwen3.5)
        # Enables zero-cost reject in MTP: restore_rollback instead of refeed
        try:
            from .n_confirmed_patch import apply_n_confirmed_patch
            if apply_n_confirmed_patch():
                logger.info("n_confirmed patch applied — zero-cost SSM rollback ready")
        except Exception:
            logger.debug("n_confirmed patch skipped", exc_info=True)

        # Apply MTP monkey-patch for Qwen3.5 models
        # Injects mtp_forward() and make_mtp_cache() into model classes
        try:
            from .mtp_patch import apply_mtp_patch
            if apply_mtp_patch():
                logger.info("MTP patch applied — mtp_forward() available")
        except Exception:
            logger.warning("MTP patch skipped", exc_info=True)

        # Load per-model settings from model_settings.json + env overrides
        self._load_model_settings()

        # Detect model's cache types for type-aware KV management
        try:
            from yunshu_kv.model_cache_config import ModelCacheConfig
            from mlx_lm.models.cache import make_prompt_cache
            test_cache = make_prompt_cache(self._model)
            self._cache_config = ModelCacheConfig.build_from_cache(test_cache)
            logger.info(
                f"Cache config: {self._cache_config.num_layers} layers, "
                f"{self._cache_config.sliceable_count} sliceable"
            )
        except Exception as e:
            logger.debug(f"Cache type detection skipped: {e}")
            self._cache_config = None

        # Initialize DeltaNet inversion for KV eviction recovery
        # Registers capture hooks on SSM layers and wires the pre-eviction
        # callback into the KV prefix cache so evicted states are inverted.
        if self._deltanet_inversion_enabled and self._model is not None:
            self._init_deltanet_inversion()

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
            use_compile = self._use_compile and not self._compiled
            result = mgr.warmup(self._model, model_type=model_type, compile=use_compile)
            logger.info(
                f"Model warmup: {result.warmup_time_s:.3f}s, "
                f"compile={result.compile_cached}, prompts={result.prompts_warmed}"
            )
            # Also do basic warmup to ensure MX compile cache is populated
            _warmup()

        await loop.run_in_executor(executor, _model_warmup)

        # mx.compile() for Metal kernel caching (SGLang CUDA Graphs equivalent)
        # Compiles the model's forward pass into optimized Metal kernels.
        # Only enabled via YUNSHU_MX_COMPILE=1 env var.
        if self._use_compile and not self._compiled:
            try:
                import mlx.core as mx

                def _compile_model():
                    mx.compile(self._model)

                await loop.run_in_executor(executor, _compile_model)
                self._compiled = True
                logger.info("Model compiled with mx.compile() — Metal kernels cached")
            except Exception as e:
                logger.warning(f"mx.compile() failed ({e}), continuing without compilation")

        # Initialize speculative decoding if model supports it (Phase 4)
        self._init_spec_decode()

        # Initialize LoRA adapter manager
        self._init_lora()

        # Warm prompt prefill: pre-populate KV cache with common system prompts
        await self._warm_prompt_prefill()

        # Auto-start EngineCore for continuous batching (default production path)
        # Previously was lazy-loaded only when use_engine_loop=True.
        # Now always starts so the scheduler is ready for concurrent requests.
        use_fast_only = os.environ.get("YUNSHU_FAST_PATH_ONLY", "").strip() in ("1", "true", "yes")
        if not use_fast_only:
            try:
                await self._ensure_engine_core()
                logger.info("EngineCore auto-started — continuous batching ready")
            except Exception as e:
                logger.warning(f"EngineCore auto-start failed ({e}), falling back to fast-path only")
                self._engine_core = None

        # KV-5: Apply auto-tuner KV quantization recommendation if available
        self._apply_auto_tuner_kv_quant()

        # Initialize Metal kernel manager for custom GPU kernels
        # (paged attention, GEMV, KIVI 2-bit KV compression)
        # Enable via YUNSHU_METAL_KERNELS=1 env var.
        if self._metal_kernels_enabled:
            self._init_metal_kernels()

    def _init_metal_kernels(self) -> None:
        """Initialize Metal kernel manager and wire into pipeline components.

        Creates a MetalKernelManager singleton, pre-compiles all kernels,
        and wires it into the EngineCore scheduler for attention/KV operations.
        Also stores reference on self for stats exposure.
        """
        try:
            from .metal_kernels import get_kernel_manager
            mgr = get_kernel_manager()
            mgr.load_default_library()
            self._metal_kernel_manager = mgr
            logger.info(
                "Metal kernels initialized: paged_attention, gemv_fp16, "
                "gemv_q4, kivi_quantize, kivi_dequantize"
            )

            # Wire into EngineCore scheduler for batch-path usage
            if self._engine_core is not None:
                self._engine_core.set_metal_kernel_manager(mgr)
                logger.info("Metal kernel manager wired into EngineCore scheduler")

        except Exception as e:
            logger.warning(f"Metal kernel init failed ({e}), continuing without custom kernels")
            self._metal_kernel_manager = None

    def _init_deltanet_inversion(self) -> None:
        """Initialize DeltaNet state inversion for KV cache eviction recovery.

        Creates a DeltaNetInverter, registers capture hooks on the model's
        SSM layers, and wires a pre-eviction callback into KVPrefixCache.
        When KV blocks are evicted under memory pressure, the callback
        analytically inverts the SSM recurrence to recover the pre-eviction
        state — 75x less overhead than checkpoint/restore.

        Only activates for models with SSM layers (GatedDeltaNet, e.g. Qwen3.5).
        Controlled via YUNSHU_DELTANET_INVERSION=1 env var.
        """
        try:
            from .deltanet_inversion import DeltaNetInverter
            self._deltanet_inverter = DeltaNetInverter()

            # Register capture hooks on any SSM layers the model has
            has_ssm = any(
                hasattr(m, 'state')
                for _, m in self._model.named_modules()
            )
            if has_ssm:
                self._deltanet_inverter.register_hooks(self._model)
                logger.info(
                    "DeltaNet inversion hooks registered — SSM eviction recovery enabled"
                )
            else:
                logger.info(
                    "DeltaNet inversion enabled but no SSM layers found — "
                    "inversion will run only on explicit invert_evicted_state() calls"
                )

            # Wire pre-eviction callback into KV prefix cache
            # Use a weak reference to avoid a reference cycle:
            # engine -> _kv_prefix_cache -> _pre_evict_callback -> engine
            if self._kv_prefix_cache is not None:
                import weakref
                _weak_self = weakref.ref(self)
                def _on_evict(prompt_tokens, cache):
                    strong = _weak_self()
                    if strong is not None:
                        strong._on_prefix_cache_eviction(prompt_tokens, cache)
                self._kv_prefix_cache._pre_evict_callback = _on_evict
                logger.info("DeltaNet eviction callback wired into KV prefix cache")

        except Exception as e:
            logger.warning(
                f"DeltaNet inversion init failed ({e}), continuing without SSM recovery"
            )
            self._deltanet_inverter = None

    def _on_prefix_cache_eviction(self, _prompt_tokens, cache) -> None:
        """Pre-eviction callback: capture DeltaNet state before KV cache is dropped.

        Called by KVPrefixCache._remove_entry() when an entry is evicted.
        Scans the cache layers for SSM state tensors and captures their
        inverted form so the evicted context can be partially recovered.
        """
        if self._deltanet_inverter is None:
            return
        self._deltanet_inversion_stats["evictions_captured"] += 1

        # Attempt to extract and invert SSM states from cache layers.
        # Standard KVCache layers are skipped — only SSM (DeltaNet) layers
        # with a .state attribute are candidates for inversion.
        if not isinstance(cache, list):
            return

        for layer in cache:
            state = getattr(layer, 'state', None)
            if state is None:
                continue
            # If the inverter has captured entries (from model hooks),
            # invert them to recover pre-step state.
            try:
                self._deltanet_inversion_stats["inversions_attempted"] += 1
                self._deltanet_inverter.start_capture()
                recovered = self._deltanet_inverter.invert_all()
                if recovered:
                    self._deltanet_inversion_stats["inversions_succeeded"] += 1
                    self._deltanet_inversion_stats["states_stored"] += len(recovered)
                    logger.debug(
                        f"DeltaNet inversion: recovered {len(recovered)} layer states "
                        f"from evicted cache"
                    )
            except Exception:
                logger.debug("DeltaNet inversion failed for evicted layer", exc_info=True)

    def invert_evicted_state(self, _prompt_tokens: list[int] | None = None) -> list:
        """Trigger DeltaNet state inversion for evicted or current SSM states.

        Manually triggers inversion of captured SSM intermediates. Useful for
        testing or for recovering state after programmatic eviction.

        Args:
            prompt_tokens: Optional prompt tokens to identify which cache entry
                to invert. If None, inverts the last captured state.

        Returns:
            List of recovered mx.array states (one per SSM layer), or empty
            list if inversion is disabled or no state is available.
        """
        if self._deltanet_inverter is None:
            logger.debug("DeltaNet inversion not enabled — invert_evicted_state() is no-op")
            return []

        try:
            self._deltanet_inversion_stats["inversions_attempted"] += 1
            results = self._deltanet_inverter.invert_all()
            if results:
                self._deltanet_inversion_stats["inversions_succeeded"] += 1
                self._deltanet_inversion_stats["states_stored"] += len(results)
            return results
        except Exception:
            logger.debug("DeltaNet invert_evicted_state failed", exc_info=True)
            return []

    def _load_model_settings(self):
        """Load per-model settings from model directory and apply to engine."""
        from .model_settings import load_model_settings
        model_path = ""
        if self._model is not None:
            config = getattr(self._model, 'config', None)
            if config is not None:
                if isinstance(config, dict):
                    model_path = config.get("_name_or_path", self.model_name)
                else:
                    model_path = getattr(config, '_name_or_path', self.model_name)
        self._settings = load_model_settings(model_path or self.model_name, self.model_name)
        self._apply_settings()

    def _apply_settings(self):
        """Apply loaded ModelSettings to engine config."""
        if self._settings is None:
            return
        s = self._settings
        if s.kv_cache_quant_bits is not None:
            self._kv_quant_bits = s.kv_cache_quant_bits
        if s.kv_cache_quant_group_size != 64:
            self._kv_quant_group_size = s.kv_cache_quant_group_size
        if s.kv_cache_quant_start_layer != 0:
            self._kv_quant_start = s.kv_cache_quant_start_layer
        if not s.prefix_cache_enabled:
            self._kv_prefix_cache = None
        if s.spec_decode_enabled:
            self._spec_enabled = True
        if s.spec_prefill_enabled:
            self._spec_prefill_enabled = True
            self._spec_prefill_threshold = s.spec_prefill_threshold
            self._spec_prefill_keep_rate = s.spec_prefill_keep_rate
        if s.ssd_cache_enabled:
            if self._kv_prefix_cache is not None:
                self._kv_prefix_cache.enable_ssd_cache(
                    cache_dir=s.ssd_cache_dir,
                    max_size_bytes=s.ssd_cache_max_gb * 1024 ** 3,
                    model_name=self.model_name,
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

    def _apply_auto_tuner_kv_quant(self) -> None:
        """KV-5: Apply auto-tuner's kv_quantization_bits recommendation.

        Checks the AutoTuner's current params for KV quantization recommendation
        and updates the engine's _kv_quant_bits if the auto-tuner suggests
        quantization that isn't already configured. This enables adaptive KV
        quantization based on observed memory pressure.

        Priority: env var > model settings > auto_tuner recommendation.
        """
        # Only apply if KV quantization isn't already explicitly configured
        if self._kv_quant_bits is not None:
            return  # Already configured via env var or model settings

        # Check EngineCore's auto-tuner if available
        if self._engine_core is not None:
            auto_tuner = getattr(self._engine_core, '_auto_tuner', None)
            if auto_tuner is not None:
                recommended_bits = auto_tuner.params.kv_quantization_bits
                if recommended_bits < 16:  # 16 means no quantization
                    self._kv_quant_bits = recommended_bits
                    logger.info(
                        f"KV-5: Auto-tuner recommends {recommended_bits}-bit KV quantization"
                    )

    def _init_lora(self):
        """Initialize LoRA adapter manager after model load."""
        max_loras = int(os.environ.get("YUNSHU_MAX_LORAS", "4"))
        from .lora_manager import LoRAAdapterManager, set_lora_manager
        self._lora_manager = LoRAAdapterManager(max_loras=max_loras)
        self._lora_manager.set_base_model(self._model)
        set_lora_manager(self._lora_manager)

        # Auto-discover adapters in model directory
        model_path = ""
        config = getattr(self._model, 'config', None)
        if config is not None:
            if isinstance(config, dict):
                model_path = config.get("_name_or_path", "")
            else:
                model_path = getattr(config, '_name_or_path', "")
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
        from .mlx_executor import get_mlx_executor
        from .engine_core import EngineCore, EngineCoreConfig

        executor = get_mlx_executor()
        arch_kwargs = self._extract_model_arch(self._model)

        # Sarathi-style hybrid chunked prefill (opt-in via YUNSHU_HYBRID_PREFILL=1)
        hybrid_prefill = os.environ.get(
            "YUNSHU_HYBRID_PREFILL", ""
        ).strip() in ("1", "true", "yes")
        hybrid_chunk = int(os.environ.get("YUNSHU_HYBRID_CHUNK_SIZE", "512"))

        # External prefill (opt-in via YUNSHU_EXTERNAL_PREFILL=1)
        # Enables memory preflight checks, chunked progress tracking,
        # and mid-prefill abort before BatchGenerator.insert().
        external_prefill = os.environ.get(
            "YUNSHU_EXTERNAL_PREFILL", ""
        ).strip() in ("1", "true", "yes")
        prefill_chunk_size = int(
            os.environ.get("YUNSHU_PREFILL_CHUNK_SIZE", "2048")
        )

        self._engine_core = EngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=EngineCoreConfig(
                stream_interval=self.stream_interval,
                enable_hybrid_prefill=hybrid_prefill,
                hybrid_chunk_size=hybrid_chunk,
                use_external_prefill=external_prefill,
                prefill_chunk_size=prefill_chunk_size,
                **arch_kwargs,
            ),
            executor=executor,
        )
        self._engine_core.scheduler.config.model_name = self.model_name

        # Propagate paged KV manager for memory pressure eviction in fast paths
        self._kv_manager = self._engine_core._kv_manager

        # Setup memory guard with model dimensions
        try:
            model_cfg = getattr(self._model, 'config', self._model) if self._model else None
            if model_cfg is not None:
                num_layers = getattr(model_cfg, 'num_hidden_layers', 0)
                num_kv_heads = getattr(model_cfg, 'num_key_value_heads', 0)
                head_dim = getattr(model_cfg, 'hidden_size', 0) // max(getattr(model_cfg, 'num_attention_heads', 1), 1)
                num_attn_heads = getattr(model_cfg, 'num_attention_heads', None)
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
            model_cfg = getattr(self._model, 'config', self._model) if self._model else None
            if model_cfg is not None:
                num_layers = getattr(model_cfg, 'num_hidden_layers', 0)
                if num_layers > 0:
                    quant_bits = getattr(model_cfg, 'kv_cache_quant_bits', None)
                    quant_start = int(os.environ.get(
                        "YUNSHU_TURBOQUANT_START_LAYER",
                        getattr(model_cfg, 'kv_cache_quant_start_layer', 0),
                    ))
                    quant_group = int(os.environ.get(
                        "YUNSHU_TURBOQUANT_GROUP_SIZE",
                        getattr(model_cfg, 'kv_cache_quant_group_size', 64),
                    ))
                    # Enable if model has quant settings or env opt-in
                    env_enable = os.environ.get("YUNSHU_TURBOQUANT", "").strip() in ("1", "true", "yes")
                    if quant_bits is not None or env_enable:
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
            model_cfg = getattr(self._model, 'config', self._model) if self._model else None
            if model_cfg is not None and self._model is not None:
                num_layers = getattr(model_cfg, 'num_hidden_layers', 0)
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
        # may not have one if ngram_spec_enabled was False in EngineCoreConfig)
        if self._ngram_proposer is not None and scheduler._ngram_proposer is None:
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
        self._ngram_proposer = None
        self._adaptive_spec = None
        self._mtp_decoder = None
        self._mtp_strategy = None
        self._medusa_proposer = None
        self._medusa_strategy = None
        self._warm_prompts = None
        self._thinking_store = None
        self._metal_kernel_manager = None

        # Unregister DeltaNet inversion hooks to restore original class methods
        if self._deltanet_inverter is not None:
            self._deltanet_inverter.unregister_hooks()
            self._deltanet_inverter = None

        # Stop KV transfer client (close network connections)
        # NOTE: must await directly, not run_until_complete (we are already
        # inside an async context so run_until_complete would crash).
        if self._kv_transfer_client is not None:
            try:
                await self._kv_transfer_client.stop()
            except Exception:
                logger.debug("KV transfer client stop failed", exc_info=True)
            self._kv_transfer_client = None

        # Stop LoRA adapter manager (unload all adapters, release weights)
        if self._lora_manager is not None:
            try:
                self._lora_manager.shutdown()
            except Exception:
                logger.debug("LoRA manager cleanup failed", exc_info=True)
            self._lora_manager = None
            from .lora_manager import set_lora_manager
            set_lora_manager(None)

        # 3. Release model + tokenizer refs, then GC + clear MLX cache
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._compiled = False

        import gc
        gc.collect()

        from .mlx_executor import get_mlx_executor, shutdown_mlx_executor
        loop = asyncio.get_running_loop()
        import mlx.core as mx

        def _cleanup():
            mx.synchronize()
            mx.clear_cache()

        try:
            await loop.run_in_executor(get_mlx_executor(), _cleanup)
        except RuntimeError:
            logger.debug("MLX executor cleanup skipped (executor already shut down)")
        shutdown_mlx_executor(wait=False)
        logger.info(f"BatchedEngine stopped: {self.model_name}")

    def _should_use_engine_loop(self, use_engine_loop: bool | None) -> bool:
        """Determine whether to route through EngineCore continuous batching.

        Auto-detects: if EngineCore has active requests, prefer the batch
        path for better throughput under concurrency. Single-request fast
        path is preferred for latency when no other requests are pending.
        """
        if use_engine_loop is not None:
            return use_engine_loop
        if getattr(self, '_engine_loop_default', False):
            return True
        # If a fast-path request is already in-flight, route to engine loop
        # for continuous batching instead of serializing on the MLX executor.
        if getattr(self, '_active_fast_path_count', 0) > 0:
            return True
        # Auto-detect: switch to batch path when concurrency is detected
        if self._engine_core is not None and self._engine_core.has_active_requests:
            return True
        return False

    # ── Non-generative tasks: embeddings, pooling ────────────────────────────

    def embed(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        """Generate embeddings for the given texts.

        Uses the model's forward pass to extract hidden states, applies mean
        pooling, and optionally L2-normalizes the result (default: True, matching
        the OpenAI embeddings API contract).

        Runs synchronously — callers in async contexts should wrap with
        ``await loop.run_in_executor(get_mlx_executor(), engine.embed, texts)``.
        """
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Engine not loaded — call start() first")

        import mlx.core as mx

        embeddings: list[list[float]] = []
        for text in texts:
            tokens = self._tokenizer.encode(text)
            if not tokens:
                # Return zero vector of the model's hidden size
                hidden_size = self._get_hidden_size()
                embeddings.append([0.0] * hidden_size)
                continue

            input_ids = mx.array([tokens])
            output = self._model(input_ids)

            # Extract hidden states from model output
            hidden = self._extract_hidden_states(output)

            # Mean pooling over sequence dimension
            pooled = mx.mean(hidden, axis=1).squeeze(0)

            if normalize:
                norm = mx.sqrt(mx.sum(pooled * pooled) + 1e-12)
                pooled = pooled / norm

            embeddings.append(pooled.tolist())

        return embeddings

    def pool(self, texts: list[str], pooling_type: str = "MEAN") -> list[list[float]]:
        """Extract pooled hidden states for the given texts.

        Args:
            texts: List of input strings.
            pooling_type: One of "MEAN", "CLS", "LAST".

        Returns:
            List of pooled hidden-state vectors (NOT normalized).
        """
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Engine not loaded — call start() first")

        import mlx.core as mx

        results: list[list[float]] = []
        for text in texts:
            tokens = self._tokenizer.encode(text)
            if not tokens:
                hidden_size = self._get_hidden_size()
                results.append([0.0] * hidden_size)
                continue

            input_ids = mx.array([tokens])
            output = self._model(input_ids)
            hidden = self._extract_hidden_states(output)

            if pooling_type.upper() == "CLS":
                pooled = hidden[0, 0, :]  # batch=0, first token
            elif pooling_type.upper() == "LAST":
                pooled = hidden[0, -1, :]  # batch=0, last token
            else:  # MEAN
                pooled = mx.mean(hidden, axis=1).squeeze(0)  # [1, seq, d] -> [1, d] -> [d]

            results.append(pooled.tolist())

        return results

    def _extract_hidden_states(self, output) -> "mx.array":
        """Extract hidden states from various model output formats.

        MLX models return either:
        - A plain mx.array (the logits or hidden states)
        - A tuple/list where the first element is hidden states
        - An object with .last_hidden_state attribute
        """
        import mlx.core as mx

        if isinstance(output, mx.array):
            # Output shape: [batch, seq_len, hidden_size]
            return output
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state
        # Fallback: try subscript access, otherwise return as-is
        try:
            return output[0]
        except (TypeError, IndexError):
            return output

    def _get_hidden_size(self) -> int:
        """Get the model's hidden dimension size."""
        if hasattr(self._model, 'config'):
            cfg = self._model.config
            for attr in ('hidden_size', 'd_model', 'n_embd', 'embed_dim'):
                val = getattr(cfg, attr, None)
                if val is not None:
                    return val
        # Heuristic: check model layers
        if hasattr(self._model, 'layers') and len(self._model.layers) > 0:
            layer = self._model.layers[0]
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'fc1'):
                return layer.mlp.fc1.weight.shape[1]
        # Default fallback for common models
        return 768

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
        cancel_event: asyncio.Event | None = None,
        priority: int = 0,
        logits_processors: list | None = None,
        timeout_seconds: float | None = None,
        images: list | None = None,
        lora_adapter: str | None = None,
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

        _use_engine_loop = self._should_use_engine_loop(use_engine_loop)

        # Memory guard preflight check
        guard_rejection = self._check_memory_guard(prompt, max_tokens)
        if guard_rejection is not None:
            return guard_rejection

        # Resolve reasoning_effort → thinking_budget if not explicitly set
        if thinking_budget is None and reasoning_effort is not None:
            thinking_budget = _REASONING_EFFORT_MAP.get(reasoning_effort, 8192)
            if enable_thinking is None:
                enable_thinking = True

        # Response cache lookup (YUNSHU_RESPONSE_CACHE=1)
        _rc_hash = None
        if not spec_decode:
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
                    )
                    _rc_hit = await _rc.get(_rc_hash)
                    if _rc_hit is not None:
                        self._response_cache_hits += 1
                        return _rc_hit
                    self._response_cache_misses += 1
            except Exception:
                logger.debug("response cache lookup failed", exc_info=True)

        # ── Wave 43: Context window truncation for long prompts ──
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            try:
                if self._tokenizer and hasattr(self._tokenizer, 'encode'):
                    text = self._apply_chat_template(prompt, enable_thinking)
                    token_count = len(self._tokenizer.encode(text))
                    max_ctx = getattr(self._model, 'max_seq_len', None)
                    if max_ctx is None:
                        max_ctx = getattr(
                            getattr(self._model, 'config', None), 'max_seq_len', None
                        ) or getattr(
                            getattr(self._model, 'args', None), 'max_seq_len', None
                        )
                    # Thinking tokens also consume context window positions —
                    # subtract them from the available prompt budget.
                    _thinking_overhead = thinking_budget if (thinking_budget and enable_thinking) else 0
                    _generation_budget = max_tokens + _thinking_overhead
                    if max_ctx and token_count + _generation_budget > max_ctx:
                        from .context_window import ContextWindowManager
                        ctx_mgr = ContextWindowManager(
                            token_counter=lambda text: len(self._tokenizer.encode(text)),
                        )
                        result = ctx_mgr.compute_truncation(
                            messages=prompt,
                            max_tokens=max_ctx - _generation_budget,
                            strategy="importance_aware",
                        )
                        prompt = result.messages
                        logger.debug(
                            f"Context window truncated: {token_count} → "
                            f"{result.truncated_token_count} tokens (saved {result.tokens_saved})"
                        )
            except Exception:
                logger.warning("context window truncation skipped", exc_info=True)

        # Speculative decoding path (Phase 4: single-request EAGLE-3)
        if spec_decode and self._spec_enabled and self._spec_decoder is not None:
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
            )

        # MTP speculative decoding (built-in multi-token prediction heads)
        if spec_decode and self._mtp_decoder is not None and not _use_engine_loop:
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
            )

        # N-gram speculative decoding (model-free, CPU-based proposal)
        if spec_decode and self._ngram_proposer is not None and not _use_engine_loop:
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
            )
            if _rc_hash is not None and result.finish_reason != "error":
                try:
                    from .gateway_optimizer import get_response_cache
                    await get_response_cache().put(_rc_hash, result)
                except Exception:
                    logger.debug("response cache store failed", exc_info=True)
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
        )

        if result is None:
            return GenerationOutput(finished=True, finish_reason="error", error="engine_core returned None")

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
        _reasoning_tok = getattr(result, 'reasoning_tokens', 0) or 0
        try:
            from .reasoning_parser import get_reasoning_parser
            rp = get_reasoning_parser(self.model_name)
            rp_out = rp.parse(output_text)
            if rp_out.reasoning and rp_out.reasoning_tokens > 0:
                _reasoning_tok = rp_out.reasoning_tokens
        except Exception:
            logger.debug("reasoning_parser failed", exc_info=True)

        # TTFT from engine_core (computed before request cleanup)
        _ttft_ms = getattr(result, 'ttft_ms', 0.0)

        # Record TTFT in Prometheus (consistency with fast path)
        if _ttft_ms > 0:
            try:
                from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
                pm = get_prometheus_metrics()
                pm.observe_histogram("ttft_seconds", _ttft_ms / 1000.0)
            except Exception:
                logger.debug("engine loop TTFT prometheus recording failed", exc_info=True)

        engine_loop_result = GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finished=True,
            finish_reason=finish_reason,
            reasoning_tokens=_reasoning_tok,
            cached_tokens=getattr(result, 'cached_tokens', 0),
            logprobs=getattr(result, 'logprobs', None),
            ttft_ms=_ttft_ms,
            error=getattr(result, 'error', None),
        )
        if _rc_hash is not None and engine_loop_result.finish_reason != "error":
            try:
                from .gateway_optimizer import get_response_cache
                await get_response_cache().put(_rc_hash, engine_loop_result)
            except Exception:
                logger.debug("response cache store failed", exc_info=True)
        return engine_loop_result

    async def _generate_fast(
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
        timeout_seconds: float = 300.0,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        json_schema: dict | str | None = None,
        cancel_event: asyncio.Event | None = None,
        logits_processors: list | None = None,
        priority: int = 0,
    ) -> GenerationOutput:
        """Fast path: run generate_step directly on executor thread.

        Bypasses EngineCore's continuous batching loop for single requests.
        Runs the entire generation in one tight GPU loop on the MLX executor,
        eliminating per-token async round-trip overhead for full GPU utilization.
        Note: priority is accepted for API consistency but has no effect in the
        single-request fast path (no scheduler contention).
        """
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        tokenizer = self._tokenizer
        model = self._model

        # ── Wave 42: Model preprocessor for multimodal input ──
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # Check for multimodal content (images, audio, video)
            has_multimodal = any(
                isinstance(c.get("content"), list)
                for c in prompt
                if isinstance(c.get("content"), list)
            )
            if has_multimodal and self._preprocessor_registry is not None:
                try:
                    model_config = {"model_type": self.model_name or ""}
                    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                        model_config["model_type"] = model.config.model_type
                    preprocessor = self._preprocessor_registry.detect(model_config)
                    if preprocessor is not None:
                        processed = preprocessor.preprocess(prompt, tokenizer)
                        if processed.token_ids:
                            prompt = processed.token_ids
                except Exception:
                    logger.debug("model preprocessor failed, using raw prompt", exc_info=True)

        # Encode prompt
        if isinstance(prompt, str):
            text = prompt
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            tpl_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                tpl_kwargs["enable_thinking"] = enable_thinking
            text = tokenizer.apply_chat_template(prompt, **tpl_kwargs)
        else:
            text = str(prompt)

        input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        # Guard against empty prompt: tokenizer.encode("") may return []
        # which causes generate_step to produce zero tokens. Inject BOS token
        # as a minimal prompt so the model can still generate.
        if not input_ids:
            bos_id = getattr(tokenizer, 'bos_token_id', None)
            if bos_id is not None:
                input_ids = [bos_id]
            else:
                # Use EOS as fallback — the model will likely stop immediately
                # but at least we won't crash with an empty tensor
                eos_id = getattr(tokenizer, 'eos_token_id', 1)
                input_ids = [eos_id]
            prompt_tokens = len(input_ids)

        # ── Context window truncation (fast path) ──
        # The engine-loop path does this at add_request(), but the fast
        # path bypasses that. Truncate from the left to keep the most
        # recent context and leave room for generation tokens.
        _max_ctx = getattr(model, 'max_seq_len', None)
        if _max_ctx is None:
            _max_ctx = getattr(
                getattr(model, 'config', None), 'max_seq_len', None
            ) or getattr(
                getattr(model, 'args', None), 'max_seq_len', None
            )
        if _max_ctx and _max_ctx > 0 and prompt_tokens > _max_ctx:
            _thinking_overhead = thinking_budget if (thinking_budget and enable_thinking) else 0
            _generation_budget = max_tokens + _thinking_overhead
            _allowed = max(1, _max_ctx - _generation_budget)
            _original_prompt_tokens = prompt_tokens
            input_ids = input_ids[-_allowed:]
            prompt_tokens = len(input_ids)
            logger.warning(
                "Fast path prompt truncated to fit context window: %d → %d tokens "
                "(max_seq_len=%d, generation_budget=%d)",
                _original_prompt_tokens, prompt_tokens, _max_ctx, _generation_budget,
            )

        stop_ids = set()
        if hasattr(tokenizer, 'eos_token_id'):
            stop_ids.add(tokenizer.eos_token_id)
        if hasattr(tokenizer, 'eos_token_ids'):
            stop_ids.update(tokenizer.eos_token_ids)

        # Pre-encode stop strings for suffix matching
        stop_suffixes = []
        if stop:
            for s in stop:
                ids = tokenizer.encode(s)
                if len(ids) == 1:
                    stop_ids.add(ids[0])
                else:
                    stop_suffixes.append(s)
        if stop_token_ids:
            stop_ids.update(stop_token_ids)

        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else 0,
            min_p=min_p,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
        )

        # JSON Schema / grammar constraint: wrap sampler with ConstrainedSampler
        if json_schema is not None:
            try:
                sampler = _build_constrained_sampler(sampler, json_schema, tokenizer)
            except Exception:
                logger.warning("Grammar constraint setup failed, falling back to unconstrained", exc_info=True)

        # SAMP-2: Save user-provided custom logits processors before building internal list
        _custom_logits_processors = logits_processors or []
        logits_processors = []
        if repetition_penalty != 1.0:
            def _rep_penalty(tokens, logits, rp=repetition_penalty, ctx=20):
                if len(tokens) > 0:
                    recent = tokens[-ctx:]
                    import mlx.core as _mx
                    sel = logits[..., recent]
                    sel = _mx.where(sel < 0, sel * rp, sel / rp)
                    logits = logits.at[..., _mx.array(recent)].set(sel)
                return logits
            logits_processors.append(_rep_penalty)
        if frequency_penalty != 0.0 or presence_penalty != 0.0:
            def _freq_pres_penalty(tokens, logits, fp=frequency_penalty, pp=presence_penalty, n_prompt=prompt_tokens):
                import mlx.core as _mx
                gen_tokens = tokens[n_prompt:] if len(tokens) > n_prompt else []
                counts = {}
                for t in gen_tokens:
                    counts[int(t)] = counts.get(int(t), 0) + 1
                for tid, cnt in counts.items():
                    if fp > 0:
                        logits = logits.at[..., tid].set(logits[..., tid] - fp * cnt)
                    if pp > 0 and cnt > 0:
                        logits = logits.at[..., tid].set(logits[..., tid] - pp)
                return logits
            logits_processors.append(_freq_pres_penalty)
        if logit_bias:
            def _logit_bias_proc(_tokens, logits, biases=logit_bias):
                import mlx.core as _mx
                for tid, bias in biases.items():
                    logits = logits.at[..., tid].set(logits[..., tid] + bias)
                return logits
            logits_processors.append(_logit_bias_proc)

        # SAMP-2: Wrap user-provided custom logits processors to adapt signature.
        # User processors take (token_ids: list[int], logits: mx.array) -> mx.array
        # but generate_step passes (tokens: mx.array, logits: mx.array).
        if _custom_logits_processors:
            logits_processors.extend(_wrap_custom_logits_processor(p) for p in _custom_logits_processors)

        # Generate inflight request ID outside the closure so it is
        # accessible in the outer exception handlers below.  Previously
        # this was defined inside _run() which caused a NameError when
        # an exception fired before the executor ran the closure.
        _inflight_req_id = f"fp-{id(generate_step)}-{int(time.monotonic()*1e6)}"

        def _run():
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            tokens = []
            token_logprobs = []
            ttft_s = 0.0
            cached_tokens = 0
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            thinking_tokens_used = 0
            think_end_token = None
            think_start_token = None
            _in_thinking = False
            _thinking_tokens: list[int] = []
            _stopped_by_suffix = False
            _stopped_by_stop_id = False

            # Prefill progress tracking for the fast path
            _prefill_req_id = f"fp-{id(_run)}-{int(time.monotonic()*1e6)}"
            _prefill_tracker = None
            try:
                from .prefill_progress import get_prefill_tracker
                _prefill_tracker = get_prefill_tracker()
                _prefill_tracker.update(_prefill_req_id, 0, prompt_tokens, self.model_name or "default")
            except Exception:
                logger.debug("prefill tracker setup failed", exc_info=True)
                _prefill_tracker = None

            if thinking_budget is not None or enable_thinking:
                # Only use token ID matching when "<think"/"</think" encode to
                # a SINGLE token.  Multi-token encodings mean encode(...)[-1]
                # picks a random last token, causing false-positive state transitions.
                try:
                    _ts_ids = tokenizer.encode("<think")
                    _te_ids = tokenizer.encode("</think")
                    if len(_ts_ids) == 1 and len(_te_ids) == 1:
                        think_start_token = _ts_ids[0]
                        think_end_token = _te_ids[0]
                    else:
                        think_start_token = think_end_token = None
                except Exception:
                    logger.debug("thinking token encode failed", exc_info=True)
                    think_start_token = think_end_token = None

            # Prompt cache: try exact-match KV lookup by messages hash
            _pc_hit = False
            if hasattr(self, '_prompt_cache') and self._prompt_cache is not None:
                try:
                    from .prompt_cache import compute_messages_hash
                    _pc_hash = compute_messages_hash(
                        [{"role": "user", "content": text}],
                        model=self.model_name,
                    )
                    _pc_entry = self._prompt_cache.lookup(_pc_hash)
                    if _pc_entry is not None and _pc_entry.kv_state is not None:
                        cache = _pc_entry.kv_state
                        _pc_hit = True
                        cached_tokens = _pc_entry.token_count
                        logger.debug(
                            f"Prompt cache hit: hash={_pc_hash[:12]}, "
                            f"tokens={cached_tokens}"
                        )
                except Exception:
                    logger.debug("prompt cache lookup failed", exc_info=True)

            # Try KV prefix cache hit (skip when prompt cache already hit —
            # prompt cache provides full KV state which is always better)
            prefix_cache = self._kv_prefix_cache
            # Proactive memory pressure eviction (vllm-mlx pattern)
            if prefix_cache is not None and self._mem_pressure_threshold > 0:
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
                # Also evict from paged KV manager when enabled
                if self._kv_manager is not None:
                    try:
                        self._kv_manager.memory_pressure_evict(
                            self._mem_pressure_threshold / 100.0
                        )
                    except Exception:
                        logger.debug("paged KV pressure eviction failed", exc_info=True)
            if not _pc_hit:
                cached_kv, _, matched = (prefix_cache.get(ids) if prefix_cache is not None else (None, None, 0))
                cache = cached_kv if cached_kv is not None else _create_prompt_cache_with_quant(model, self._kv_quant_bits, self._kv_quant_group_size)
                if cached_kv is not None:
                    cached_tokens = matched
                    ids_to_prefill = ids[matched:]
                else:
                    ids_to_prefill = ids
            else:
                # Prompt cache provided full KV — skip prefix cache lookup.
                # ids_to_prefill = tokens beyond what's already cached (empty
                # for full-match prompt cache, so generate_step starts decode
                # from the last cached position).
                cached_kv = None
                ids_to_prefill = ids[cached_tokens:]

            # Inflight prefix sharing (SGLang pattern): check for in-flight
            # prefills with matching prefix to share partial KV blocks
            _inflight_entry = None
            if cached_kv is None:
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker
                    _tracker = get_inflight_tracker()
                    _inflight_entry = _tracker.find_prefix(
                        [int(t) for t in ids], self.model_name or ""
                    )
                    if _inflight_entry is not None and _inflight_entry.kv_cache_ref is not None:
                        cache = _inflight_entry.kv_cache_ref
                        shared_len = min(len(_inflight_entry.token_ids), len(ids))
                        cached_tokens = shared_len
                        ids_to_prefill = ids[shared_len:]
                        logger.debug(
                            "inflight prefix reuse: %d tokens from req=%s",
                            shared_len, _inflight_entry.request_id[:12],
                        )
                except Exception:
                    logger.debug("inflight prefix lookup failed", exc_info=True)

            # Register our prefill as in-flight for concurrent requests to share
            try:
                from .inflight_prefix_sharing import get_inflight_tracker
                get_inflight_tracker().register(
                    _inflight_req_id,
                    [int(t) for t in ids],
                    cache,
                    self.model_name or "",
                )
            except Exception:
                logger.debug("inflight prefix register failed", exc_info=True)

            # Thinking segment KV lookup — reuse reasoning KV from prior turns
            if self._thinking_store is not None and enable_thinking:
                try:
                    import hashlib as _hl
                    _conv_id = _hl.sha256(str(ids[:16]).encode()).hexdigest()[:16]
                    _conv_segs = self._thinking_store.get_conversation_segments(_conv_id)
                    if _conv_segs:
                        _best = max(_conv_segs, key=lambda s: s.last_accessed)
                        if _best.kv_data is not None:
                            cached_tokens += _best.num_tokens
                            logger.debug(
                                f"Thinking KV reuse: {_conv_id} → "
                                f"{_best.num_tokens} tokens, hash={_best.step_hash}"
                            )
                except Exception:
                    logger.debug("Thinking segment lookup failed", exc_info=True)

            gen_t0 = time.perf_counter()
            _timeout_deadline = gen_t0 + timeout_seconds
            first = True
            _itl_samples: list[float] = []
            _last_tok_time = 0.0
            _lprocs = logits_processors if logits_processors else None
            spec_prefill_done = False

            # SpecPrefill: sparse prefill for long prompts
            if (self._spec_prefill_enabled
                and self._spec_prefill_draft_model is not None
                and len(ids_to_prefill) >= self._spec_prefill_threshold):
                from .spec_prefill import (
                    score_tokens, select_chunks, sparse_prefill, cleanup_rope,
                )
                try:
                    importance = score_tokens(self._spec_prefill_draft_model, ids_to_prefill)
                    selected = select_chunks(importance, keep_pct=self._spec_prefill_keep_rate)
                    logits = sparse_prefill(model, ids_to_prefill, selected, cache)
                    mx.eval(logits)
                    ttft_s = time.perf_counter() - gen_t0
                    first = False
                    first_token = int(sampler(logits[:, -1:, :]))
                    tokens.append(first_token)
                    if logprobs:
                        log_probs = mx.log(mx.softmax(logits[:, -1, :].astype(mx.float32), axis=-1))
                        tok_lp = float(log_probs[0, first_token])
                        token_logprobs.append({"token_id": first_token, "logprob": tok_lp})
                    if first_token in stop_ids:
                        tokens.pop()
                        _stopped_by_stop_id = True
                    else:
                        detokenizer.add_token(first_token)
                        # Check stop suffix on first_token (was missing)
                        if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                            tokens.pop()
                            _stopped_by_suffix = True
                        else:
                            # Check if first_token starts a thinking segment
                            if think_start_token is not None and first_token == think_start_token:
                                _in_thinking = True
                                _thinking_tokens = []
                    remaining = max_tokens - 1
                    if remaining > 0 and first_token not in stop_ids and not _stopped_by_suffix:
                        for token, logits in generate_step(
                            mx.array([first_token]).reshape(1, -1), model,
                            max_tokens=remaining, sampler=sampler,
                            prompt_cache=cache, logits_processors=_lprocs,
                        ):
                            tokens.append(token)
                            if logprobs:
                                log_probs = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1))
                                tok_lp = float(log_probs[token])
                                token_logprobs.append({"token_id": int(token), "logprob": tok_lp})
                            if token in stop_ids:
                                tokens.pop()
                                _stopped_by_stop_id = True
                                break
                            detokenizer.add_token(token)
                            if stop_suffixes:
                                if any(detokenizer.text.endswith(s) for s in stop_suffixes):
                                    tokens.pop()  # Exclude suffix-triggering token from count
                                    _stopped_by_suffix = True
                                    break
                            # Cancellation check — after append so the token
                            # is not silently lost (consistent with main loop).
                            if _is_cancelled(cancel_event):
                                mx.synchronize()
                                break
                            # Timeout check (was missing — SpecPrefill could run indefinitely)
                            if len(tokens) % 32 == 0 and time.perf_counter() > _timeout_deadline:
                                logger.warning(f"SpecPrefill generation timed out after {timeout_seconds}s ({len(tokens)} tokens)")
                                break
                            # Track thinking segment boundaries BEFORE budget check
                            # (was missing — thinking mode was non-functional in SpecPrefill)
                            if think_start_token is not None:
                                if not _in_thinking and token == think_start_token:
                                    _in_thinking = True
                                    _thinking_tokens = []
                                elif _in_thinking:
                                    _thinking_tokens.append(token)
                                    if token == think_end_token:
                                        _in_thinking = False
                            # Thinking budget enforcement (same as main loop).
                            # Only force-append think_end_token if the current token
                            # is NOT already the natural closing tag (avoids duplicate).
                            if thinking_budget is not None and _in_thinking:
                                thinking_tokens_used += 1
                                if thinking_tokens_used >= thinking_budget and think_end_token is not None:
                                    _in_thinking = False
                                    if token != think_end_token:
                                        tokens.append(think_end_token)
                                        _thinking_tokens.append(think_end_token)
                                        detokenizer.add_token(think_end_token)
                                    break
                    cleanup_rope(model)
                    spec_prefill_done = True
                except Exception:
                    logger.warning("SpecPrefill failed, falling back to standard prefill", exc_info=True)
                    tokens.clear()
                    detokenizer.reset()
                    cache = _create_prompt_cache_with_quant(model, self._kv_quant_bits, self._kv_quant_group_size)
                    ids_to_prefill = ids
                    first = True

            if not spec_prefill_done:
                _timeout_check_interval = 32
                with _wired_limit_ctx(model):
                    for token, logits in generate_step(
                        ids_to_prefill, model, max_tokens=max_tokens, sampler=sampler,
                        prompt_cache=cache, logits_processors=_lprocs,
                    ):
                        if first:
                            ttft_s = time.perf_counter() - gen_t0
                            first = False
                            # Prefill complete — remove from progress tracker
                            if _prefill_tracker is not None:
                                _prefill_tracker.update(
                                    _prefill_req_id, prompt_tokens, prompt_tokens,
                                    self.model_name or "default",
                                )
                            _last_tok_time = time.perf_counter()
                        else:
                            # ITL tracking
                            _tok_now = time.perf_counter()
                            _itl = _tok_now - _last_tok_time
                            _last_tok_time = _tok_now
                            if _itl > 0 and _itl < 10:
                                _itl_samples.append(_itl)
                        tokens.append(token)
                        # Request-level timeout: check every N tokens
                        if len(tokens) % _timeout_check_interval == 0:
                            if time.perf_counter() > _timeout_deadline:
                                logger.warning(f"Generation timed out after {timeout_seconds}s ({len(tokens)} tokens)")
                                break
                        # Progressive KV quantization (C6: keep memory flat during generation)
                        if self._kv_quant_bits is not None:
                            _progressive_quantize_kv_cache(
                                cache, self._kv_quant_start,
                                self._kv_quant_group_size, self._kv_quant_bits,
                                len(tokens),
                            )
                        # Compute logprobs BEFORE stop checks — logprobs for stop
                        # tokens are trimmed later via lp_result[:len(tokens)].
                        if logprobs:
                            import mlx.core as mx
                            log_probs = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1))
                            tok_lp = float(log_probs[token])
                            entry = {"token_id": int(token), "logprob": tok_lp}
                            if top_logprobs and top_logprobs > 0:
                                k = min(top_logprobs, log_probs.shape[0])
                                sorted_idx = mx.argsort(-log_probs)
                                top_k_idx = sorted_idx[:k]
                                entry["top_logprobs"] = [
                                    {"token_id": int(top_k_idx[j]), "logprob": float(log_probs[int(top_k_idx[j])])}
                                    for j in range(k)
                                ]
                            token_logprobs.append(entry)
                        # Stop ID check — must happen before thinking budget so that
                        # a stop token gets finish_reason="stop" even during thinking.
                        if token in stop_ids:
                            tokens.pop()  # Exclude stop token from output
                            _stopped_by_stop_id = True
                            break
                        # Always add token to detokenizer for incremental state
                        # consistency — previously only added when stop_suffixes
                        # was non-empty, leaving the detokenizer empty and its
                        # state stale when no suffix matching was requested.
                        detokenizer.add_token(token)
                        if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                            tokens.pop()  # Exclude suffix-triggering token from count
                            _stopped_by_suffix = True
                            break
                        # Cancellation check
                        if _is_cancelled(cancel_event):
                            mx.synchronize()
                            break
                        # Track thinking segment boundaries BEFORE budget check
                        # so that a natural </think token is detected first and
                        # the budget enforcement does not append a duplicate.
                        if think_start_token is not None:
                            if not _in_thinking and token == think_start_token:
                                _in_thinking = True
                                _thinking_tokens = []
                                self._lookahead_reasoning.check_thinking_state_text("<think")
                            elif _in_thinking:
                                _thinking_tokens.append(token)
                                if token == think_end_token:
                                    _in_thinking = False
                                    self._lookahead_reasoning.check_thinking_state_text("</think")

                        # Thinking budget enforcement: cap thinking tokens.
                        # Only force-append think_end_token if the current token
                        # is NOT already the natural closing tag (avoids duplicate).
                        if thinking_budget is not None and _in_thinking:
                            thinking_tokens_used += 1
                            if thinking_tokens_used >= thinking_budget and think_end_token is not None:
                                _in_thinking = False
                                # Add forced closing tag to tokens so it appears in
                                # tokenizer.decode(tokens) output (non-streaming path).
                                # Also add to detokenizer so detokenizer.text stays
                                # consistent with the tokens list — previously the
                                # forced tag was only in tokens[], causing
                                # detokenizer.text to miss it when suffix matching
                                # chose the detokenizer path for output assembly.
                                if token != think_end_token:
                                    tokens.append(think_end_token)
                                    _thinking_tokens.append(think_end_token)
                                    detokenizer.add_token(think_end_token)
                                break

            # Cache the completed KV state for future prefix matching
            # Quantize cache layers to save memory (mlx-lm pattern)
            if self._kv_quant_bits is not None:
                _maybe_quantize_kv_cache(
                    cache, self._kv_quant_start,
                    self._kv_quant_group_size, self._kv_quant_bits,
                )
            prefix_cache.add(ids, cache)

            # KV Transfer: serialize and send KV blocks to remote decode node.
            # In single-node mode, this is a no-op (client is None).
            if self._kv_transfer_client is not None:
                try:
                    from .kv_transfer import extract_kv_blocks_from_cache
                    blocks = extract_kv_blocks_from_cache(
                        cache,
                        [int(t) for t in ids],
                    )
                    if blocks:
                        result = self._kv_transfer_client.send_blocks_sync(
                            blocks=blocks,
                            model_name=self.model_name,
                            total_tokens=len(ids),
                            layer_count=len(cache),
                        )
                        if result.status.value == "completed":
                            self._kv_transfer_stats["blocks_transferred"] += result.blocks_transferred
                            self._kv_transfer_stats["bytes_transferred"] += result.bytes_transferred
                        else:
                            self._kv_transfer_stats["transfer_failures"] += 1
                except Exception:
                    logger.debug("KV transfer send failed", exc_info=True)

            # Prompt cache: store KV state for exact-match reuse
            if not _pc_hit and hasattr(self, '_prompt_cache') and self._prompt_cache is not None:
                try:
                    from .prompt_cache import compute_messages_hash
                    _pc_hash = compute_messages_hash(
                        [{"role": "user", "content": text}],
                        model=self.model_name,
                    )
                    self._prompt_cache.store(
                        _pc_hash, cache,
                        token_count=len(ids) + len(tokens),
                    )
                except Exception:
                    logger.debug("prompt cache store failed", exc_info=True)

            # Store thinking segment KV for future reuse (if enabled)
            if _thinking_tokens and self._thinking_store is not None:
                try:
                    import hashlib as _hl
                    conv_id = _hl.sha256(str(ids[:16]).encode()).hexdigest()[:16]
                    # Snapshot the cache so the thinking store doesn't hold a
                    # reference to the same mutable list as prefix_cache.
                    _thinking_kv = [c for c in cache] if cache else None
                    self._thinking_store.store(
                        conversation_id=conv_id,
                        thinking_tokens=_thinking_tokens,
                        context_tokens=[int(t) for t in ids],
                        kv_data=_thinking_kv,
                    )
                except Exception:
                    logger.debug("Thinking segment store failed", exc_info=True)

            # Finalize detokenizer to flush any remaining partial UTF-8 bytes
            # before assembling final output text.
            try:
                detokenizer.finalize()
            except Exception:
                logger.debug("detokenizer finalize failed in fast path", exc_info=True)

            # When stop_suffix matching is active, use detokenizer text for
            # output because tokenizer.decode(tokens) may contain a partial
            # suffix that leaked across token boundaries.  The detokenizer
            # has the complete incremental text including the suffix, which
            # we trim below.  When no suffix matching, tokenizer.decode is
            # authoritative and avoids detokenizer state issues.
            if _stopped_by_suffix and stop_suffixes:
                output_text = detokenizer.text
                # Trim the matched suffix from detokenizer text
                for s in stop_suffixes:
                    if output_text.endswith(s):
                        output_text = output_text[:-len(s)]
                        break
                output_text = _clean_special_tokens(output_text)
            else:
                output_text = tokenizer.decode(tokens, skip_special_tokens=True)
            mx.synchronize()

            # Unregister from inflight prefix tracker
            try:
                from .inflight_prefix_sharing import get_inflight_tracker
                get_inflight_tracker().unregister(_inflight_req_id)
            except Exception:
                logger.debug("inflight prefix unregister failed", exc_info=True)

            return tokens, output_text, token_logprobs, ttft_s, cached_tokens, _stopped_by_suffix, _stopped_by_stop_id, _itl_samples, _thinking_tokens

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        _fp_lock = getattr(self, '_fast_path_lock', None)
        if _fp_lock is not None:
            with _fp_lock:
                self._active_fast_path_count += 1
        try:
            try:
                tokens, output_text, token_logprobs, ttft_s, cached_tokens, _stopped_by_suffix, _stopped_by_stop_id, _itl_samples, _thinking_tokens = await loop.run_in_executor(executor, _run)
            except MemoryError:
                logger.warning("OOM during generation — returning memory_limit finish reason")
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker
                    get_inflight_tracker().unregister(_inflight_req_id)
                except Exception:
                    logger.debug("inflight prefix unregister failed in OOM handler", exc_info=True)
                # Clear Metal buffers left behind by the OOM
                try:
                    import mlx.core as _mx
                    await loop.run_in_executor(executor, lambda: (_mx.synchronize(), _mx.clear_cache()))
                except Exception:
                    pass
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error="OOM during generation",
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            except RuntimeError as e:
                if "memory" in str(e).lower() or "out of" in str(e).lower():
                    logger.warning(f"MLX OOM during generation: {e}")
                    try:
                        from .inflight_prefix_sharing import get_inflight_tracker
                        get_inflight_tracker().unregister(_inflight_req_id)
                    except Exception:
                        logger.debug("inflight prefix unregister failed in OOM handler", exc_info=True)
                    # Clear Metal buffers left behind by the OOM
                    try:
                        import mlx.core as _mx
                        await loop.run_in_executor(executor, lambda: (_mx.synchronize(), _mx.clear_cache()))
                    except Exception:
                        pass
                    return GenerationOutput(
                        finished=True,
                        finish_reason="memory_limit",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        error=str(e),
                        ttft_ms=0.0,
                        cached_tokens=0,
                    )
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker
                    get_inflight_tracker().unregister(_inflight_req_id)
                except Exception:
                    logger.debug("inflight prefix unregister failed in error handler", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"Unexpected error during generation: {e}", exc_info=True)
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker
                    get_inflight_tracker().unregister(_inflight_req_id)
                except Exception:
                    logger.debug("inflight prefix unregister failed in error handler", exc_info=True)
                # Re-raise so the gateway can report the error to the client.
                # Only MemoryError and RuntimeError were handled above; anything
                # else is a bug or unexpected condition that should propagate.
                raise

            # Decode token strings for logprobs
            lp_result = None
            if logprobs and token_logprobs:
                for lp_entry in token_logprobs:
                    tid = lp_entry["token_id"]
                    try:
                        lp_entry["token"] = tokenizer.decode([tid])
                        lp_entry["bytes"] = list(lp_entry["token"].encode("utf-8"))
                    except Exception:
                        logger.debug("logprob token decode failed", exc_info=True)
                        lp_entry["token"] = ""
                        lp_entry["bytes"] = []
                    if "top_logprobs" in lp_entry:
                        for tlp in lp_entry["top_logprobs"]:
                            try:
                                tlp["token"] = tokenizer.decode([tlp["token_id"]])
                                tlp["bytes"] = list(tlp["token"].encode("utf-8"))
                            except Exception:
                                logger.debug("top_logprob token decode failed", exc_info=True)
                                tlp["token"] = ""
                                tlp["bytes"] = []
                lp_result = token_logprobs

            output_text = _clean_special_tokens(output_text)

            # Trim stop suffix from output text when matched during generation
            if _stopped_by_suffix and stop_suffixes:
                for s in stop_suffixes:
                    if output_text.endswith(s):
                        output_text = output_text[:-len(s)]
                        break

            # Determine finish_reason.
            # Priority: cancel > stop (suffix or stop_id) > length
            # When cancel_event or timeout triggers, the loop breaks without
            # setting _stopped_by_suffix or _stopped_by_stop_id, so those
            # tokens correctly show up as "stop" only when genuinely stopped.
            _cancelled = _is_cancelled(cancel_event)
            if _cancelled:
                finish_reason = "stop"
            elif _stopped_by_suffix or _stopped_by_stop_id:
                finish_reason = "stop"
            else:
                finish_reason = "length"

            # BUG FIX: When the first token is a stop_id (SpecPrefill path),
            # it is popped from `tokens` but its logprob entry remains in
            # `token_logprobs`.  Trim the stale entry so logprobs count matches
            # `completion_tokens`.
            if lp_result is not None:
                lp_result = lp_result[:len(tokens)]

            # Record TTFT + ITL in Prometheus
            _ttft_ms_val = round(ttft_s * 1000, 1)
            if ttft_s > 0:
                try:
                    from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
                    pm = get_prometheus_metrics()
                    pm.observe_histogram("ttft_seconds", ttft_s)
                    if cached_tokens > 0:
                        pm.set_gauge("kv_prefix_cache_hits", 1)
                    else:
                        pm.set_gauge("kv_prefix_cache_misses", 1)
                    # ITL: record individual inter-token latency samples into histogram
                    if _itl_samples:
                        for _itl_sample in _itl_samples:
                            pm.observe_histogram("itl_seconds", _itl_sample)
                except Exception:
                    logger.debug("TTFT/ITL prometheus recording failed", exc_info=True)

            # Record in ServerMetrics (consistency with engine loop path)
            try:
                from .server_metrics import get_server_metrics
                _sm = get_server_metrics()
                _total_gen_s = sum(_itl_samples) + ttft_s if _itl_samples else ttft_s
                _sm.record_request_complete(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(tokens),
                    cached_tokens=cached_tokens,
                    prefill_duration=ttft_s,
                    generation_duration=_total_gen_s,
                    model_id=self.model_name,
                )
                if _itl_samples:
                    for _itl in _itl_samples:
                        _sm.record_itl(_itl)
            except Exception:
                logger.debug("ServerMetrics recording failed in fast path", exc_info=True)

            self._total_reasoning_tokens += len(_thinking_tokens)

            # Reasoning parser: supplement token-level tracking with model-specific
            # reasoning extraction when thinking tokens were not explicitly tracked
            _reasoning_tok = len(_thinking_tokens)
            if _reasoning_tok == 0 and output_text:
                try:
                    from .reasoning_parser import get_reasoning_parser
                    rp = get_reasoning_parser(self.model_name)
                    rp_out = rp.parse(output_text)
                    if rp_out.reasoning:
                        _reasoning_tok = rp_out.reasoning_tokens
                        if rp_out.content != output_text:
                            output_text = rp_out.content
                except Exception:
                    logger.debug("reasoning_parser failed in fast path", exc_info=True)

            return GenerationOutput(
                text=output_text,
                new_text=output_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(tokens),
                finished=True,
                finish_reason=finish_reason,
                cached_tokens=cached_tokens,
                logprobs=lp_result,
                ttft_ms=_ttft_ms_val,
                reasoning_tokens=_reasoning_tok,
            )
        finally:
            _fp_lock = getattr(self, '_fast_path_lock', None)
            if _fp_lock is not None:
                with _fp_lock:
                    self._active_fast_path_count -= 1

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
        priority: int = 0,
        logprobs: bool | int = False,
        top_logprobs: int | None = None,
        logits_processors: list | None = None,
        cancel_event: asyncio.Event | None = None,
        timeout_seconds: float | None = None,
        images: list | None = None,
        lora_adapter: str | None = None,
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

        _use_engine_loop = self._should_use_engine_loop(use_engine_loop)
        # Resolve reasoning_effort → thinking_budget if not explicitly set
        if thinking_budget is None and reasoning_effort is not None:
            thinking_budget = _REASONING_EFFORT_MAP.get(reasoning_effort, 8192)
            if enable_thinking is None:
                enable_thinking = True

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
            _tracker = get_request_tracker()
            _active_gen = _tracker.register(_stream_req_id, self.model_name or "")
            _cancel_event = _active_gen.cancel_event
        except Exception:
            logger.debug("request tracker registration failed", exc_info=True)
            _cancel_event = None
            _tracker = None

        # If the gateway passes an external cancel_event, wrap both events
        # so that checking .is_set() on the wrapper detects either source.
        if cancel_event is not None:
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
                        if hasattr(_internal, '_value'):
                            if _internal._value:
                                return True
                        elif _internal.is_set():
                            return True
                    return _is_cancelled(_external)

            _cancel_event = _CompositeCancelEvent()

        # Speculative decoding path (Phase 4)
        if spec_decode and self._spec_enabled and self._spec_decoder is not None:
            try:
                async for output in self._stream_generate_speculative(
                    prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                    logprobs=logprobs, top_logprobs=top_logprobs,
                    stop=stop, stop_token_ids=stop_token_ids,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    json_schema=json_schema,
                    cancel_event=_cancel_event,
                    logits_processors=logits_processors,
                ):
                    yield output
            finally:
                if _tracker is not None:
                    try:
                        _tracker.unregister(_stream_req_id)
                    except Exception:
                        logger.debug("request tracker cleanup failed", exc_info=True)
            return

        # MTP speculative decoding streaming (built-in multi-token prediction)
        if spec_decode and self._mtp_decoder is not None and not _use_engine_loop:
            try:
                async for output in self._stream_generate_mtp(
                    prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                    logprobs=logprobs, top_logprobs=top_logprobs,
                    stop=stop, stop_token_ids=stop_token_ids,
                    seed=seed, cancel_event=_cancel_event,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    timeout_seconds=timeout_seconds or 300.0,
                    json_schema=json_schema,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    logits_processors=logits_processors,
                ):
                    yield output
            finally:
                if _tracker is not None:
                    try:
                        _tracker.unregister(_stream_req_id)
                    except Exception:
                        logger.debug("request tracker cleanup failed", exc_info=True)
            return

        # N-gram speculative decoding streaming (model-free)
        if spec_decode and self._ngram_proposer is not None and not _use_engine_loop:
            try:
                async for output in self._stream_generate_ngram_spec(
                    prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                    stop=stop, stop_token_ids=stop_token_ids, seed=seed,
                    json_schema=json_schema, logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    cancel_event=_cancel_event,
                    timeout_seconds=timeout_seconds or 300.0,
                    logits_processors=logits_processors,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
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
                    prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                    stop=stop, stop_token_ids=stop_token_ids,
                    seed=seed, enable_thinking=enable_thinking,
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
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty, logit_bias=logit_bias,
            stop=stop, stop_token_ids=stop_token_ids,
            seed=seed, json_schema=json_schema,
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
        )

        finished_normally = False
        _first_token = True
        _stream_ttft_ms = 0.0
        try:
            async for output in self._engine_core.stream_outputs(request_id, cancel_event=_cancel_event):
                # Check cancel event (gateway disconnect or internal cancel)
                if _cancel_event is not None and _cancel_event.is_set():
                    logger.debug(f"Cancel event triggered during streaming: {request_id}")
                    # Yield terminal stop chunk so consumer sees finished=True
                    yield GenerationOutput(
                        text="",
                        new_text="",
                        prompt_tokens=getattr(output, 'prompt_tokens', 0) if hasattr(output, 'prompt_tokens') else 0,
                        completion_tokens=getattr(output, 'completion_tokens', 0) if hasattr(output, 'completion_tokens') else 0,
                        finished=True,
                        finish_reason="stop",
                        ttft_ms=_stream_ttft_ms,
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
                    _stream_ttft_ms = round((time.perf_counter() - _stream_t0) * 1000, 1)
                    _ttft_ms = _stream_ttft_ms
                    _first_token = False
                    # Record TTFT in Prometheus (consistency with fast path)
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
                        pm = get_prometheus_metrics()
                        pm.observe_histogram("ttft_seconds", _stream_ttft_ms / 1000.0)
                    except Exception:
                        logger.debug("engine loop streaming TTFT prometheus recording failed", exc_info=True)
                gen_output = GenerationOutput(
                    text=_clean_special_tokens(output.output_text),
                    new_text=cleaned,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=finish_reason,
                    reasoning_tokens=getattr(output, 'reasoning_tokens', 0),
                    cached_tokens=getattr(output, 'cached_tokens', 0),
                    logprobs=getattr(output, 'logprobs', None),
                    ttft_ms=_ttft_ms,
                    current_state=getattr(output, 'current_state', None),
                    error=getattr(output, 'error', None),
                    prefill_progress=getattr(output, 'prefill_progress', None),
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

    async def _stream_generate_fast(
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
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        json_schema: dict | str | None = None,
        cancel_event: asyncio.Event | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        logits_processors: list | None = None,
        priority: int = 0,
        timeout_seconds: float = 300.0,
    ) -> AsyncIterator[GenerationOutput]:
        """Fast streaming: runs generate_step on executor, yields via asyncio.Queue.

        When YUNSHU_STREAMING_PIPELINE=1, wraps generation with:
        - StreamingBackpressureController to prevent OOM on slow clients
        - TokenPipeline for GPU/CPU overlap (future: full pipeline)
        - PrefetchSampler for sampling plan pre-computation (future: per-step)
        """
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        # Streaming optimizer components
        from .streaming_optimizer import StreamingBackpressureController
        _backpressure = StreamingBackpressureController(max_queue_size=100)

        # TokenPipeline for GPU/CPU overlap — activated via YUNSHU_STREAMING_PIPELINE=1
        _pipeline = None
        if self._streaming_pipeline_enabled:
            from .streaming_optimizer import TokenPipeline, PipelineConfig
            _pipeline = TokenPipeline(PipelineConfig(enable_overlap=True))
            _pipeline.start_pipeline(request=None)
            logger.debug("TokenPipeline active for streaming fast path")

        tokenizer = self._tokenizer
        model = self._model

        # Model-specific preprocessing (§16.5 pattern)
        if self._preprocessor_registry is not None:
            try:
                model_config = {"model_type": self.model_name or ""}
                if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                    model_config["model_type"] = model.config.model_type
                preprocessor = self._preprocessor_registry.detect(model_config)
                if preprocessor is not None:
                    processed = preprocessor.preprocess(prompt, tokenizer)
                    if processed.token_ids:
                        prompt = processed.token_ids
            except Exception:
                logger.debug("model preprocessor failed in streaming", exc_info=True)

        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            tpl_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                tpl_kwargs["enable_thinking"] = enable_thinking
            prompt = tokenizer.apply_chat_template(prompt, **tpl_kwargs)

        input_ids = tokenizer.encode(prompt)
        prompt_tokens = len(input_ids)

        # Guard against empty prompt (same as _generate_fast)
        if not input_ids:
            bos_id = getattr(tokenizer, 'bos_token_id', None)
            if bos_id is not None:
                input_ids = [bos_id]
            else:
                eos_id = getattr(tokenizer, 'eos_token_id', 1)
                input_ids = [eos_id]
            prompt_tokens = len(input_ids)

        # ── Context window truncation (streaming fast path) ──
        # The engine-loop path does this at add_request(), but the
        # streaming fast path bypasses that. Truncate from the left to
        # keep the most recent context and leave room for generation.
        _max_ctx = getattr(model, 'max_seq_len', None)
        if _max_ctx is None:
            _max_ctx = getattr(
                getattr(model, 'config', None), 'max_seq_len', None
            ) or getattr(
                getattr(model, 'args', None), 'max_seq_len', None
            )
        if _max_ctx and _max_ctx > 0 and prompt_tokens > _max_ctx:
            _thinking_overhead = thinking_budget if (thinking_budget and enable_thinking) else 0
            _generation_budget = max_tokens + _thinking_overhead
            _allowed = max(1, _max_ctx - _generation_budget)
            _original_prompt_tokens = prompt_tokens
            input_ids = input_ids[-_allowed:]
            prompt_tokens = len(input_ids)
            logger.warning(
                "Streaming fast path prompt truncated to fit context window: %d → %d tokens "
                "(max_seq_len=%d, generation_budget=%d)",
                _original_prompt_tokens, prompt_tokens, _max_ctx, _generation_budget,
            )

        stop_ids = set()
        if hasattr(tokenizer, 'eos_token_id'):
            stop_ids.add(tokenizer.eos_token_id)
        if hasattr(tokenizer, 'eos_token_ids'):
            stop_ids.update(tokenizer.eos_token_ids)

        stop_suffixes = []
        if stop:
            for s in stop:
                ids = tokenizer.encode(s)
                if len(ids) == 1:
                    stop_ids.add(ids[0])
                else:
                    stop_suffixes.append(s)
        if stop_token_ids:
            stop_ids.update(stop_token_ids)

        sampler = make_sampler(
            temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0,
            min_p=min_p, xtc_probability=xtc_probability, xtc_threshold=xtc_threshold,
        )

        # Grammar constraint for streaming fast path
        if json_schema is not None:
            try:
                sampler = _build_constrained_sampler(sampler, json_schema, tokenizer)
            except Exception:
                logger.warning("Grammar constraint setup failed in streaming", exc_info=True)

        # Build logits processors for penalty/bias params
        _custom_logits_processors = logits_processors or []
        logits_processors = []
        if repetition_penalty != 1.0:
            def _repetition_penalty(tokens, logits, rp=repetition_penalty, ctx=20):
                if len(tokens) > 0:
                    recent = tokens[-ctx:]
                    import mlx.core as _mx
                    sel = logits[..., recent]
                    sel = _mx.where(sel < 0, sel * rp, sel / rp)
                    logits = logits.at[..., _mx.array(recent)].set(sel)
                return logits
            logits_processors.append(_repetition_penalty)
        if frequency_penalty != 0.0 or presence_penalty != 0.0:
            def _freq_pres_penalty(tokens, logits, fp=frequency_penalty, pp=presence_penalty, n_prompt=prompt_tokens):
                import mlx.core as _mx
                gen_tokens = tokens[n_prompt:] if len(tokens) > n_prompt else []
                counts = {}
                for t in gen_tokens:
                    counts[int(t)] = counts.get(int(t), 0) + 1
                for tid, cnt in counts.items():
                    if fp > 0:
                        logits = logits.at[..., tid].set(logits[..., tid] - fp * cnt)
                    if pp > 0 and cnt > 0:
                        logits = logits.at[..., tid].set(logits[..., tid] - pp)
                return logits
            logits_processors.append(_freq_pres_penalty)
        if logit_bias:
            def _logit_bias_proc(_tokens, logits, biases=logit_bias):
                import mlx.core as _mx
                for tid, bias in biases.items():
                    logits = logits.at[..., tid].set(logits[..., tid] + bias)
                return logits
            logits_processors.append(_logit_bias_proc)

        # SAMP-2: Wrap user-provided custom logits processors to adapt signature.
        if _custom_logits_processors:
            logits_processors.extend(_wrap_custom_logits_processor(p) for p in _custom_logits_processors)

        # Thread-safe bridge: executor puts via call_soon_threadsafe so the
        # event loop's async consumer is woken for every token.
        _sentinel = object()
        _q: asyncio.Queue = asyncio.Queue(maxsize=512)
        loop = asyncio.get_running_loop()
        # Cross-thread cancel: set by the async consumer on timeout so the
        # GPU generation loop in _run_inner stops producing tokens.
        _timeout_cancel = threading.Event()

        def _put(item):
            # Backpressure-aware queue with retry: slow down producer when
            # queue is nearly full to avoid dropping tokens.  Tokens that
            # are dropped silently corrupt structured output (JSON, tool
            # calls) because the SSE client sees a gap with no error.
            if _q.qsize() > 400:  # 78% of 512
                time.sleep(0.001)  # yield to consumer thread
            # NOTE: We do NOT call _q.get_nowait() here because this
            # function runs on the MLX executor thread (not the asyncio
            # event loop thread).  asyncio.Queue.get_nowait() mutates the
            # internal deque AND calls _wakeup_next() which modifies
            # asyncio.Future objects — neither operation is thread-safe.
            for _attempt in range(10):
                if not _q.full():
                    loop.call_soon_threadsafe(_q.put_nowait, item)
                    return
                if _attempt < 9:
                    time.sleep(0.005)
            # Queue is persistently full — put an error sentinel so the
            # consumer sees finish_reason="error" instead of silently
            # missing content.
            logger.warning(
                "Streaming queue overflow after 50ms — sending error sentinel. "
                "Client will see finish_reason=error."
            )
            try:
                loop.call_soon_threadsafe(
                    _q.put_nowait,
                    Exception("Streaming queue overflow — output truncated"),
                )
            except Exception:
                pass

        # Inflight prefix sharing: defined at _run level so it's accessible
        # from exception handlers even if _run_inner crashes early
        _inflight_req_id = f"fp-s-{int(time.monotonic()*1e6)}"

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker
                get_inflight_tracker().unregister(_inflight_req_id)
            except Exception:
                logger.debug("inflight prefix unregister failed in streaming", exc_info=True)

        _stream_gen_t0 = time.perf_counter()  # TTFT timing for streaming fast path
        _stream_ttft_recorded = [False]  # mutable box to track first-token observation
        _stream_ttft_box = [0.0]  # mutable box for TTFT value
        _stream_itl_samples = []  # ITL samples for streaming fast path

        # Create detokenizer at _run scope so the error handler can finalize
        # it even if _run_inner() crashes before its own cleanup paths run.
        _detokenizer_ref = [None]  # mutable box shared with _run_inner

        def _run_inner():
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            _detokenizer_ref[0] = detokenizer
            n_tok = 0
            thinking_tokens_used = 0
            think_end_token = None
            think_start_token = None
            _first_token = True
            _in_thinking = False
            _thinking_tokens: list[int] = []

            # Prefill progress tracking for streaming fast path
            _prefill_req_id = f"fp-s-{id(_run_inner)}-{int(time.monotonic()*1e6)}"
            _prefill_tracker = None
            try:
                from .prefill_progress import get_prefill_tracker
                _prefill_tracker = get_prefill_tracker()
                _prefill_tracker.update(_prefill_req_id, 0, prompt_tokens, self.model_name or "default")
            except Exception:
                logger.debug("prefill tracker setup failed", exc_info=True)
                _prefill_tracker = None

            if thinking_budget is not None or enable_thinking:
                # Only use token ID matching when "<think"/"</think" encode to
                # a SINGLE token.  Multi-token encodings mean encode(...)[-1]
                # picks a random last token, causing false-positive state transitions.
                try:
                    _ts_ids = tokenizer.encode("<think")
                    _te_ids = tokenizer.encode("</think")
                    if len(_ts_ids) == 1 and len(_te_ids) == 1:
                        think_start_token = _ts_ids[0]
                        think_end_token = _te_ids[0]
                    else:
                        think_start_token = think_end_token = None
                except Exception:
                    logger.debug("thinking token encode failed", exc_info=True)
                    think_start_token = think_end_token = None

            # KV prefix cache for streaming
            prefix_cache = self._kv_prefix_cache
            # Proactive memory pressure eviction (vllm-mlx pattern)
            if prefix_cache is not None and self._mem_pressure_threshold > 0:
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
                # Also evict from paged KV manager when enabled
                if self._kv_manager is not None:
                    try:
                        self._kv_manager.memory_pressure_evict(
                            self._mem_pressure_threshold / 100.0
                        )
                    except Exception:
                        logger.debug("paged KV pressure eviction failed", exc_info=True)
            cached_kv, _, matched = (prefix_cache.get(ids) if prefix_cache is not None else (None, None, 0))
            cache = cached_kv if cached_kv is not None else _create_prompt_cache_with_quant(model, self._kv_quant_bits, self._kv_quant_group_size)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids
            _stream_cached_tokens = matched
            _cached_tokens_box[0] = _stream_cached_tokens

            # Inflight prefix sharing (SGLang pattern)
            _inflight_entry = None
            if cached_kv is None:
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker
                    _tracker = get_inflight_tracker()
                    _inflight_entry = _tracker.find_prefix(
                        [int(t) for t in ids], self.model_name or ""
                    )
                    if _inflight_entry is not None and _inflight_entry.kv_cache_ref is not None:
                        cache = _inflight_entry.kv_cache_ref
                        shared_len = min(len(_inflight_entry.token_ids), len(ids))
                        ids_to_prefill = ids[shared_len:]
                        _stream_cached_tokens = shared_len
                        _cached_tokens_box[0] = shared_len
                except Exception:
                    logger.debug("inflight prefix lookup failed in streaming", exc_info=True)

            # Register our prefill as in-flight for concurrent requests to share
            try:
                from .inflight_prefix_sharing import get_inflight_tracker
                get_inflight_tracker().register(
                    _inflight_req_id,
                    [int(t) for t in ids],
                    cache,
                    self.model_name or "",
                )
            except Exception:
                logger.debug("inflight prefix register failed in streaming", exc_info=True)

            _lprocs = logits_processors if logits_processors else None
            _last_stream_tok_time = 0.0
            with _wired_limit_ctx(model):
                for token, logits in generate_step(
                    ids_to_prefill, model, max_tokens=max_tokens, sampler=sampler,
                    prompt_cache=cache, logits_processors=_lprocs,
                ):
                    n_tok += 1
                    # Check stop_ids BEFORE adding to detokenizer to avoid emitting stop text
                    stop_hit = token in stop_ids
                    suffix_hit = False
                    if not stop_hit:
                        detokenizer.add_token(token)
                        if stop_suffixes:
                            suffix_hit = any(detokenizer.text.endswith(s) for s in stop_suffixes)
                    # Compute per-token logprobs (same pattern as _generate_fast)
                    _lp_entry = None
                    if logprobs and logits is not None:
                        import mlx.core as _mx
                        _log_probs = _mx.log(_mx.softmax(logits.astype(_mx.float32), axis=-1))
                        _tok_lp = float(_log_probs[token])
                        if _tok_lp != _tok_lp or _tok_lp == float('-inf'):
                            _tok_lp = -100.0
                        _lp_entry = {"token_id": int(token), "logprob": _tok_lp}
                        if top_logprobs and top_logprobs > 0:
                            _k = min(top_logprobs, _log_probs.shape[0])
                            _sorted_idx = _mx.argsort(-_log_probs)
                            _top_k_idx = _sorted_idx[:_k]
                            _top_entries = []
                            for j in range(_k):
                                _tlp = float(_log_probs[int(_top_k_idx[j])])
                                if _tlp != _tlp or _tlp == float('-inf'):
                                    _tlp = -100.0
                                _top_entries.append({"token_id": int(_top_k_idx[j]), "logprob": _tlp})
                            _lp_entry["top_logprobs"] = _top_entries
                    # TokenPipeline: submit GPU stages for tracking
                    if _pipeline is not None and _pipeline.is_running:
                        _ptok = _pipeline.submit_stage1_result(logits=None, token_id=int(token))
                        _ptok = _pipeline.submit_stage2_result(_ptok, sampled_id=int(token))
                    # Check cancellation
                    if _is_cancelled(cancel_event):
                        mx.synchronize()
                        # Flush remaining detokenizer bytes before cancelling
                        try:
                            detokenizer.finalize()
                            remaining = detokenizer.last_segment
                            if remaining:
                                _put((remaining, n_tok, None, len(_thinking_tokens), None, "reasoning" if _in_thinking else "normal"))
                        except Exception:
                            logger.debug("detokenizer finalize in cancel handler failed", exc_info=True)
                        # Emit terminal stop chunk so consumer sees finished=True
                        _put(("", n_tok, "stop", len(_thinking_tokens), None, "reasoning" if _in_thinking else "normal"))
                        if _pipeline is not None:
                            _pipeline.finish()
                        if _prefill_tracker is not None:
                            _prefill_tracker.remove(_prefill_req_id)
                        _unregister_inflight()
                        return
                    # Check timeout-driven cancel from consumer
                    if _timeout_cancel.is_set():
                        mx.synchronize()
                        try:
                            detokenizer.finalize()
                            remaining = detokenizer.last_segment
                            if remaining:
                                _put((remaining, n_tok, None, len(_thinking_tokens), None, "reasoning" if _in_thinking else "normal"))
                        except Exception:
                            logger.debug("detokenizer finalize in timeout cancel failed", exc_info=True)
                        _put(("", n_tok, "timeout", len(_thinking_tokens), None, "reasoning" if _in_thinking else "normal"))
                        if _pipeline is not None:
                            _pipeline.finish()
                        if _prefill_tracker is not None:
                            _prefill_tracker.remove(_prefill_req_id)
                        _unregister_inflight()
                        return
                    # Prefill complete on first token — remove from progress tracker
                    if _first_token:
                        _first_token = False
                        # Record TTFT for Prometheus
                        if not _stream_ttft_recorded[0]:
                            _stream_ttft_recorded[0] = True
                            _stream_ttft_box[0] = time.perf_counter() - _stream_gen_t0
                        _last_stream_tok_time = time.perf_counter()
                        if _prefill_tracker is not None:
                            _prefill_tracker.update(
                                _prefill_req_id, prompt_tokens, prompt_tokens,
                                self.model_name or "default",
                            )
                    else:
                        # ITL tracking for streaming fast path
                        _tok_now = time.perf_counter()
                        _itl = _tok_now - _last_stream_tok_time
                        _last_stream_tok_time = _tok_now
                        if _itl > 0 and _itl < 10:
                            _stream_itl_samples.append(_itl)
                    new_text = "" if stop_hit else detokenizer.last_segment
                    if suffix_hit:
                        new_text = ""  # Don't emit suffix text
                    # Exclude stop/suffix-triggering token from completion count
                    if stop_hit or suffix_hit:
                        n_tok -= 1
                    # Track thinking segment boundaries BEFORE budget check
                    # so that a natural </think token is detected first and
                    # the budget enforcement does not append a duplicate.
                    if think_start_token is not None:
                        if not _in_thinking and token == think_start_token:
                            _in_thinking = True
                            _thinking_tokens = []
                            self._lookahead_reasoning.check_thinking_state_text("<think")
                        elif _in_thinking:
                            _thinking_tokens.append(token)
                            if token == think_end_token:
                                _in_thinking = False
                                self._lookahead_reasoning.check_thinking_state_text("</think")
                    # Thinking budget enforcement in streaming.
                    # Only force-append think_end_token + add to detokenizer
                    # if the current token is NOT already the natural closing tag.
                    if thinking_budget is not None and _in_thinking:
                        thinking_tokens_used += 1
                        if thinking_tokens_used >= thinking_budget and think_end_token is not None:
                            # Token was already appended at line ~3298 — don't
                            # duplicate it.  Only force-emit the closing tag.
                            _in_thinking = False
                            # Emit current token's text first (still reasoning content)
                            if new_text:
                                _put((new_text, n_tok, None, len(_thinking_tokens), _lp_entry, "reasoning"))
                            # Only force-emit closing tag if the token isn't already it
                            if token != think_end_token:
                                n_tok += 1  # Count the forced closing tag token
                                detokenizer.add_token(think_end_token)
                                _end_text = detokenizer.last_segment
                                if _end_text:
                                    _put((_end_text, n_tok, None, len(_thinking_tokens), None, "reasoning"))
                            # Store thinking segment before returning
                            if _thinking_tokens and self._thinking_store is not None:
                                _store_thinking_segment(ids, _thinking_tokens, self._thinking_store, kv_cache=cache)
                            detokenizer.finalize()
                            _remaining = detokenizer.last_segment
                            if _remaining:
                                _put((_remaining, n_tok, None, len(_thinking_tokens), None, "normal"))
                            # Final stop chunk — consumer breaks on this
                            _put(("", n_tok, "stop", len(_thinking_tokens), None, "normal"))
                            if _pipeline is not None:
                                _pipeline.finish()
                            prefix_cache.add(ids, cache)
                            mx.synchronize()
                            if _prefill_tracker is not None:
                                _prefill_tracker.remove(_prefill_req_id)
                            _unregister_inflight()
                            return
                    _is_stopping = stop_hit or suffix_hit
                    _cur_state = "reasoning" if _in_thinking else "normal"
                    if _is_stopping:
                        # Emit current text without finish_reason so the consumer
                        # reads it before breaking on done=True below.
                        if new_text:
                            _put((new_text, n_tok, None, len(_thinking_tokens), _lp_entry, _cur_state))
                    else:
                        _put((new_text, n_tok, None, len(_thinking_tokens), _lp_entry, _cur_state))
                    if _is_stopping:
                        # Store thinking segment on stop
                        if _thinking_tokens and self._thinking_store is not None:
                            _store_thinking_segment(ids, _thinking_tokens, self._thinking_store, kv_cache=cache)
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        # Trim stop suffix from remaining text — the suffix may
                        # span multiple tokens, so detokenizer.text still contains
                        # it even after finalize().  Without this, the partial
                        # suffix text leaks into the output.
                        if suffix_hit and stop_suffixes and _remaining:
                            for s in stop_suffixes:
                                if _remaining.endswith(s):
                                    _remaining = _remaining[:-len(s)]
                                    break
                        if _remaining:
                            _put((_remaining, n_tok, None, len(_thinking_tokens), None, _cur_state))
                        # Final stop chunk — consumer breaks on this
                        _put(("", n_tok, "stop", len(_thinking_tokens), None, _cur_state))
                        if _pipeline is not None:
                            _pipeline.finish()
                        prefix_cache.add(ids, cache)
                        mx.synchronize()
                        _unregister_inflight()
                        return
                # Store thinking segment at end of generation
                if _thinking_tokens and self._thinking_store is not None:
                    _store_thinking_segment(ids, _thinking_tokens, self._thinking_store, kv_cache=cache)
                prefix_cache.add(ids, cache)
                detokenizer.finalize()
                remaining = detokenizer.last_segment
                _final_state = "reasoning" if _in_thinking else "normal"
                if remaining:
                    _put((remaining, n_tok, None, len(_thinking_tokens), None, _final_state))
                _put(("", n_tok, "length", len(_thinking_tokens), None, _final_state))
                mx.synchronize()
                # Finish pipeline tracking at end of generation
                if _pipeline is not None:
                    _pipeline.finish()
                # Clean up prefill progress entry (may persist if first token wasn't reached)
                if _prefill_tracker is not None:
                    _prefill_tracker.remove(_prefill_req_id)
                _unregister_inflight()

        def _finalize_detokenizer():
            """Finalize the detokenizer if it was created, to flush internal byte buffers."""
            _dtk = _detokenizer_ref[0]
            if _dtk is not None:
                try:
                    _dtk.finalize()
                except Exception:
                    logger.debug("detokenizer finalize in error handler failed", exc_info=True)

        def _run():
            try:
                _run_inner()
            except (MemoryError, RuntimeError) as e:
                _unregister_inflight()
                _finalize_detokenizer()
                try:
                    import mlx.core as _cleanup_mx
                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                    import gc
                    gc.collect()  # Force GC of KV cache tensors
                except Exception:
                    pass
                if isinstance(e, MemoryError) or "memory" in str(e).lower():
                    logger.warning(f"OOM during streaming: {e}")
                _put(e)
            except Exception as e:
                _unregister_inflight()
                _finalize_detokenizer()
                try:
                    import mlx.core as _cleanup_mx
                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                except Exception:
                    pass
                _put(e)
            finally:
                _put(_sentinel)

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run)

        accumulated = ""
        n_tok = 0
        _reasoning_tokens = 0
        _cached_tokens_box = [0]  # Mutable box for inner _run_inner to set
        _fp_lock = getattr(self, '_fast_path_lock', None)
        if _fp_lock is not None:
            with _fp_lock:
                self._active_fast_path_count += 1
        try:
            while True:
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    logger.warning(f"Streaming fast path timeout: no token for {timeout_seconds}s")
                    _timeout_cancel.set()  # Signal GPU loop to stop
                    # Yield terminal output so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="timeout",
                        error=f"Streaming timeout: no token for {timeout_seconds}s",
                        ttft_ms=round(_stream_ttft_box[0] * 1000, 1) if _stream_ttft_box[0] > 0 else 0.0,
                        cached_tokens=_cached_tokens_box[0],
                        reasoning_tokens=_reasoning_tokens,
                    )
                    break
                if item is _sentinel:
                    break
                if isinstance(item, BaseException):
                    err_msg = str(item).lower()
                    is_oom = isinstance(item, MemoryError) or "memory" in err_msg
                    if is_oom:
                        yield GenerationOutput(
                            text=_clean_special_tokens(accumulated) if accumulated else "",
                            new_text="",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=n_tok,
                            finished=True,
                            finish_reason="memory_limit",
                            error=str(item),
                            ttft_ms=round(_stream_ttft_box[0] * 1000, 1) if _stream_ttft_box[0] > 0 else 0.0,
                            cached_tokens=_cached_tokens_box[0],
                            reasoning_tokens=_reasoning_tokens,
                        )
                    else:
                        # Non-OOM exception: yield terminal error output
                        yield GenerationOutput(
                            text=_clean_special_tokens(accumulated) if accumulated else "",
                            new_text="",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=n_tok,
                            finished=True,
                            finish_reason="error",
                            error=str(item),
                            ttft_ms=round(_stream_ttft_box[0] * 1000, 1) if _stream_ttft_box[0] > 0 else 0.0,
                            cached_tokens=_cached_tokens_box[0],
                            reasoning_tokens=_reasoning_tokens,
                        )
                    break
                if len(item) >= 6:
                    new_text, tok_count, _fr_val, _reasoning_tokens, _lp_entry, _cur_state = item[:6]
                elif len(item) == 5:
                    new_text, tok_count, _fr_val, _reasoning_tokens, _lp_entry = item
                    _cur_state = None
                elif len(item) == 4:
                    new_text, tok_count, _fr_val, _reasoning_tokens = item
                    _lp_entry = None
                    _cur_state = None
                else:
                    new_text, tok_count, _fr_val = item
                    _lp_entry = None
                    _cur_state = None
                # Backward compat: _fr_val may be bool (from old-style tuples)
                # or str ("stop"/"length") or None (intermediate token)
                if isinstance(_fr_val, bool):
                    finish_reason = "stop" if _fr_val else None
                    done = _fr_val
                else:
                    finish_reason = _fr_val  # str or None
                    done = _fr_val is not None
                accumulated += new_text
                n_tok = tok_count

                # TokenPipeline: run stage 3 overlap for stats tracking
                if _pipeline is not None and _pipeline.is_running and _pipeline._current is not None:
                    await _pipeline.next_token(
                        detokenize_fn=lambda _tid, _t=new_text: _t,
                    )

                # Streaming backpressure: slow down if client can't keep up
                if _backpressure.check_backpressure(_q.qsize()):
                    _delay = _backpressure.get_delay_ms(_q.qsize())
                    if _delay > 0:
                        await asyncio.sleep(_delay / 1000)

                # Record TTFT in Prometheus on first token
                if _stream_ttft_recorded[0] and _stream_ttft_box[0] > 0:
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
                        pm = get_prometheus_metrics()
                        pm.observe_histogram("ttft_seconds", _stream_ttft_box[0])
                    except Exception:
                        logger.debug("streaming TTFT prometheus recording failed", exc_info=True)
                    _stream_ttft_recorded[0] = False  # only observe once
                # Attach logprobs to output if computed for this token
                _lp_list = None
                if _lp_entry is not None:
                    # Decode token string for the logprob entry
                    try:
                        _lp_entry["token"] = tokenizer.decode([_lp_entry["token_id"]])
                        _lp_entry["bytes"] = list(_lp_entry["token"].encode("utf-8"))
                    except Exception:
                        logger.debug("logprob token decode failed in streaming", exc_info=True)
                        _lp_entry["token"] = ""
                        _lp_entry["bytes"] = []
                    _lp_list = [_lp_entry]
                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                    reasoning_tokens=_reasoning_tokens,
                    logprobs=_lp_list,
                    cached_tokens=_cached_tokens_box[0],
                    ttft_ms=round(_stream_ttft_box[0] * 1000, 1) if _stream_ttft_box[0] > 0 else 0.0,
                    current_state=_cur_state,
                )
                if done:
                    break
        finally:
            # Record in ServerMetrics for streaming fast path (consistency)
            if n_tok > 0:
                try:
                    from .server_metrics import get_server_metrics
                    _sm = get_server_metrics()
                    _sm.record_request_complete(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        cached_tokens=_cached_tokens_box[0],
                        prefill_duration=_stream_ttft_box[0],
                        generation_duration=(time.perf_counter() - _stream_gen_t0),
                        model_id=self.model_name,
                    )
                    # Record ITL samples from streaming path
                    if _stream_itl_samples:
                        for _itl in _stream_itl_samples:
                            _sm.record_itl(_itl)
                        # Record average ITL in Prometheus
                        try:
                            from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
                            pm = get_prometheus_metrics()
                            avg_itl = sum(_stream_itl_samples) / len(_stream_itl_samples)
                            pm.observe_histogram("itl_seconds", avg_itl)
                        except Exception:
                            logger.debug("streaming ITL prometheus recording failed", exc_info=True)
                except Exception:
                    logger.debug("ServerMetrics recording failed in streaming fast path", exc_info=True)
            # Stop pipeline and log stats
            if _pipeline is not None:
                _pipeline.stop()
                logger.info(
                    "TokenPipeline stats: %s",
                    _pipeline.get_stats(),
                )
            if not future.done():
                future.cancel()
                try:
                    await future
                except (asyncio.CancelledError, Exception):
                    pass
            # Drain remaining queue items to unblock the executor thread's
            # call_soon_threadsafe calls, preventing GPU work from continuing
            # after the consumer has stopped iterating.
            while not _q.empty():
                try:
                    _q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            # Decrement active fast path count
            _fp_lock = getattr(self, '_fast_path_lock', None)
            if _fp_lock is not None:
                with _fp_lock:
                    self._active_fast_path_count -= 1

    async def chat(
        self,
        messages: list[dict],
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
        enable_thinking: bool | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Non-streaming chat completion (messages → template → generate)."""
        prompt = self._apply_chat_template(messages, enable_thinking)
        return await self.generate(
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
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict],
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
        enable_thinking: bool | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Streaming chat completion (messages → template → stream_generate)."""
        prompt = self._apply_chat_template(messages, enable_thinking)
        async for output in self.stream_generate(
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
            **kwargs,
        ):
            yield output

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
        import os
        if os.environ.get("YUNSHU_NGRAM_SPEC", "").strip() not in ("0", "false", "no"):
            from .ngram_proposer import NgramProposer, NgramConfig
            max_n = int(os.environ.get("YUNSHU_NGRAM_MAX_N", "5"))
            k = int(os.environ.get("YUNSHU_NGRAM_K", "5"))
            mode = os.environ.get("YUNSHU_NGRAM_MODE", "lps").strip()
            self._ngram_proposer = NgramProposer(NgramConfig(max_n=max_n, k=k, mode=mode))
            logger.info(f"N-gram proposer initialized: max_n={max_n}, k={k}, mode={mode}")

        # Adaptive spec controller (requires N-gram proposer active)
        if self._ngram_proposer is not None:
            from .adaptive_spec import AdaptiveSpecController
            self._adaptive_spec = AdaptiveSpecController.from_env()

        # SpecPrefill for long prompts (requires YUNSHU_SPEC_PREFILL=1)
        if os.environ.get("YUNSHU_SPEC_PREFILL", "").strip() in ("1", "true", "yes"):
            self._spec_prefill_enabled = True
            self._spec_prefill_threshold = int(os.environ.get("YUNSHU_SPEC_PREFILL_THRESHOLD", "8192"))
            self._spec_prefill_keep_rate = float(os.environ.get("YUNSHU_SPEC_PREFILL_KEEP_RATE", "0.20"))
            logger.info(
                f"SpecPrefill enabled: threshold={self._spec_prefill_threshold}, "
                f"keep_rate={self._spec_prefill_keep_rate}"
            )

        from .speculative_decoder import detect_spec_heads, auto_configure_speculative

        model_config = {}
        config_obj = getattr(self._model, 'config', None) or getattr(self._model, 'args', None)
        if config_obj is not None:
            if hasattr(config_obj, 'to_dict'):
                model_config = config_obj.to_dict()
            elif hasattr(config_obj, '__dict__'):
                model_config = {k: v for k, v in config_obj.__dict__.items()
                                if not k.startswith('_')}

        head_info = detect_spec_heads(model_config)
        if head_info.head_type == "none":
            logger.debug("No speculative decoding heads detected")
            return

        spec_config = auto_configure_speculative(model_config)
        if spec_config.draft_length == 0:
            return

        logger.info(
            f"Speculative decoding available: type={head_info.head_type}, "
            f"draft_length={spec_config.draft_length}"
        )

        # Attempt to load draft model if configured (EAGLE/EAGLE-3 path)
        draft_path = os.environ.get("YUNSHU_DRAFT_MODEL", "").strip()
        if not draft_path and model_config:
            draft_path = model_config.get("draft_model_path", "")
        if draft_path:
            try:
                from mlx_lm.utils import load as load_model
                draft_model, _ = load_model(draft_path)
                from .speculative_decoder import SpeculativeDecoder
                self._spec_decoder = SpeculativeDecoder(
                    self._model, draft_model, self._tokenizer,
                    lookahead=self._lookahead_reasoning,
                )
                self._spec_enabled = True
                logger.info(
                    f"Draft model loaded from {draft_path}: "
                    f"speculative decoding ACTIVE (type={head_info.head_type})"
                )
            except Exception as e:
                logger.warning(f"Draft model load failed ({e}), speculative decoding disabled")

        # MTP path: use built-in multi-token prediction heads (no external draft model)
        # Supported by Qwen3.5, DeepSeek-V3, and other models with MTP layers.
        # Uses n_confirmed=1 for zero-cost reject on GatedDeltaNet SSM layers.
        if head_info.head_type == "mtp" and self._spec_decoder is None:
            try:
                # Load MTP head weights if available
                inner = getattr(self._model, "language_model", self._model)
                if not hasattr(inner, "mtp"):
                    try:
                        from .mtp_patch import load_model_with_mtp
                        model_name_or_path = model_config.get("_name_or_path", self.model_name)
                        self._model = load_model_with_mtp(model_name_or_path)
                        logger.info("MTP head weights loaded from model directory")
                    except FileNotFoundError as e:
                        logger.info(f"MTP weights not found ({e}), using backbone-only MTP")
                    except Exception as e:
                        logger.warning(f"MTP weights load failed ({e}), using backbone-only MTP")

                from .mtp_decoder import MTPDecoder, MTPConfig
                mtp_config = MTPConfig(
                    max_tokens=256,
                    cooldown_on_reject=os.environ.get(
                        "YUNSHU_MTP_COOLDOWN", ""
                    ).strip() in ("1", "true", "yes"),
                    fastmtp_top_k=int(os.environ.get("YUNSHU_MTP_FASTMTP_TOP_K", "0")),
                    use_n_confirmed=True,
                )
                self._mtp_decoder = MTPDecoder(
                    self._model, self._tokenizer, mtp_config,
                )
                from .spec_interface import MTPStrategy
                self._mtp_strategy = MTPStrategy(
                    decoder=self._mtp_decoder, config=mtp_config,
                )
                self._spec_enabled = True
                logger.info(
                    f"MTP decoder initialized: heads={head_info.num_heads}, "
                    f"draft_length={head_info.draft_length}, "
                    f"n_confirmed=True, config={head_info.head_config}"
                )
            except Exception as e:
                logger.warning(f"MTP decoder init failed ({e})")

        # Medusa path: add prediction heads on top of hidden state (no draft model)
        # Enable via YUNSHU_MEDUSA=1 env var. Compatible with ngram via CompositeStrategy.
        if os.environ.get("YUNSHU_MEDUSA", "").strip() in ("1", "true", "yes"):
            try:
                from .medusa_proposer import MedusaProposer, MedusaConfig
                num_heads = int(os.environ.get("YUNSHU_MEDUSA_HEADS", "4"))
                tree_size = int(os.environ.get("YUNSHU_MEDUSA_TREE_SIZE", "5"))
                medusa_config = MedusaConfig(
                    num_heads=num_heads,
                    tree_size=tree_size,
                    enabled=True,
                )
                self._medusa_proposer = MedusaProposer(medusa_config)
                self._medusa_proposer.attach(self._model)
                from .spec_interface import MedusaStrategy
                self._medusa_strategy = MedusaStrategy(
                    proposer=self._medusa_proposer,
                    config=medusa_config,
                )
                self._spec_enabled = True
                logger.info(
                    f"Medusa proposer initialized: heads={num_heads}, "
                    f"tree_size={tree_size}, attached={self._medusa_proposer.is_attached}"
                )
            except Exception as e:
                logger.warning(f"Medusa proposer init failed ({e})")

        # Store config for on-demand decoder creation
        self._spec_config = spec_config
        self._spec_head_info = head_info
        self._spec_enabled = True

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
    ) -> GenerationOutput:
        """Generate using speculative decoding (single-request EAGLE-3 path).

        This path bypasses the continuous batching scheduler and runs the
        SpeculativeDecoder directly. Best for single-request scenarios where
        the draft model can propose K tokens for the target to verify.
        """
        if self._spec_decoder is None:
            # Fall back to standard generation if no decoder
            return await self._generate_fast(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop, stop_token_ids=stop_token_ids, seed=seed,
                enable_thinking=enable_thinking,
                logprobs=logprobs, top_logprobs=top_logprobs,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                json_schema=json_schema,
                cancel_event=cancel_event,
                logits_processors=logits_processors,
                timeout_seconds=timeout_seconds,
            )

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        # Tokenize prompt — handle messages-format (list of dicts) like _generate_fast
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            tpl_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                tpl_kwargs["enable_thinking"] = enable_thinking
            text = self._tokenizer.apply_chat_template(prompt, **tpl_kwargs)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)

        input_ids = self._tokenizer.encode(text)
        import mlx.core as mx

        if seed is not None:
            mx.random.seed(seed)

        input_array = mx.array(input_ids).reshape(1, -1)

        # Build EOS + stop token sets
        eos_ids = set()
        if hasattr(self._tokenizer, 'eos_token_id'):
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
                    logger.debug(f"failed to encode stop sequence: {s!r}", exc_info=True)

        # Run speculative generation on the MLX executor thread
        # Use incremental detokenizer for correct multi-byte UTF-8
        detokenizer = self._tokenizer.detokenizer
        detokenizer.reset()

        def _run_spec():
            token_ids = self._spec_decoder.generate(
                input_ids=input_array,
                max_tokens=max_tokens,
                temperature=temperature,
                cancel_event=cancel_event,
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

        _spec_gen_t0 = time.perf_counter()
        try:
            token_ids, hit_stop = await asyncio.wait_for(
                loop.run_in_executor(executor, _run_spec),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Speculative generation timed out after {timeout_seconds}s")
            try:
                import mlx.core as _mx
                await loop.run_in_executor(executor, lambda: (_mx.synchronize(), _mx.clear_cache()))
            except Exception:
                pass
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error=f"Speculative generation timed out after {timeout_seconds}s",
                ttft_ms=0.0,
            )
        except MemoryError:
            logger.warning("OOM during speculative generation — returning memory_limit finish reason")
            try:
                import mlx.core as _mx
                await loop.run_in_executor(executor, lambda: (_mx.synchronize(), _mx.clear_cache()))
            except Exception:
                pass
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
                    await loop.run_in_executor(executor, lambda: (_mx.synchronize(), _mx.clear_cache()))
                except Exception:
                    pass
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=len(input_ids),
                    completion_tokens=0,
                    error=str(e),
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            raise
        except Exception as e:
            logger.error(f"Unexpected error during speculative generation: {e}", exc_info=True)
            raise
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
                from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
                pm = get_prometheus_metrics()
                pm.observe_histogram("ttft_seconds", _spec_ttft_s)
            except Exception:
                logger.debug("TTFT prometheus recording failed in spec decode path", exc_info=True)

        # Determine finish_reason with cancel awareness
        _cancelled = _is_cancelled(cancel_event)
        if _cancelled:
            _finish_reason = "stop"
        elif hit_stop:
            _finish_reason = "stop"
        elif len(token_ids) < max_tokens:
            _finish_reason = "stop"
        else:
            _finish_reason = "length"

        return GenerationOutput(
            text=text,
            new_text=text,
            prompt_tokens=len(input_ids),
            completion_tokens=len(token_ids),
            finished=True,
            finish_reason=_finish_reason,
            reasoning_tokens=0,
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
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using speculative decoding (single-request path).

        Yields chunks as they are verified by the target model.
        Each yield contains the accepted tokens from one verify step.
        """
        if self._spec_decoder is None:
            # Fall back to fast path streaming (avoid recursive dispatch)
            async for output in self._stream_generate_fast(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop, stop_token_ids=stop_token_ids, seed=seed,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                json_schema=json_schema,
                cancel_event=cancel_event,
                logprobs=bool(logprobs),
                top_logprobs=top_logprobs,
                logits_processors=logits_processors,
            ):
                yield output
            return

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        # Tokenize prompt — handle messages-format (list of dicts) like _generate_fast
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            tpl_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                tpl_kwargs["enable_thinking"] = enable_thinking
            text = self._tokenizer.apply_chat_template(prompt, **tpl_kwargs)
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
        if hasattr(self._tokenizer, 'eos_token_id'):
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
                    elif len(ids) > 1:
                        stop_suffixes.append(s)
                except Exception:
                    logger.debug(f"failed to encode stop sequence: {s!r}", exc_info=True)
                    if len(s) > 1:
                        stop_suffixes.append(s)

        if hasattr(self._tokenizer, 'eos_token_ids'):
            eos_ids.update(self._tokenizer.eos_token_ids)

        # Run speculative steps on executor thread, yielding after each step
        from mlx_lm.models.cache import make_prompt_cache
        target_cache = make_prompt_cache(self._spec_decoder.target)
        draft_cache = make_prompt_cache(self._spec_decoder.draft)

        generated_tokens = []
        prompt_tokens = len(input_ids)

        # Incremental detokenizer for correct multi-byte UTF-8
        detokenizer = self._tokenizer.detokenizer
        detokenizer.reset()

        # Prefill both models
        def _prefill():
            self._spec_decoder.target(input_array, cache=target_cache)
            self._spec_decoder.draft(input_array, cache=draft_cache)

            # Roll back both caches by 1 position so the last prompt token is
            # NOT in the cache.  This prevents double-feeding:
            #   - generate_draft will feed current_ids (last prompt token) into
            #     the draft cache — correct since it was rolled back.
            #   - verify_draft will feed [current_ids, d0, ...] into the target
            #     cache — correct since current_ids was rolled back.
            # Without this rollback, both methods would double-feed the last
            # prompt token (it's already in both caches from prefill).
            try:
                from mlx_lm.models.cache import trim_prompt_cache
                trim_prompt_cache(target_cache, 1)
                trim_prompt_cache(draft_cache, 1)
            except Exception:
                for c in target_cache:
                    if hasattr(c, "trim"):
                        c.trim(1)
                for c in draft_cache:
                    if hasattr(c, "trim"):
                        c.trim(1)

        await loop.run_in_executor(executor, _prefill)
        _spec_gen_t0 = time.perf_counter()  # TTFT timing starts after prefill

        # After prefill + rollback, both caches have prompt[:-1].
        # current_ids is the last prompt token, which will be fed into both
        # caches by generate_draft and verify_draft respectively — no double-feed.
        current_ids = input_array[:, -1:]  # [1, 1] last prompt token

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
                )
                break

            # Snapshot draft cache for rollback on rejection
            draft_snap = self._spec_decoder._snapshot_cache(draft_cache)

            def _spec_step():
                # generate_draft: current_ids is [1,1] single token.
                # First iteration: last prompt token (already in cache from prefill,
                # so forward pass advances the cache by 1 and returns logits).
                # Subsequent iterations: last accepted/bonus token.
                draft_result = self._spec_decoder.generate_draft(current_ids, draft_cache)
                # verify_draft: includes current_ids as alignment token so logits
                # are correctly positioned. Returns accepted tokens + bonus.
                # NOTE: verify_draft feeds [current_ids, draft_tokens] to target,
                # populating target_cache with K+1 tokens. We must NOT re-feed
                # accepted tokens to target (that would double-populate the cache).
                verify_result = self._spec_decoder.verify_draft(
                    draft_result, current_ids, target_cache,
                )
                return draft_result, verify_result

            _, verify_result = await loop.run_in_executor(executor, _spec_step)

            K = self._spec_decoder.config.draft_length
            accepted_count = verify_result.accepted_count
            new_tokens = verify_result.accepted_ids + [verify_result.bonus_token_id]

            self._spec_decoder._stats["total_draft_tokens"] += K
            self._spec_decoder._stats["total_accepted_tokens"] += accepted_count
            self._spec_decoder._stats["total_bonus_tokens"] += 1
            self._spec_decoder._stats["total_steps"] += 1
            if self._lookahead_reasoning is not None:
                self._lookahead_reasoning.record_accept(accepted_count)

            hit_eos = False
            _hit_suffix = False
            for token_id in new_tokens:
                if token_id in eos_ids:
                    hit_eos = True
                    break
                generated_tokens.append(token_id)
                detokenizer.add_token(token_id)
                if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                    _hit_suffix = True
                    # Remove the suffix-triggering token — it should not
                    # appear in the output, matching the non-spec pattern.
                    generated_tokens.pop()
                    # Reset detokenizer to state before the suffix token was
                    # added.  Simply popping from detokenizer.tokens is not
                    # sufficient because NaiveStreamingDetokenizer computes
                    # .text from _current_tokens (an internal list), not from
                    # the .tokens attribute.  Re-decode all remaining tokens
                    # to produce clean text without the suffix.
                    _kept_tokens = list(detokenizer.tokens[:-1]) if detokenizer.tokens else []
                    _saved_offset = detokenizer.offset
                    detokenizer.reset()
                    for _t in _kept_tokens:
                        detokenizer.add_token(_t)
                    detokenizer.offset = _saved_offset
                    break

            # Compute TTFT before first yield
            if not _spec_ttft_recorded:
                _spec_ttft_recorded = True
                _spec_ttft_s = time.perf_counter() - _spec_gen_t0
                _spec_ttft_ms_val = round(_spec_ttft_s * 1000, 1)
                try:
                    from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
                    pm = get_prometheus_metrics()
                    pm.observe_histogram("ttft_seconds", _spec_ttft_s)
                except Exception:
                    logger.debug("spec streaming TTFT prometheus recording failed", exc_info=True)

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
                        chunk_text = chunk_text[:-len(s)]
                        break

            # Build logprobs from target model verification
            _chunk_logprobs = None
            if logprobs and new_tokens:
                _chunk_logprobs = []
                for i, tid in enumerate(new_tokens):
                    tok_text = _clean_special_tokens(self._tokenizer.decode([tid]))
                    lp = verify_result.target_logprobs[i] if i < len(verify_result.target_logprobs) else 0.0
                    _chunk_logprobs.append({
                        "token": tok_text,
                        "logprob": float(lp),
                        "top_logprobs": [{"token": tok_text, "logprob": float(lp)}],
                    })

            yield GenerationOutput(
                text=_clean_special_tokens(detokenizer.text),
                new_text=chunk_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(generated_tokens),
                finished=finish_reason is not None,
                finish_reason=finish_reason,
                reasoning_tokens=0,
                cached_tokens=0,
                logprobs=_chunk_logprobs,
                ttft_ms=_spec_ttft_ms_val,
            )

            if finish_reason is not None:
                detokenizer.finalize()
                break

            # Update caches for next iteration:
            # - Target cache: verify_draft already fed [last_tok, d0..dK-1].
            #   If all accepted, target has exactly the right state (last_tok + K drafts).
            #   If partially accepted, we need to trim rejected tokens from target cache.
            # - Draft cache: if all accepted, draft already has K tokens from generate_draft.
            #   If partially accepted, restore snapshot and refeed accepted + bonus.
            def _update_caches():
                nonlocal draft_snap

                if accepted_count < K:
                    # Partial acceptance: trim target cache to remove rejected entries.
                    # verify_draft fed K+1 tokens (last_tok + K drafts).
                    # We want to keep: last_tok + accepted_count drafts = accepted_count + 1
                    # Trim: (K+1) - (accepted_count + 1) = K - accepted_count entries.
                    from mlx_lm.models.cache import trim_prompt_cache
                    trim_count = K - accepted_count
                    try:
                        trim_prompt_cache(target_cache, trim_count)
                    except Exception:
                        for c in target_cache:
                            if hasattr(c, "trim"):
                                c.trim(trim_count)

                    # Restore draft cache to pre-draft state and refeed accepted tokens.
                    # Do NOT refeed the bonus token here — generate_draft will feed
                    # current_ids (= bonus token) at the start of the next iteration,
                    # so including it now would cause a double-feed.
                    self._spec_decoder._restore_cache(draft_cache, draft_snap)
                    for tok in verify_result.accepted_ids:
                        self._spec_decoder.draft(mx.array([[tok]]), cache=draft_cache)

                # Feed bonus token to both caches (target already has it from verify_draft
                # when all accepted; when partial, we trimmed and need to re-add).
                # For draft: bonus token needs to be fed in both cases.
                # For target: when all accepted, bonus is the last token from verify_draft
                #   logits[K] — already in cache. When partial, we trimmed and re-added
                #   accepted+bonus above, so target is current.

                # Update current_ids for next iteration
                return mx.array([[verify_result.bonus_token_id]])

            current_ids = await loop.run_in_executor(executor, _update_caches)
        except GeneratorExit:
            logger.debug("Client disconnected during spec decode streaming")
            # Clean up KV caches on disconnect to prevent memory leaks
            try:
                del target_cache
            except Exception:
                pass
            try:
                del draft_cache
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Spec decode streaming error: {e}", exc_info=True)
            raise

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
    ) -> GenerationOutput:
        """Generate using N-gram speculative decoding (model-free).

        Uses the N-gram proposer to predict K draft tokens, then verifies
        each by running the target model forward. Accepts matching tokens,
        resamples on mismatch.
        """
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
        from .mlx_executor import get_mlx_executor
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        tokenizer = self._tokenizer
        model = self._model
        # Create a per-request NgramProposer to avoid race conditions
        # when concurrent requests call reset()/propose() on a shared instance.
        from .ngram_proposer import NgramProposer as _NgramProposer, NgramConfig as _NgramConfig
        proposer = _NgramProposer(_NgramConfig(
            max_n=self._ngram_proposer.config.max_n,
            k=self._ngram_proposer.config.k,
            mode=self._ngram_proposer.config.mode,
        ))

        # Handle messages-format prompts (list of dicts) — apply chat template
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            tpl_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                tpl_kwargs["enable_thinking"] = enable_thinking
            text = tokenizer.apply_chat_template(prompt, **tpl_kwargs)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)

        input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        # Build stop token sets
        stop_ids = set()
        stop_suffixes = []
        if hasattr(tokenizer, 'eos_token_id'):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                stop_ids.update(eid)
            elif eid is not None:
                stop_ids.add(eid)
        if hasattr(tokenizer, 'eos_token_ids'):
            stop_ids.update(tokenizer.eos_token_ids)
        if stop_token_ids:
            stop_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                try:
                    ids = tokenizer.encode(s)
                    if len(ids) == 1:
                        stop_ids.add(ids[0])
                except Exception:
                    logger.debug(f"failed to encode stop sequence: {s!r}", exc_info=True)
                if len(s) > 1:
                    stop_suffixes.append(s)

        sampler = make_sampler(
            temp=temperature, top_p=top_p,
            top_k=top_k if top_k > 0 else 0, min_p=min_p,
            xtc_probability=xtc_probability, xtc_threshold=xtc_threshold,
        )

        # Grammar constraint: pre-validate draft tokens against allowed set
        _grammar_constraint = None
        if json_schema is not None:
            try:
                sampler = _build_constrained_sampler(sampler, json_schema, tokenizer)
                _grammar_constraint = sampler.constraint if hasattr(sampler, 'constraint') else None
            except Exception:
                logger.warning("Grammar constraint setup failed for n-gram spec", exc_info=True)

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
            _grammar_constraint.checkpoint()
            allowed = _grammar_constraint.get_allowed_tokens(tokenizer, generated_ids)
            if not allowed:
                _grammar_constraint.rollback()
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
                    allowed = _grammar_constraint.get_allowed_tokens(tokenizer, generated_ids + filtered)
                    if allowed:
                        allowed_set = set(allowed)
                    else:
                        break
                else:
                    break
            _grammar_constraint.rollback()
            return filtered

        # Inflight prefix sharing: defined before _run so cleanup is accessible
        # in exception handlers. Use timestamp instead of id(_run) since _run
        # is not yet defined at this point.
        _ng_inflight_req_id = f"ng-{int(time.monotonic()*1e6)}"

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker
                get_inflight_tracker().unregister(_ng_inflight_req_id)
            except Exception:
                logger.debug("inflight prefix unregister failed in n-gram spec", exc_info=True)

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
            cached_kv, _, matched = (prefix_cache.get(ids) if prefix_cache is not None else (None, None, 0))
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids

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
                logger.debug("inflight prefix register failed in n-gram spec", exc_info=True)

            try:
                gen_t0 = time.perf_counter()
                _timeout_deadline = gen_t0 + timeout_seconds
                _timeout_check_interval = 32
                detokenizer = tokenizer.detokenizer
                detokenizer.reset()
                all_token_ids = list(input_ids)  # Track full history for N-gram matching

                with _wired_limit_ctx(model):
                    # Step 1: Prefill
                    first_logits = None
                    for token, logits in generate_step(
                        ids_to_prefill, model, max_tokens=1, sampler=sampler,
                        prompt_cache=cache,
                    ):
                        first_logits = logits
                        break

                    if first_logits is None:
                        return tokens, "", [], time.perf_counter() - gen_t0, matched, False, False

                    ttft_s = time.perf_counter() - gen_t0

                    # Get first token — use the sampler-applied token from generate_step,
                    # not argmax (which would ignore temperature/top_p/top_k settings).
                    first_token = int(token)
                    tokens.append(first_token)
                    all_token_ids.append(first_token)

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
                        # Propose K draft tokens via N-gram
                        # Use adaptive K if controller is active, else use proposer default
                        _adaptive_k = self._adaptive_spec.get_draft_length() if self._adaptive_spec else None
                        draft_ids = proposer.propose(all_token_ids)[:(_adaptive_k or len(all_token_ids))]
                        # Grammar-aware draft filtering: reject drafts that violate constraints
                        draft_ids = _grammar_filter_drafts(draft_ids, all_token_ids)
                        n_draft = min(len(draft_ids), remaining)

                        if n_draft == 0:
                            # No N-gram proposal — generate one token normally
                            step_input = mx.array([tokens[-1]]).reshape(1, -1)
                            for token, logits in generate_step(
                                step_input, model, max_tokens=1, sampler=sampler,
                                prompt_cache=cache,
                            ):
                                token_id = int(token)
                                tokens.append(token_id)
                                all_token_ids.append(token_id)
                                remaining -= 1
                                if token_id in stop_ids:
                                    tokens.pop()
                                    _stopped_by_stop_id = True
                                    break
                                detokenizer.add_token(token_id)
                                if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                                    tokens.pop()  # Exclude suffix-triggering token from count
                                    _stopped_by_suffix = True
                                    break
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
                            if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
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
                                    logger.debug("Grammar constraint advance failed for token %d", tid, exc_info=True)

                        # Emit bonus token (model's own prediction at rejection/last point)
                        if not _stopped and result.bonus_token is not None and remaining > 0:
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
                                if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                                    tokens.pop()  # Exclude suffix-triggering token from count
                                    _stopped_by_suffix = True
                                    _stopped = True
                            # Advance sampler's grammar constraint for bonus token
                            if _grammar_constraint is not None:
                                try:
                                    tok_text = tokenizer.decode([bonus])
                                    _grammar_constraint.advance(tok_text)
                                except Exception:
                                    pass
                            # CRITICAL: Feed the bonus token through the model cache so it
                            # becomes the last KV entry.  Without this, the next iteration's
                            # verify_with_last_token(last_token_id=bonus) would roll back the
                            # last ACCEPTED draft (which IS in cache) instead of the bonus
                            # token (which is NOT), corrupting the KV cache.
                            if not _stopped_by_stop_id and not _stopped_by_suffix:
                                model(mx.array([[bonus]]), cache=cache)

                        self._ngram_stats["accepted"] += result.accepted_count

                        # Feed back to adaptive spec controller
                        if self._adaptive_spec is not None:
                            self._adaptive_spec.record_step(n_draft, result.accepted_count)

                # Cache KV state
                if self._kv_quant_bits is not None:
                    _maybe_quantize_kv_cache(cache, self._kv_quant_start, self._kv_quant_group_size, self._kv_quant_bits)
                prefix_cache.add(mx.array(input_ids), cache)

                # Finalize detokenizer to flush partial UTF-8 bytes before
                # assembling output.  Without this, multi-byte characters
                # at token boundaries can be truncated, causing incorrect
                # suffix detection via detokenizer.text.endswith() above.
                try:
                    detokenizer.finalize()
                except Exception:
                    logger.debug("detokenizer finalize failed in n-gram spec", exc_info=True)

                # Use detokenizer text when suffix matching is active (same
                # pattern as _generate_fast) because tokenizer.decode(tokens)
                # may contain a partial suffix that leaked across boundaries.
                if _stopped_by_suffix and stop_suffixes:
                    output_text = detokenizer.text
                    for s in stop_suffixes:
                        if output_text.endswith(s):
                            output_text = output_text[:-len(s)]
                            break
                    output_text = _clean_special_tokens(output_text)
                else:
                    output_text = tokenizer.decode(tokens, skip_special_tokens=True)
                mx.synchronize()
                return tokens, output_text, [], ttft_s, matched, _stopped_by_suffix, _stopped_by_stop_id
            finally:
                _unregister_inflight()

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        try:
            tokens, output_text, _, ttft_s, cached_tokens, _stopped_by_suffix, _stopped_by_stop_id = await loop.run_in_executor(executor, _run)
        except MemoryError:
            logger.warning("OOM during N-gram spec generation — returning memory_limit finish reason")
            try:
                import mlx.core as _mx
                await loop.run_in_executor(executor, lambda: (_mx.synchronize(), _mx.clear_cache()))
            except Exception:
                pass
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
                    await loop.run_in_executor(executor, lambda: (_mx.synchronize(), _mx.clear_cache()))
                except Exception:
                    pass
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=str(e),
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            raise
        except Exception as e:
            logger.error(f"Unexpected error during N-gram spec generation: {e}", exc_info=True)
            raise

        # Determine finish_reason with cancel awareness.
        # Stop tokens are popped from `tokens`, so check the flags instead.
        _cancelled = _is_cancelled(cancel_event)
        if _cancelled:
            finish_reason = "stop"
        elif _stopped_by_suffix or _stopped_by_stop_id:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        output_text = _clean_special_tokens(output_text)

        # Trim stop suffix from output text when matched during generation
        if _stopped_by_suffix and stop_suffixes:
            for s in stop_suffixes:
                if output_text.endswith(s):
                    output_text = output_text[:-len(s)]
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
                from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
                pm = get_prometheus_metrics()
                pm.observe_histogram("ttft_seconds", ttft_s)
            except Exception:
                logger.debug("TTFT prometheus recording failed in n-gram spec path", exc_info=True)

        return GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(tokens),
            finished=True,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
            ttft_ms=round(ttft_s * 1000, 1),
            reasoning_tokens=0,
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
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using N-gram speculative decoding (queue-based)."""
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
        from .mlx_executor import get_mlx_executor
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        tokenizer = self._tokenizer
        model = self._model
        # Create a per-request NgramProposer to avoid race conditions
        # when concurrent requests call reset()/propose() on a shared instance.
        from .ngram_proposer import NgramProposer as _NgramProposer, NgramConfig as _NgramConfig
        proposer = _NgramProposer(_NgramConfig(
            max_n=self._ngram_proposer.config.max_n,
            k=self._ngram_proposer.config.k,
            mode=self._ngram_proposer.config.mode,
        ))

        # Handle messages-format prompts (list of dicts) — apply chat template
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            tpl_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                tpl_kwargs["enable_thinking"] = enable_thinking
            text = tokenizer.apply_chat_template(prompt, **tpl_kwargs)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)

        input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        stop_ids = set()
        stop_suffixes = []
        if hasattr(tokenizer, 'eos_token_id'):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                stop_ids.update(eid)
            elif eid is not None:
                stop_ids.add(eid)
        if hasattr(tokenizer, 'eos_token_ids'):
            stop_ids.update(tokenizer.eos_token_ids)
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
                    logger.debug(f"failed to encode stop sequence: {s!r}", exc_info=True)
                    if len(s) > 1:
                        stop_suffixes.append(s)

        sampler = make_sampler(
            temp=temperature, top_p=top_p,
            top_k=top_k if top_k > 0 else 0, min_p=min_p,
            xtc_probability=xtc_probability, xtc_threshold=xtc_threshold,
        )

        # Grammar constraint: pre-validate draft tokens against allowed set
        _stream_grammar_constraint = None
        if json_schema is not None:
            try:
                sampler = _build_constrained_sampler(sampler, json_schema, tokenizer)
                _stream_grammar_constraint = sampler.constraint if hasattr(sampler, 'constraint') else None
            except Exception:
                logger.warning("Grammar constraint setup failed for streaming n-gram spec", exc_info=True)

        def _stream_grammar_filter_drafts(
            draft_ids: list[int],
            generated_ids: list[int],
        ) -> list[int]:
            """Filter draft tokens that violate grammar constraints (streaming path)."""
            if _stream_grammar_constraint is None:
                return draft_ids
            _stream_grammar_constraint.checkpoint()
            allowed = _stream_grammar_constraint.get_allowed_tokens(tokenizer, generated_ids)
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
                        logger.debug("grammar constraint advance failed in streaming filter", exc_info=True)
                        break
                    allowed = _stream_grammar_constraint.get_allowed_tokens(tokenizer, generated_ids + filtered)
                    if allowed:
                        allowed_set = set(allowed)
                    else:
                        break
                else:
                    break
            _stream_grammar_constraint.rollback()
            return filtered

        _sentinel = object()
        _q: asyncio.Queue = asyncio.Queue(maxsize=512)
        loop = asyncio.get_running_loop()
        from .streaming_optimizer import StreamingBackpressureController
        _backpressure = StreamingBackpressureController(max_queue_size=100)

        def _put(item):
            # Backpressure-aware queue with retry (same logic as main streaming path)
            if _q.qsize() > 400:  # 78% of 512
                time.sleep(0.01)
            # Retry up to 3 times if the queue is full, sleeping 1ms between
            # attempts.  Thread-safe: skip get_nowait() — see main streaming
            # _put for rationale (executor thread must not mutate asyncio Queue).
            for _attempt in range(4):  # 1 initial + 3 retries
                if not _q.full():
                    loop.call_soon_threadsafe(_q.put_nowait, item)
                    return
                if _attempt < 3:
                    time.sleep(0.001)
            logger.warning(
                "N-gram spec streaming queue overflow after 3 retries — sending error sentinel. "
                "Client will see finish_reason=error."
            )
            try:
                loop.call_soon_threadsafe(
                    _q.put_nowait,
                    Exception("N-gram streaming queue overflow — output truncated"),
                )
            except Exception:
                pass

        # Inflight prefix sharing for streaming n-gram spec
        _ng_s_inflight_req_id = f"ng-s-{int(time.monotonic()*1e6)}"

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker
                get_inflight_tracker().unregister(_ng_s_inflight_req_id)
            except Exception:
                logger.debug("inflight prefix unregister failed in streaming n-gram spec", exc_info=True)

        def _run():
            try:
                _run_inner()
            except Exception as e:
                logger.error(f"N-gram streaming generation failed: {e}", exc_info=True)
                try:
                    import mlx.core as _cleanup_mx
                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                except Exception:
                    pass
                _put(e)
            finally:
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
            cached_kv, _, matched = (prefix_cache.get(ids) if prefix_cache is not None else (None, None, 0))
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids

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
                logger.debug("inflight prefix register failed in streaming n-gram spec", exc_info=True)

            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            n_tok = 0

            with _wired_limit_ctx(model):
                # Prefill + first token
                for token, _logits in generate_step(
                    ids_to_prefill, model, max_tokens=1, sampler=sampler,
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
                    _put((detokenizer.last_segment, n_tok, None, first_token))

                # Decode with N-gram lookahead
                remaining = max_tokens - 1
                while remaining > 0:
                    if _is_cancelled(cancel_event):
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        if _remaining:
                            _put((_remaining, n_tok, None, 0))
                        _put(("", n_tok, "stop", 0))
                        return
                    _adaptive_k = self._adaptive_spec.get_draft_length() if self._adaptive_spec else None
                    draft_ids = proposer.propose(all_token_ids)
                    if _adaptive_k is not None:
                        draft_ids = draft_ids[:_adaptive_k]
                    # Grammar-aware draft filtering: reject drafts that violate constraints
                    draft_ids = _stream_grammar_filter_drafts(draft_ids, all_token_ids)
                    n_draft = min(len(draft_ids), remaining)

                    if n_draft == 0:
                        step_input = mx.array([tokens[-1]]).reshape(1, -1)
                        for token, _logits in generate_step(
                            step_input, model, max_tokens=1, sampler=sampler,
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
                            suffix_hit = False
                            if stop_suffixes:
                                suffix_hit = any(detokenizer.text.endswith(s) for s in stop_suffixes)
                            _text = "" if suffix_hit else detokenizer.last_segment
                            if suffix_hit:
                                n_tok -= 1  # Exclude suffix-triggering token from count
                            _put((_text, n_tok, "stop" if suffix_hit else None, token_id))
                            if suffix_hit:
                                detokenizer.finalize()
                                _remaining = detokenizer.last_segment
                                if stop_suffixes and _remaining:
                                    for s in stop_suffixes:
                                        if _remaining.endswith(s):
                                            _remaining = _remaining[:-len(s)]
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
                    if hasattr(batch_logits, 'logits'):
                        batch_logits = batch_logits.logits

                    # GPU-accelerated rejection sampling when enabled
                    if self._gpu_rejection_enabled:
                        from .gpu_rejection import GPURejectionSampler as _GRS
                        rej_result = self._gpu_rejection_sampler.verify_greedy(
                            batch_logits[0, :n_draft], draft_ids[:n_draft]
                        )
                        accepted = rej_result.accepted_count

                        for i in range(n_draft):
                            if i < accepted:
                                accepted_id = draft_ids[i]
                            elif i == accepted:
                                accepted_id = _GRS.compute_bonus_token(
                                    batch_logits[0, :n_draft], accepted
                                )
                            else:
                                break
                            tokens.append(accepted_id)
                            all_token_ids.append(accepted_id)
                            remaining -= 1
                            n_tok += 1
                            stop_hit = accepted_id in stop_ids
                            if stop_hit:
                                n_tok -= 1  # Exclude stop token from count
                                _put(("", n_tok, "stop", accepted_id))
                                stopped = True
                            else:
                                detokenizer.add_token(accepted_id)
                                suffix_hit = False
                                if stop_suffixes:
                                    suffix_hit = any(detokenizer.text.endswith(s) for s in stop_suffixes)
                                _text = "" if suffix_hit else detokenizer.last_segment
                                if suffix_hit:
                                    n_tok -= 1  # Exclude suffix-triggering token from count
                                _put((_text, n_tok, "stop" if suffix_hit else None, accepted_id))
                                if suffix_hit:
                                    stopped = True
                            # Advance grammar constraint for accepted/bonus token
                            if _stream_grammar_constraint is not None and not stop_hit:
                                try:
                                    tok_text = tokenizer.decode([accepted_id])
                                    _stream_grammar_constraint.advance(tok_text)
                                except Exception:
                                    pass
                            if i >= accepted:
                                stopped = True
                            if stopped:
                                break

                        # Trim KV cache to remove entries for rejected draft tokens.
                        # The batch forward populated the cache with n_draft entries,
                        # but only `accepted` were verified. Trim the rejected ones.
                        # Then feed the bonus/correction token so its KV entry is
                        # present for the next iteration's batch forward.
                        if accepted < n_draft:
                            try:
                                from mlx_lm.models.cache import trim_prompt_cache
                                trim_prompt_cache(cache, n_draft - accepted)
                            except Exception:
                                for c in cache:
                                    if hasattr(c, "trim"):
                                        c.trim(n_draft - accepted)
                            # Feed the correction token to populate its KV entry
                            if not stopped and tokens:
                                _correction = tokens[-1]
                                _ = model(mx.array([[_correction]]), cache=cache)
                    else:
                        # CPU sequential fallback
                        for i in range(n_draft):
                            model_pick = int(mx.argmax(batch_logits[0, i], axis=-1).item())
                            draft_id = draft_ids[i]
                            is_accept = (model_pick == draft_id)
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
                                suffix_hit = False
                                if stop_suffixes:
                                    suffix_hit = any(detokenizer.text.endswith(s) for s in stop_suffixes)
                                _text = "" if suffix_hit else detokenizer.last_segment
                                if suffix_hit:
                                    n_tok -= 1  # Exclude suffix-triggering token from count
                                _put((_text, n_tok, "stop" if suffix_hit else None, accepted_id))
                                if suffix_hit:
                                    stopped = True
                            # Advance grammar constraint for accepted/bonus token
                            if _stream_grammar_constraint is not None and not stop_hit:
                                try:
                                    tok_text = tokenizer.decode([accepted_id])
                                    _stream_grammar_constraint.advance(tok_text)
                                except Exception:
                                    pass
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
                                    _remaining = _remaining[:-len(s)]
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
        _ng_fp_lock = getattr(self, '_fast_path_lock', None)
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
                except asyncio.TimeoutError:
                    logger.warning(f"N-gram streaming timeout: no token for {timeout_seconds}s")
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
                n_tok = tok_count

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
                        from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
                        pm = get_prometheus_metrics()
                        pm.observe_histogram("ttft_seconds", _ng_ttft_s)
                    except Exception:
                        logger.debug("N-gram streaming TTFT prometheus recording failed", exc_info=True)

                # Build logprobs for this token — n-gram spec decode does not
                # expose per-token logits, so we cannot compute real logprobs.
                # Return None instead of fake 0.0 to avoid misleading consumers.
                _chunk_logprobs = None
                # Real logprobs unavailable from n-gram spec path

                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                    reasoning_tokens=0,
                    cached_tokens=0,
                    logprobs=_chunk_logprobs,
                    ttft_ms=_ng_ttft_ms_val,
                )
                if done:
                    break
        finally:
            # Decrement active fast path count (prevents model eviction mid-generation)
            _ng_fp_lock = getattr(self, '_fast_path_lock', None)
            if _ng_fp_lock is not None:
                with _ng_fp_lock:
                    self._active_fast_path_count -= 1
            if not future.done():
                future.cancel()
                try:
                    await future
                except (asyncio.CancelledError, Exception):
                    pass
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
    ) -> GenerationOutput:
        """Generate using MTP speculative decoding (built-in prediction heads).

        Uses the model's own MTP heads to propose draft tokens, then verifies
        via the backbone forward with n_confirmed=1 for zero-cost reject.
        Best for Qwen3.5 and other models with GatedDeltaNet SSM layers.
        """
        from .mlx_executor import get_mlx_executor
        import mlx.core as mx
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        tokenizer = self._tokenizer
        mtp_decoder = self._mtp_decoder

        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            tpl_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                tpl_kwargs["enable_thinking"] = enable_thinking
            text = tokenizer.apply_chat_template(prompt, **tpl_kwargs)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)

        input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        # Build EOS + stop token sets
        eos_ids = set()
        if hasattr(tokenizer, 'eos_token_id'):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
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
                    logger.debug(f"failed to encode stop sequence: {s!r}", exc_info=True)

        if seed is not None:
            mx.random.seed(seed)

        # Build sampler for MTP path — applied to bonus tokens and rejection
        # corrections while the draft/verify comparison stays greedy.
        from mlx_lm.sample_utils import make_sampler
        _mtp_sampler = make_sampler(
            temp=temperature, top_p=top_p,
            top_k=top_k if top_k > 0 else 0, min_p=min_p,
            xtc_probability=xtc_probability, xtc_threshold=xtc_threshold,
        ) if temperature > 0 or top_p < 1.0 or top_k > 0 or min_p > 0 or xtc_probability > 0 else None

        # Use incremental detokenizer for correct multi-byte UTF-8
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()

        def _run():
            return mtp_decoder.generate(
                input_ids, max_tokens=max_tokens,
                cancel_event=cancel_event,
                sampler=_mtp_sampler,
            )

        _mtp_gen_t0 = time.perf_counter()
        try:
            token_ids = await asyncio.wait_for(
                loop.run_in_executor(executor, _run),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"MTP generation timed out after {timeout_seconds}s")
            try:
                import mlx.core as _mx
                await loop.run_in_executor(executor, lambda: (_mx.synchronize(), _mx.clear_cache()))
            except Exception:
                pass
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
            logger.warning("OOM during MTP generation — returning memory_limit finish reason")
            try:
                import mlx.core as _mx
                await loop.run_in_executor(executor, lambda: (_mx.synchronize(), _mx.clear_cache()))
            except Exception:
                pass
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
                    await loop.run_in_executor(executor, lambda: (_mx.synchronize(), _mx.clear_cache()))
                except Exception:
                    pass
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=str(e),
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            raise
        except Exception as e:
            logger.error(f"Unexpected error during MTP generation: {e}", exc_info=True)
            raise
        _mtp_ttft_s = time.perf_counter() - _mtp_gen_t0

        # Thinking budget enforcement (MTP decoder does not support it natively).
        # Detect <think/</think token IDs and truncate thinking content when the
        # budget is exceeded.  This is a post-processing approximation — the MTP
        # decoder already generated all tokens, but we cap the output to respect
        # the budget, matching the behavior of _generate_fast.
        _mtp_thinking_tokens_used = 0
        _mtp_think_end_token = None
        _mtp_in_thinking = False
        _mtp_think_budget_truncate_idx = None
        if (thinking_budget is not None or enable_thinking) and token_ids:
            try:
                _te_ids = tokenizer.encode("</think")
                if len(_te_ids) == 1:
                    _mtp_think_end_token = _te_ids[0]
                _ts_ids = tokenizer.encode("<think")
                _mtp_think_start_token = _ts_ids[0] if len(_ts_ids) == 1 else None
            except Exception:
                logger.debug("MTP thinking token encode failed", exc_info=True)
            if _mtp_think_end_token is not None:
                for _i, _tid in enumerate(token_ids):
                    if not _mtp_in_thinking and _tid == _mtp_think_start_token:
                        _mtp_in_thinking = True
                    elif _mtp_in_thinking:
                        _mtp_thinking_tokens_used += 1
                        if _tid == _mtp_think_end_token:
                            _mtp_in_thinking = False
                        elif thinking_budget is not None and _mtp_thinking_tokens_used >= thinking_budget:
                            # Truncate at this point and append forced think_end
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
            if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                # Roll back: remove the suffix-triggering token from the
                # detokenizer so its state matches the truncated token_ids.
                # NaiveStreamingDetokenizer supports .tokens attribute.
                if hasattr(detokenizer, 'tokens') and detokenizer.tokens:
                    detokenizer.tokens.pop()
                # Re-initialize detokenizer state from remaining tokens
                # to ensure .text is consistent (simply popping .tokens
                # does not update the internal byte buffer).
                _kept = list(detokenizer.tokens) if hasattr(detokenizer, 'tokens') else token_ids[:i]
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
        if _cancelled:
            finish_reason = "stop"
        elif hit_stop or hit_suffix:
            finish_reason = "stop"
        else:
            finish_reason = "length"

        # Record MTP stats + TTFT in Prometheus
        try:
            from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
            pm = get_prometheus_metrics()
            s = mtp_decoder.stats
            if s.total_cycles > 0:
                pm.set_gauge("mtp_acceptance_rate", s.accepts / s.total_cycles)
                pm.set_gauge("mtp_total_cycles", s.total_cycles)
            pm.observe_histogram("ttft_seconds", _mtp_ttft_s)
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
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using MTP speculative decoding (queue-based).

        Runs MTPDecoder on the executor thread and yields accepted tokens
        as they are verified by the backbone forward pass.
        """
        from .mlx_executor import get_mlx_executor
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        tokenizer = self._tokenizer
        model = self._model
        mtp_decoder = self._mtp_decoder

        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            tpl_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                tpl_kwargs["enable_thinking"] = enable_thinking
            text = tokenizer.apply_chat_template(prompt, **tpl_kwargs)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)

        input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        eos_ids = set()
        stop_suffixes: list[str] = []
        if hasattr(tokenizer, 'eos_token_id'):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
        if hasattr(tokenizer, 'eos_token_ids'):
            eos_ids.update(tokenizer.eos_token_ids)
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
                    logger.debug(f"failed to encode stop sequence: {s!r}", exc_info=True)

        if seed is not None:
            mx.random.seed(seed)

        # Build sampler for MTP streaming path — applied to bonus tokens,
        # rejection corrections, and first token (NOT draft/verify comparison).
        from mlx_lm.sample_utils import make_sampler
        _mtp_sampler = make_sampler(
            temp=temperature, top_p=top_p,
            top_k=top_k if top_k > 0 else 0, min_p=min_p,
            xtc_probability=xtc_probability, xtc_threshold=xtc_threshold,
        ) if temperature > 0 or top_p < 1.0 or top_k > 0 or min_p > 0 or xtc_probability > 0 else None

        # Inflight prefix sharing: register for concurrent KV block sharing
        _inflight_req_id = f"mtp-s-{id(self)}-{int(time.monotonic()*1e6)}"
        try:
            from .inflight_prefix_sharing import get_inflight_tracker
            get_inflight_tracker().register(
                _inflight_req_id,
                token_ids=input_ids,
                kv_cache_ref=None,
            )
        except Exception:
            logger.debug("MTP inflight prefix register failed", exc_info=True)

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker
                get_inflight_tracker().unregister(_inflight_req_id)
            except Exception:
                logger.debug("MTP inflight prefix unregister failed", exc_info=True)

        _sentinel = object()
        _q: asyncio.Queue = asyncio.Queue(maxsize=512)
        loop = asyncio.get_running_loop()
        from .streaming_optimizer import StreamingBackpressureController
        _backpressure = StreamingBackpressureController(max_queue_size=100)

        def _put(item):
            # Backpressure-aware queue with retry (same logic as main streaming path)
            if _q.qsize() > 400:  # 78% of 512
                time.sleep(0.01)
            # Retry up to 3 times if the queue is full, sleeping 1ms between
            # attempts.  Thread-safe: skip get_nowait() — see main streaming
            # _put for rationale (executor thread must not mutate asyncio Queue).
            for _attempt in range(4):  # 1 initial + 3 retries
                if not _q.full():
                    loop.call_soon_threadsafe(_q.put_nowait, item)
                    return
                if _attempt < 3:
                    time.sleep(0.001)
            logger.warning(
                "MTP streaming queue overflow after 3 retries — sending error sentinel. "
                "Client will see finish_reason=error."
            )
            try:
                loop.call_soon_threadsafe(
                    _q.put_nowait,
                    Exception("MTP streaming queue overflow — output truncated"),
                )
            except Exception:
                pass

        def _run():
            try:
                ids = mx.array(input_ids)
                cache = make_prompt_cache(model)
                detokenizer = tokenizer.detokenizer
                detokenizer.reset()

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

                # Yield first token via incremental detokenizer
                detokenizer.add_token(first)
                chunk = _clean_special_tokens(detokenizer.last_segment)
                _put((chunk, 1, None, first))

                from .n_confirmed_patch import clear_rollback, restore_rollback

                _early_stop = False  # Set True when while loop breaks due to stop/cancel

                while len(generated) < max_tokens:
                    # Check cancel_event
                    if _is_cancelled(cancel_event):
                        # Emit stop chunk before breaking so consumer sees finished=True
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        if _remaining:
                            _put((_remaining, len(generated), None, None))
                        _put(("", len(generated), "stop", None))
                        _early_stop = True
                        break

                    # MTP draft — always greedy
                    draft = mtp_decoder._mtp_draft(primary_h, primary)

                    # Verify: backbone forward [primary, draft] with n_confirmed=1
                    verify_out, verify_h = model(
                        mx.array([[primary, draft]]), cache=cache,
                        return_hidden=True, n_confirmed=1,
                    )
                    mx.synchronize()
                    # v0 MUST be greedy for spec decode acceptance check
                    v0 = int(mx.argmax(verify_out[0, 0, :]).item())
                    # v1 (bonus) can use sampler for non-greedy output
                    if _mtp_sampler is not None:
                        v1 = int(_mtp_sampler(verify_out[0, 1:2, :]).item())
                    else:
                        v1 = int(mx.argmax(verify_out[0, 1, :]).item())

                    if v0 == draft:
                        # Accept
                        clear_rollback(cache)
                        generated.append(draft)
                        if draft in eos_ids:
                            # Stop token — don't add to detokenizer, exclude from count
                            _put(("", len(generated) - 1, "stop", draft))
                            _early_stop = True
                            break

                        detokenizer.add_token(draft)
                        # Check stop suffixes on draft token
                        _suffix_hit = stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes)
                        chunk = "" if _suffix_hit else _clean_special_tokens(detokenizer.last_segment)
                        _put((chunk, len(generated) if not _suffix_hit else len(generated) - 1, "stop" if _suffix_hit else None, draft))
                        if _suffix_hit:
                            detokenizer.finalize()
                            _remaining = detokenizer.last_segment
                            if stop_suffixes and _remaining:
                                for s in stop_suffixes:
                                    if _remaining.endswith(s):
                                        _remaining = _remaining[:-len(s)]
                                        break
                            if _remaining:
                                _put((_remaining, len(generated) - 1, None, draft))
                            _early_stop = True
                            break

                        # Bonus token
                        generated.append(v1)
                        if v1 in eos_ids:
                            # Stop token — don't add to detokenizer, exclude from count
                            _put(("", len(generated) - 1, "stop", v1))
                            _early_stop = True
                            break

                        detokenizer.add_token(v1)
                        # Check stop suffixes on bonus token
                        _suffix_hit = stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes)
                        chunk = "" if _suffix_hit else _clean_special_tokens(detokenizer.last_segment)
                        _put((chunk, len(generated) if not _suffix_hit else len(generated) - 1, "stop" if _suffix_hit else None, v1))
                        if _suffix_hit:
                            detokenizer.finalize()
                            _remaining = detokenizer.last_segment
                            if stop_suffixes and _remaining:
                                for s in stop_suffixes:
                                    if _remaining.endswith(s):
                                        _remaining = _remaining[:-len(s)]
                                        break
                            if _remaining:
                                _put((_remaining, len(generated) - 1, None, v1))
                            _early_stop = True
                            break
                        primary = v1
                        primary_h = verify_h[:, -1:, :]
                    else:
                        # Reject: restore rollback (zero-cost)
                        restore_rollback(cache)
                        # Apply sampler to rejection correction token
                        if _mtp_sampler is not None:
                            v0 = int(_mtp_sampler(verify_out[0, 0:1, :]).item())
                        generated.append(v0)
                        if v0 in eos_ids:
                            # Stop token — don't add to detokenizer, exclude from count
                            _put(("", len(generated) - 1, "stop", v0))
                            _early_stop = True
                            break

                        detokenizer.add_token(v0)
                        # Check stop suffixes on rejection correction token
                        _suffix_hit = stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes)
                        chunk = "" if _suffix_hit else _clean_special_tokens(detokenizer.last_segment)
                        _put((chunk, len(generated) if not _suffix_hit else len(generated) - 1, "stop" if _suffix_hit else None, v0))
                        if _suffix_hit:
                            detokenizer.finalize()
                            _remaining = detokenizer.last_segment
                            if stop_suffixes and _remaining:
                                for s in stop_suffixes:
                                    if _remaining.endswith(s):
                                        _remaining = _remaining[:-len(s)]
                                        break
                            if _remaining:
                                _put((_remaining, len(generated) - 1, None, v0))
                            _early_stop = True
                            break
                        primary = v0
                        # Re-feed correction token through rolled-back cache to
                        # get a hidden state consistent with the new primary token.
                        # Using verify_h[:, 0:1, :] here is WRONG because verify_h
                        # was computed before rollback — the cache state has changed.
                        _out_corr, _hid_corr = model(
                            mx.array([[v0]]), cache=cache,
                            return_hidden=True,
                        )
                        mx.synchronize()
                        primary_h = _hid_corr[:, -1:, :]

                # Emit "length" finish chunk only when max_tokens exhausted naturally.
                # If the loop broke early (stop/cancel), a terminal chunk was already
                # emitted inside the loop — skip the spurious second one.
                if not _early_stop:
                    detokenizer.finalize()
                    _remaining = detokenizer.last_segment
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
                    logger.debug("detokenizer finalize in MTP error handler failed", exc_info=True)
                try:
                    mx.synchronize()
                    mx.clear_cache()
                except Exception:
                    pass
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
        _mtp_fp_lock = getattr(self, '_fast_path_lock', None)
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
                except asyncio.TimeoutError:
                    logger.warning(f"MTP streaming timeout: no token for {timeout_seconds}s")
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
                n_tok = tok_count

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
                        from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
                        pm = get_prometheus_metrics()
                        pm.observe_histogram("ttft_seconds", _mtp_ttft_s)
                    except Exception:
                        logger.debug("MTP streaming TTFT prometheus recording failed", exc_info=True)

                # Build logprobs for this token — MTP uses greedy decoding
                # internally and does not expose per-token logits.
                # Return None instead of fake 0.0 to avoid misleading.
                _chunk_logprobs = None
                # Real logprobs unavailable from MTP path

                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                    reasoning_tokens=0,
                    cached_tokens=0,
                    logprobs=_chunk_logprobs,
                    ttft_ms=_mtp_ttft_ms_val,
                )
                if done:
                    break
        finally:
            _unregister_inflight()
            # Decrement active fast path count (prevents model eviction mid-generation)
            _mtp_fp_lock = getattr(self, '_fast_path_lock', None)
            if _mtp_fp_lock is not None:
                with _mtp_fp_lock:
                    self._active_fast_path_count -= 1
            if not future.done():
                future.cancel()
                try:
                    await future
                except (asyncio.CancelledError, Exception):
                    pass
            # Drain queue to unblock any pending call_soon_threadsafe from
            # the executor thread, preventing GPU work from continuing after
            # the consumer has stopped iterating.
            while not _q.empty():
                try:
                    _q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def _apply_chat_template(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
    ) -> str:
        """Apply chat template to convert messages to text."""
        thinking = enable_thinking if enable_thinking is not None else self.enable_thinking
        tokenizer = self._tokenizer

        # Apply model-specific message adapter (oMLX §13.2 pattern)
        try:
            from yunshu_engine.message_adapter import adapt_messages
            messages = adapt_messages(messages, self.model_name)
        except Exception:
            logger.debug("message adapter failed", exc_info=True)

        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                clean = []
                for m in messages:
                    msg = {"role": m.get("role", "user"), "content": m.get("content", "")}
                    # Preserve tool-related fields for correct template rendering
                    if m.get("tool_calls"):
                        msg["tool_calls"] = m["tool_calls"]
                    if m.get("tool_call_id"):
                        msg["tool_call_id"] = m["tool_call_id"]
                    if m.get("name"):
                        msg["name"] = m["name"]
                    clean.append(msg)
                kwargs = {"tokenize": False, "add_generation_prompt": True}
                if thinking is not None:
                    kwargs["enable_thinking"] = thinking
                try:
                    text = tokenizer.apply_chat_template(clean, **kwargs)
                except TypeError as e:
                    if 'enable_thinking' in str(e):
                        logger.warning(f"Model {self.model_name} doesn't support enable_thinking, retrying without")
                        kwargs.pop('enable_thinking', None)
                        text = tokenizer.apply_chat_template(clean, **kwargs)
                    else:
                        raise
                if text:
                    return text
            except Exception:
                logger.debug("chat template failed, using fallback", exc_info=True)

        # Generic fallback
        parts = []
        for m in messages:
            parts.append(f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def has_active_requests(self) -> bool:
        """Check if engine has in-flight requests (including fast-path)."""
        if getattr(self, '_active_fast_path_count', 0) > 0:
            return True
        if self._engine_core:
            return bool(self._engine_core.has_active_requests)
        return False

    def resolve_model_id(self, model_id: str) -> bool:
        """Check if a model ID matches this engine."""
        if not self.model_name:
            return False
        display = self.model_name.rsplit("/", 1)[-1] if "/" in self.model_name else self.model_name
        known = {display, self.model_name}
        known_lower = {k.lower() for k in known if k}
        if model_id in known or model_id.lower() in known_lower:
            return True
        if "/" in model_id:
            stripped = model_id.rsplit("/", 1)[-1]
            if stripped in known or stripped.lower() in known_lower:
                return True
        return False

    def get_stats(self) -> dict:
        if self._engine_core:
            stats = self._engine_core.get_stats()
            stats["model"] = self.model_name
            stats["loaded"] = self._loaded
        else:
            stats = {"model": self.model_name, "loaded": self._loaded}
        if self._thinking_store is not None:
            stats["thinking_segment_store"] = self._thinking_store.get_stats()
        if self._adaptive_spec is not None:
            stats["adaptive_spec"] = self._adaptive_spec.get_stats()
        if getattr(self, '_spec_decoder', None) is not None:
            stats["spec_decode"] = {
                **self._spec_decoder._stats,
                "enabled": getattr(self, '_spec_enabled', False),
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
        # Metal kernel manager status (when enabled via YUNSHU_METAL_KERNELS=1)
        stats["metal_kernels"] = {
            "enabled": getattr(self, '_metal_kernel_manager', None) is not None,
        }
        if getattr(self, '_metal_kernel_manager', None) is not None:
            from .metal_kernels import get_compilation_status
            stats["metal_kernels"].update(get_compilation_status())
        # ANE embedding co-processor status (when enabled via YUNSHU_ANE_EMBEDDINGS=1)
        try:
            from .ane_embedding import get_ane_embedding_stats
            stats["ane_embeddings"] = get_ane_embedding_stats()
        except Exception:
            logger.debug("ane embedding stats failed", exc_info=True)
            stats["ane_embeddings"] = {"enabled": False, "active": False}
        # DeltaNet inversion status (when enabled via YUNSHU_DELTANET_INVERSION=1)
        stats["deltanet_inversion"] = {
            "enabled": getattr(self, '_deltanet_inversion_enabled', False),
            "hooks_registered": getattr(self, '_deltanet_inverter', None) is not None,
            **getattr(self, '_deltanet_inversion_stats', {
                "evictions_captured": 0,
                "inversions_attempted": 0,
                "inversions_succeeded": 0,
                "states_stored": 0,
            }),
        }
        # Wave 42: Model preprocessor registry stats
        if hasattr(self, '_preprocessor_registry') and self._preprocessor_registry is not None:
            stats["model_preprocessor"] = self._preprocessor_registry.get_stats()
        stats["reasoning_tokens"] = getattr(self, '_total_reasoning_tokens', 0)
        stats["response_cache"] = {
            "hits": getattr(self, '_response_cache_hits', 0),
            "misses": getattr(self, '_response_cache_misses', 0),
        }
        # KV Transfer stats (distributed prefill/decode wire protocol)
        stats["kv_transfer"] = {
            "enabled": getattr(self, '_kv_transfer_client', None) is not None,
            **getattr(self, '_kv_transfer_stats', {
                "blocks_transferred": 0,
                "bytes_transferred": 0,
                "transfer_failures": 0,
            }),
        }
        # Prompt cache stats (exact-match KV state reuse)
        if hasattr(self, '_prompt_cache') and self._prompt_cache is not None:
            stats["prompt_cache"] = self._prompt_cache.get_stats()
        # Warm prompt preloading stats (prefill popular prefixes at startup)
        stats["warm_prompt_prefill"] = getattr(self, '_warm_prompt_stats', {
            "prompts_loaded": 0,
            "prompts_prefilled": 0,
            "prompts_skipped_cached": 0,
            "prompts_failed": 0,
            "total_tokens_prefilled": 0,
            "prefill_time_s": 0.0,
            "source": "none",
        })
        # Inflight prefix sharing stats (SGLang cache_unfinished_req pattern)
        try:
            from .inflight_prefix_sharing import get_inflight_tracker
            stats["inflight_prefix_sharing"] = get_inflight_tracker().get_stats()
        except Exception:
            logger.debug("inflight prefix stats unavailable", exc_info=True)
            stats["inflight_prefix_sharing"] = {"enabled": False}
        return stats

    def get_kv_cache_stats(self) -> dict:
        """Return KV cache statistics (prefix cache + paged KV)."""
        result = {"prefix_cache": self._kv_prefix_cache.get_stats()}
        if self._engine_core:
            paged = self._engine_core.get_kv_cache_stats()
            result["paged_kv"] = paged
        return result

    def get_radix_tree_stats(self) -> dict:
        """Return RadixTree statistics (node count, eviction metrics, block usage)."""
        if not self._engine_core:
            return {"enabled": False}
        scheduler = getattr(self._engine_core, "_scheduler", None)
        if scheduler is None:
            return {"enabled": False}
        kv_mgr = getattr(scheduler, "_kv_manager", None)
        if kv_mgr is None:
            return {"enabled": False}
        tree = getattr(kv_mgr, "_radix_tree", None)
        if tree is None:
            return {"enabled": False}
        return {"enabled": True, **tree.get_stats()}

    def get_metal_kernel_manager(self):
        """Return the MetalKernelManager instance, or None if not enabled.

        Callers should check the return value before using:
            mgr = engine.get_metal_kernel_manager()
            if mgr is not None:
                result = mgr.paged_attention_decode(...)
        """
        return self._metal_kernel_manager

    @staticmethod
    def _extract_model_arch(model: Any) -> dict:
        """Extract model architecture parameters for KV cache sizing.

        Reads from the model's config attribute (standard HuggingFace pattern).
        Returns kwargs dict for EngineCoreConfig.
        """
        if model is None:
            return {}

        config = getattr(model, "config", None) or getattr(model, "args", None)
        if config is None:
            return {}

        # Standard HuggingFace config fields
        num_layers = getattr(config, "num_hidden_layers", 0)
        num_kv_heads = getattr(config, "num_key_value_heads", 0)
        head_dim = getattr(config, "hidden_size", 0) // max(
            getattr(config, "num_attention_heads", 1), 1
        )

        if num_layers and num_kv_heads and head_dim:
            return {
                "num_layers": num_layers,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }
        return {}

    def _get_spec_strategy(self):
        """Return the appropriate unified SpecStrategy for this engine.

        Priority:
          1. MTP strategy (built-in prediction heads, if model supports it)
          2. Env-based strategy from SpecStrategyFactory
          3. Otherwise → None

        Returns:
            A SpecStrategy instance, or None if spec decode is not configured.
        """
        if self._mtp_strategy is not None:
            return self._mtp_strategy
        from .spec_interface import SpecStrategyFactory
        return SpecStrategyFactory.from_env()

    def _check_memory_guard(
        self, prompt: str | list, max_tokens: int,
    ) -> GenerationOutput | None:
        """Run memory guard preflight check. Returns None if OK.

        Returns a GenerationOutput with finish_reason="memory_limit"
        if the memory guard rejects the request.
        """
        guard = getattr(self._engine_core, '_memory_guard', None)
        if guard is None:
            return None

        # Estimate prompt tokens
        if self._tokenizer is not None:
            try:
                if isinstance(prompt, list):
                    # Chat messages: estimate from stringified messages
                    text = str(prompt)
                    num_prompt_tokens = len(self._tokenizer.encode(text))
                else:
                    num_prompt_tokens = len(self._tokenizer.encode(prompt))
            except Exception:
                logger.debug("prompt token estimation failed", exc_info=True)
                num_prompt_tokens = len(str(prompt).split()) * 2  # rough estimate
        else:
            num_prompt_tokens = len(str(prompt).split()) * 2

        ok, reason = guard.preflight_check(
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=max_tokens,
        )
        if not ok:
            logger.info(f"Memory guard rejected request: {reason}")
            return GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=num_prompt_tokens,
                completion_tokens=0,
                error=f"Memory guard rejected: {reason}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        return None


def _clean_special_tokens(text: str) -> str:
    """Remove special tokens from output (oMLX pattern)."""
    import re
    # Remove common special tokens that may leak from chat templates
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'<\|endoftext\|>', '', text)
    text = re.sub(r'<\|end\|>', '', text)
    return text
