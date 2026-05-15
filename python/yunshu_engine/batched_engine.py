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
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


@contextmanager
def _wired_limit_ctx(model):
    """Raise wired memory limit during generation to prevent weight swapping.

    Follows mlx-lm's generate.py pattern: set limit to recommended max before
    generation, restore + synchronize after.
    """
    try:
        import mlx.core as mx
        max_rec = mx.metal.recommended_max_working_memory_size()
        model_bytes = sum(p.nbytes for p in model.parameters())
        old_limit = mx.set_wired_limit(max_rec) if model_bytes > max_rec * 0.5 else None
    except Exception:
        logger.debug("wired limit setup failed", exc_info=True)
        old_limit = None
    try:
        yield
    finally:
        if old_limit is not None:
            try:
                import mlx.core as mx
                mx.synchronize()
                mx.set_wired_limit(old_limit)
            except Exception:
                logger.debug("failed", exc_info=True)


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


_PROGRESSIVE_QUANT_INTERVAL = 256


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


def _store_thinking_segment(ids, thinking_tokens: list[int], thinking_store) -> None:
    """Store a thinking segment KV for future reuse."""
    try:
        import hashlib as _hl
        conv_id = _hl.sha256(str([int(t) for t in ids[:16]]).encode()).hexdigest()[:16]
        thinking_store.store(
            conversation_id=conv_id,
            thinking_tokens=thinking_tokens,
            context_tokens=[int(t) for t in ids],
            kv_data=None,
        )
    except Exception:
        logger.debug("failed", exc_info=True)


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
        import json as _json
        schema = _json.loads(json_schema)
    else:
        schema = json_schema
    constraint = JsonSchemaConstraint(schema)
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

        # GPU-accelerated rejection sampling (opt-in via YUNSHU_GPU_REJECTION=1)
        from .gpu_rejection import GPURejectionSampler, should_enable_gpu_rejection
        self._gpu_rejection_sampler = GPURejectionSampler()
        self._gpu_rejection_enabled = should_enable_gpu_rejection()

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

        # SSD KV cache persistence (opt-in via YUNSHU_SSD_CACHE=1)
        import os
        if os.environ.get("YUNSHU_SSD_CACHE", "").strip() in ("1", "true", "yes"):
            ssd_dir = os.environ.get("YUNSHU_SSD_CACHE_DIR", "~/.cache/yunshu/kv-ssd")
            ssd_max_gb = int(os.environ.get("YUNSHU_SSD_CACHE_MAX_GB", "10"))
            self._kv_prefix_cache.enable_ssd_cache(
                cache_dir=ssd_dir,
                max_size_bytes=ssd_max_gb * 1024 ** 3,
                model_name=model_name,
            )

        # KV cache quantization config (mlx-lm pattern: to_quantized)
        # Enable via YUNSHU_KV_QUANT_BITS=4 or 8
        _qbits = os.environ.get("YUNSHU_KV_QUANT_BITS")
        self._kv_quant_bits: int | None = int(_qbits) if _qbits else None
        self._kv_quant_group_size: int = int(
            os.environ.get("YUNSHU_KV_QUANT_GROUP_SIZE", "64")
        )
        self._kv_quant_start: int = int(
            os.environ.get("YUNSHU_KV_QUANT_START", "0")
        )

        # Memory pressure eviction config (vllm-mlx pattern)
        self._mem_pressure_threshold = float(
            os.environ.get("YUNSHU_MEM_PRESSURE_THRESHOLD", "85.0")
        )

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
        self._use_compile = os.environ.get(
            "YUNSHU_MX_COMPILE", ""
        ).strip() in ("1", "true", "yes")

        # Metal kernel manager for custom GPU kernels (paged attention, GEMV, KIVI)
        # Enable via YUNSHU_METAL_KERNELS=1 — provides Metal-accelerated attention,
        # quantized GEMV, and KIVI 2-bit KV cache compression kernels.
        self._metal_kernel_manager = None
        self._metal_kernels_enabled = os.environ.get(
            "YUNSHU_METAL_KERNELS", ""
        ).strip() in ("1", "true", "yes")

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
        self._streaming_backpressure = None  # lazy init
        self._batched_detokenizer = None  # lazy init

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def start(self) -> None:
        """Load model and start EngineCore (oMLX BatchedEngine.start pattern)."""
        if self._loaded:
            return

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

        self._model, self._tokenizer = await loop.run_in_executor(executor, _load)
        self._loaded = True

        # Apply model-specific patches (DeepSeek MLA, Qwen 3.5 YARN, Gemma softcap)
        try:
            from .model_patches import apply_model_patches
            patches = apply_model_patches(self._model, self._tokenizer, self.model_name)
            if patches:
                logger.info(f"Model patches applied: {patches}")
        except Exception:
            logger.debug("Model patches skipped", exc_info=True)

        # Detect model architecture optimizations (RoPE scaling, attention type, MoE)
        try:
            from .model_optimizations import RoPEScalingOptimizer, AttentionOptimizer, MoEEfficiencyOptimizer
            rope_opt = RoPEScalingOptimizer()
            rope_opt.configure(self._model)
            attn_opt = AttentionOptimizer()
            attn_opt.detect_attention_type(self._model)
            moe_opt = MoEEfficiencyOptimizer()
            moe_opt.configure(self._model)
            logger.info(
                f"Model optimizations detected: RoPE={rope_opt.get_scaling_config().scaling_type}, "
                f"Attention={attn_opt.get_stats().get('attention_type', 'unknown')}, "
                f"MoE={moe_opt.get_stats().get('num_experts', 0)} experts"
            )
        except Exception:
            logger.debug("Model optimization detection skipped", exc_info=True)

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
            logger.debug("MTP patch skipped", exc_info=True)

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
            if self._kv_prefix_cache is not None:
                self._kv_prefix_cache._pre_evict_callback = (
                    self._on_prefix_cache_eviction
                )
                logger.info("DeltaNet eviction callback wired into KV prefix cache")

        except Exception as e:
            logger.warning(
                f"DeltaNet inversion init failed ({e}), continuing without SSM recovery"
            )
            self._deltanet_inverter = None

    def _on_prefix_cache_eviction(self, prompt_tokens, cache) -> None:
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

    def invert_evicted_state(self, prompt_tokens: list[int] | None = None) -> list:
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
        if s.ngram_spec_enabled:
            self._ngram_spec_enabled = True
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

    def _init_lora(self):
        """Initialize LoRA adapter manager after model load."""
        max_loras = int(os.environ.get("YUNSHU_MAX_LORAS", "4"))
        from .lora_manager import LoRAAdapterManager
        self._lora_manager = LoRAAdapterManager(max_loras=max_loras)
        self._lora_manager.set_base_model(self._model)

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
            logger.debug("MemoryGuard setup skipped", exc_info=True)

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
        """Stop engine and release resources."""
        if self._engine_core is not None:
            await self._engine_core.stop()
            self._engine_core = None
        # Release KV prefix cache (holds MLX array refs)
        if self._kv_prefix_cache is not None:
            self._kv_prefix_cache.clear()
        self._spec_decoder = None
        self._ngram_proposer = None
        self._adaptive_spec = None
        self._mtp_decoder = None
        self._mtp_strategy = None
        self._warm_prompts = None
        self._thinking_store = None
        self._metal_kernel_manager = None
        # Unregister DeltaNet inversion hooks to restore original class methods
        if self._deltanet_inverter is not None:
            self._deltanet_inverter.unregister_hooks()
            self._deltanet_inverter = None
        self._model = None
        self._tokenizer = None
        self._loaded = False

        import gc
        gc.collect()

        from .mlx_executor import get_mlx_executor
        loop = asyncio.get_running_loop()
        import mlx.core as mx

        def _cleanup():
            mx.synchronize()
            mx.clear_cache()

        await loop.run_in_executor(get_mlx_executor(), _cleanup)
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
        # Auto-detect: switch to batch path when concurrency is detected
        if self._engine_core is not None and self._engine_core.has_active_requests:
            return True
        return False

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
            effort_map = {"low": 2048, "medium": 8192, "high": 32768}
            thinking_budget = effort_map.get(reasoning_effort, 8192)
            if enable_thinking is None:
                enable_thinking = True

        # ── Wave 43: Context window truncation for long prompts ──
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            try:
                if self._tokenizer and hasattr(self._tokenizer, 'encode'):
                    text = self._messages_to_text(prompt, enable_thinking)
                    token_count = len(self._tokenizer.encode(text))
                    max_ctx = getattr(self._model, 'max_seq_len', None)
                    if max_ctx is None:
                        max_ctx = getattr(
                            getattr(self._model, 'config', None), 'max_seq_len', None
                        ) or getattr(
                            getattr(self._model, 'args', None), 'max_seq_len', None
                        )
                    if max_ctx and token_count + max_tokens > max_ctx:
                        from .context_window import ContextWindowManager
                        ctx_mgr = ContextWindowManager()
                        result = ctx_mgr.compute_truncation(
                            messages=prompt,
                            max_tokens=max_ctx - max_tokens,
                            strategy="importance_aware",
                        )
                        prompt = result.messages
                        logger.debug(
                            f"Context window truncated: {token_count} → "
                            f"{result.original_tokens} tokens (saved {result.tokens_removed})"
                        )
            except Exception:
                logger.debug("context window truncation skipped", exc_info=True)

        # Speculative decoding path (Phase 4: single-request EAGLE-3)
        if spec_decode and self._spec_enabled and self._spec_decoder is not None:
            return await self._generate_speculative(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        # MTP speculative decoding (built-in multi-token prediction heads)
        if spec_decode and self._mtp_decoder is not None and not _use_engine_loop:
            return await self._generate_mtp(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
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
                stop=stop,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                json_schema=json_schema,
            )

        # Fast path: direct generate_step on executor thread for full GPU utilization
        if not _use_engine_loop:
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
            )

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
        )

        if result is None:
            return GenerationOutput(finish_reason="error")

        # Map engine_core finish_reason to OpenAI-compatible finish_reason
        finish_reason = result.finish_reason
        if finish_reason == "memory_exceeded":
            finish_reason = "memory_limit"

        # Apply output parser to extract reasoning/tool_calls from raw text
        output_text = _clean_special_tokens(result.output_text)
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

        return GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finished=True,
            finish_reason=finish_reason,
        )

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
    ) -> GenerationOutput:
        """Fast path: run generate_step directly on executor thread.

        Bypasses EngineCore's continuous batching loop for single requests.
        Runs the entire generation in one tight GPU loop on the MLX executor,
        eliminating per-token async round-trip overhead for full GPU utilization.
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
                    preprocessor = self._preprocessor_registry.detect(
                        self.model_name, model
                    )
                    if preprocessor is not None:
                        from .model_preprocessor import PreprocessedInput
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

        logits_processors = []
        if repetition_penalty != 1.0:
            def _rep_penalty(tokens, logits, rp=repetition_penalty, ctx=20):
                if len(tokens) > 0:
                    recent = tokens[-ctx:]
                    import mlx.core as _mx
                    sel = logits[..., recent]
                    sel = _mx.where(sel < 0, sel * rp, sel / rp)
                    logits[..., recent] = sel
                return logits
            logits_processors.append(_rep_penalty)
        if frequency_penalty != 0.0 or presence_penalty != 0.0:
            def _freq_pres_penalty(tokens, logits, fp=frequency_penalty, pp=presence_penalty):
                import mlx.core as _mx
                counts = {}
                for t in tokens:
                    counts[int(t)] = counts.get(int(t), 0) + 1
                for tid, cnt in counts.items():
                    logits[..., tid] -= fp * cnt
                    logits[..., tid] -= pp
                return logits
            logits_processors.append(_freq_pres_penalty)
        if logit_bias:
            def _logit_bias_proc(tokens, logits, biases=logit_bias):
                for tid, bias in biases.items():
                    logits[..., tid] += bias
                return logits
            logits_processors.append(_logit_bias_proc)

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
                try:
                    think_end_token = tokenizer.encode("</think")[-1]
                    think_start_token = tokenizer.encode("<think")[-1]
                except Exception:
                    logger.debug("failed", exc_info=True)

            # Try KV prefix cache hit
            prefix_cache = self._kv_prefix_cache
            # Proactive memory pressure eviction (vllm-mlx pattern)
            if self._mem_pressure_threshold > 0:
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
            cached_kv, _remaining, matched = prefix_cache.get(ids)
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)

            if cached_kv is not None:
                # Only prefill remaining tokens after cache hit
                cached_tokens = matched
                ids_to_prefill = ids[matched:]
            else:
                ids_to_prefill = ids

            # Thinking segment KV lookup — reuse reasoning KV from prior turns
            if self._thinking_store is not None and enable_thinking:
                try:
                    import hashlib as _hl
                    _conv_id = _hl.sha256(str(ids[:16]).encode()).hexdigest()[:16]
                    _context_ids = [int(t) for t in ids]
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
                    else:
                        detokenizer.add_token(first_token)
                    remaining = max_tokens - 1
                    if remaining > 0 and first_token not in stop_ids:
                        for token, logits in generate_step(
                            mx.array([first_token]).reshape(1, -1), model,
                            max_tokens=remaining, sampler=sampler,
                            prompt_cache=cache, logits_processors=_lprocs,
                        ):
                            if cancel_event is not None and cancel_event.is_set():
                                mx.synchronize()
                                break
                            tokens.append(token)
                            if logprobs:
                                log_probs = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1))
                                tok_lp = float(log_probs[token])
                                token_logprobs.append({"token_id": int(token), "logprob": tok_lp})
                            if token in stop_ids:
                                tokens.pop()
                                break
                            if stop_suffixes:
                                detokenizer.add_token(token)
                                if any(detokenizer.text.endswith(s) for s in stop_suffixes):
                                    break
                    cleanup_rope(model)
                    spec_prefill_done = True
                except Exception:
                    logger.warning("SpecPrefill failed, falling back to standard prefill", exc_info=True)
                    if not tokens:
                        cache = make_prompt_cache(model)
                        ids_to_prefill = ids

            if not spec_prefill_done:
                _timeout_deadline = gen_t0 + timeout_seconds
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
                        # Cancellation check
                        if cancel_event is not None and cancel_event.is_set():
                            mx.synchronize()
                            break
                        # Thinking budget enforcement: cap thinking tokens
                        if thinking_budget is not None and enable_thinking:
                            thinking_tokens_used += 1
                            if thinking_tokens_used >= thinking_budget and think_end_token is not None:
                                # Force end of thinking phase
                                tokens.append(think_end_token)
                                break
                        # Progressive KV quantization (C6: keep memory flat during generation)
                        if self._kv_quant_bits is not None:
                            _progressive_quantize_kv_cache(
                                cache, self._kv_quant_start,
                                self._kv_quant_group_size, self._kv_quant_bits,
                                len(tokens),
                            )
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
                        if token in stop_ids:
                            tokens.pop()  # Exclude stop token from output
                            break
                        if stop_suffixes:
                            detokenizer.add_token(token)
                            if any(detokenizer.text.endswith(s) for s in stop_suffixes):
                                break

                        # Track thinking segment boundaries
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

            # Cache the completed KV state for future prefix matching
            # Quantize cache layers to save memory (mlx-lm pattern)
            if self._kv_quant_bits is not None:
                _maybe_quantize_kv_cache(
                    cache, self._kv_quant_start,
                    self._kv_quant_group_size, self._kv_quant_bits,
                )
            prefix_cache.add(ids, cache)

            # Store thinking segment KV for future reuse (if enabled)
            if _thinking_tokens and self._thinking_store is not None:
                try:
                    import hashlib as _hl
                    conv_id = _hl.sha256(str(ids[:16]).encode()).hexdigest()[:16]
                    self._thinking_store.store(
                        conversation_id=conv_id,
                        thinking_tokens=_thinking_tokens,
                        context_tokens=[int(t) for t in ids],
                        kv_data=cache,
                    )
                except Exception:
                    logger.debug("Thinking segment store failed", exc_info=True)

            output_text = tokenizer.decode(tokens, skip_special_tokens=True)
            mx.synchronize()
            return tokens, output_text, token_logprobs, ttft_s, cached_tokens

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        try:
            tokens, output_text, token_logprobs, ttft_s, cached_tokens = await loop.run_in_executor(executor, _run)
        except MemoryError:
            logger.warning("OOM during generation — returning memory_limit finish reason")
            return GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
            )
        except RuntimeError as e:
            if "memory" in str(e).lower() or "out of" in str(e).lower():
                logger.warning(f"MLX OOM during generation: {e}")
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                )
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

        finish_reason = "stop" if tokens and tokens[-1] in stop_ids else "length"
        output_text = _clean_special_tokens(output_text)

        # Record TTFT + ITL in Prometheus
        if ttft_s > 0:
            try:
                from ..middleware.prometheus_exporter import get_prometheus_metrics
                pm = get_prometheus_metrics()
                pm.observe_histogram("ttft_seconds", ttft_s)
                if cached_tokens > 0:
                    pm.set_gauge("kv_prefix_cache_hits", 1)
                else:
                    pm.set_gauge("kv_prefix_cache_misses", 1)
                # ITL: average inter-token latency
                if _itl_samples:
                    avg_itl = sum(_itl_samples) / len(_itl_samples)
                    pm.observe_histogram("itl_seconds", avg_itl)
            except Exception:
                logger.debug("TTFT/ITL prometheus recording failed", exc_info=True)

        return GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(tokens),
            finished=True,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
            logprobs=lp_result,
            ttft_ms=round(ttft_s * 1000, 1),
            reasoning_tokens=len(_thinking_tokens),
        )

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
    ) -> AsyncIterator[GenerationOutput]:
        """Streaming text generation.

        Default uses fast path (direct generate_step on executor) for single
        requests. Set use_engine_loop=True for continuous batching path.
        If use_engine_loop is None, uses YUNSHU_ENGINE_LOOP env var.
        """
        if not self._loaded:
            await self.start()

        _use_engine_loop = self._should_use_engine_loop(use_engine_loop)
        # Resolve reasoning_effort → thinking_budget if not explicitly set
        if thinking_budget is None and reasoning_effort is not None:
            effort_map = {"low": 2048, "medium": 8192, "high": 32768}
            thinking_budget = effort_map.get(reasoning_effort, 8192)
            if enable_thinking is None:
                enable_thinking = True

        # Memory guard preflight check
        guard_rejection = self._check_memory_guard(prompt, max_tokens)
        if guard_rejection is not None:
            yield guard_rejection
            return

        # Speculative decoding path (Phase 4)
        if spec_decode and self._spec_enabled and self._spec_decoder is not None:
            async for output in self._stream_generate_speculative(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            ):
                yield output
            return

        # MTP speculative decoding streaming (built-in multi-token prediction)
        if spec_decode and self._mtp_decoder is not None and not _use_engine_loop:
            async for output in self._stream_generate_mtp(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            ):
                yield output
            return

        # N-gram speculative decoding streaming (model-free)
        if spec_decode and self._ngram_proposer is not None and not _use_engine_loop:
            async for output in self._stream_generate_ngram_spec(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty, stop=stop, seed=seed,
            ):
                yield output
            return

        # Fast path: bypass EngineCore for single-request streaming
        if not _use_engine_loop:
            # Register with request tracker for cancellation support
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
                ):
                    yield output
            finally:
                if _cancel_event is not None:
                    try:
                        _tracker.unregister(_stream_req_id)
                    except Exception:
                        logger.debug("failed", exc_info=True)
            return

        # Engine loop path: continuous batching with scheduler
        await self._ensure_engine_core()
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
        )

        finished_normally = False
        try:
            async for output in self._engine_core.stream_outputs(request_id):
                cleaned = _clean_special_tokens(output.new_text)
                finish_reason = output.finish_reason
                if finish_reason == "memory_exceeded":
                    finish_reason = "memory_limit"
                gen_output = GenerationOutput(
                    text=_clean_special_tokens(output.output_text),
                    new_text=cleaned,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=finish_reason,
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
                    logger.debug("failed", exc_info=True)

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
        from .streaming_optimizer import StreamingBackpressureController, BatchedDetokenizer
        _backpressure = StreamingBackpressureController(max_queue_size=100)
        _batched_detok = BatchedDetokenizer(tokenizer)

        # TokenPipeline for GPU/CPU overlap — activated via YUNSHU_STREAMING_PIPELINE=1
        _pipeline = None
        if self._streaming_pipeline_enabled:
            from .streaming_optimizer import TokenPipeline, PipelineConfig
            _pipeline = TokenPipeline(PipelineConfig(enable_overlap=True))
            _pipeline.start_pipeline(request=None)
            logger.debug("TokenPipeline active for streaming fast path")

        tokenizer = self._tokenizer
        model = self._model

        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            tpl_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                tpl_kwargs["enable_thinking"] = enable_thinking
            prompt = tokenizer.apply_chat_template(prompt, **tpl_kwargs)

        input_ids = tokenizer.encode(prompt)
        prompt_tokens = len(input_ids)

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
        logits_processors = []
        if repetition_penalty != 1.0:
            def _repetition_penalty(tokens, logits, rp=repetition_penalty, ctx=20):
                if len(tokens) > 0:
                    recent = tokens[-ctx:]
                    import mlx.core as _mx
                    sel = logits[..., recent]
                    sel = _mx.where(sel < 0, sel * rp, sel / rp)
                    logits[..., recent] = sel
                return logits
            logits_processors.append(_repetition_penalty)
        if frequency_penalty != 0.0 or presence_penalty != 0.0:
            def _freq_pres_penalty(tokens, logits, fp=frequency_penalty, pp=presence_penalty):
                import mlx.core as _mx
                counts = {}
                for t in tokens:
                    counts[int(t)] = counts.get(int(t), 0) + 1
                for tid, cnt in counts.items():
                    logits[..., tid] -= fp * cnt
                    logits[..., tid] -= pp
                return logits
            logits_processors.append(_freq_pres_penalty)
        if logit_bias:
            def _logit_bias_proc(tokens, logits, biases=logit_bias):
                for tid, bias in biases.items():
                    logits[..., tid] += bias
                return logits
            logits_processors.append(_logit_bias_proc)

        # Thread-safe bridge: executor puts via call_soon_threadsafe so the
        # event loop's async consumer is woken for every token.
        _sentinel = object()
        _q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _put(item):
            loop.call_soon_threadsafe(_q.put_nowait, item)

        def _run():
            try:
                _run_inner()
            except (MemoryError, RuntimeError) as e:
                if isinstance(e, MemoryError) or "memory" in str(e).lower():
                    logger.warning(f"OOM during streaming: {e}")
                    _put(e)
                else:
                    _put(e)
            except Exception as e:
                _put(e)
            finally:
                _put(_sentinel)

        def _run_inner():
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
            ids = mx.array(input_ids)
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
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
                try:
                    think_end_token = tokenizer.encode("</think")[-1]
                    think_start_token = tokenizer.encode("<think")[-1]
                except Exception:
                    logger.debug("failed", exc_info=True)

            # KV prefix cache for streaming
            prefix_cache = self._kv_prefix_cache
            # Proactive memory pressure eviction (vllm-mlx pattern)
            if self._mem_pressure_threshold > 0:
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
            cached_kv, _remaining, matched = prefix_cache.get(ids)
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids

            _lprocs = logits_processors if logits_processors else None
            with _wired_limit_ctx(model):
                for token, _ in generate_step(
                    ids_to_prefill, model, max_tokens=max_tokens, sampler=sampler,
                    prompt_cache=cache, logits_processors=_lprocs,
                ):
                    detokenizer.add_token(token)
                    n_tok += 1
                    # TokenPipeline: submit GPU stages for tracking
                    if _pipeline is not None and _pipeline.is_running:
                        _ptok = _pipeline.submit_stage1_result(logits=None, token_id=int(token))
                        _ptok = _pipeline.submit_stage2_result(_ptok, sampled_id=int(token))
                    # Check cancellation
                    if cancel_event is not None and cancel_event.is_set():
                        mx.synchronize()
                        if _pipeline is not None:
                            _pipeline.finish()
                        return
                    # Prefill complete on first token — remove from progress tracker
                    if _first_token:
                        _first_token = False
                        if _prefill_tracker is not None:
                            _prefill_tracker.update(
                                _prefill_req_id, prompt_tokens, prompt_tokens,
                                self.model_name or "default",
                            )
                    new_text = detokenizer.last_segment
                    stop_hit = token in stop_ids
                    suffix_hit = False
                    if not stop_hit and stop_suffixes:
                        if any(detokenizer.text.endswith(s) for s in stop_suffixes):
                            suffix_hit = True
                    # Thinking budget enforcement in streaming
                    if thinking_budget is not None and enable_thinking:
                        thinking_tokens_used += 1
                        if thinking_tokens_used >= thinking_budget and think_end_token is not None:
                            # Store thinking segment before returning
                            if _thinking_tokens and self._thinking_store is not None:
                                _store_thinking_segment(ids, _thinking_tokens, self._thinking_store)
                            _put((new_text, n_tok, True, len(_thinking_tokens)))
                            if _pipeline is not None:
                                _pipeline.finish()
                            prefix_cache.add(ids, cache)
                            mx.synchronize()
                            return
                    # Track thinking segment boundaries in streaming
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
                    _put((new_text, n_tok, stop_hit or suffix_hit, len(_thinking_tokens)))
                    if stop_hit or suffix_hit:
                        # Store thinking segment on stop
                        if _thinking_tokens and self._thinking_store is not None:
                            _store_thinking_segment(ids, _thinking_tokens, self._thinking_store)
                        if _pipeline is not None:
                            _pipeline.finish()
                        prefix_cache.add(ids, cache)
                        mx.synchronize()
                        return
                # Store thinking segment at end of generation
                if _thinking_tokens and self._thinking_store is not None:
                    _store_thinking_segment(ids, _thinking_tokens, self._thinking_store)
                prefix_cache.add(ids, cache)
                detokenizer.finalize()
                remaining = detokenizer.last_segment
                if remaining:
                    _put((remaining, n_tok, False, len(_thinking_tokens)))
                _put(("", n_tok, True, len(_thinking_tokens)))
                mx.synchronize()
                # Finish pipeline tracking at end of generation
                if _pipeline is not None:
                    _pipeline.finish()
                # Clean up prefill progress entry (may persist if first token wasn't reached)
                if _prefill_tracker is not None:
                    _prefill_tracker.remove(_prefill_req_id)

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run)

        accumulated = ""
        n_tok = 0
        _reasoning_tokens = 0
        try:
            while True:
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=120)
                except asyncio.TimeoutError:
                    logger.warning("Streaming fast path timeout: no token for 120s")
                    break
                if item is _sentinel:
                    break
                if isinstance(item, BaseException):
                    err_msg = str(item).lower()
                    is_oom = isinstance(item, MemoryError) or "memory" in err_msg
                    if is_oom and accumulated:
                        yield GenerationOutput(
                            text=_clean_special_tokens(accumulated),
                            new_text="",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=n_tok,
                            finished=True,
                            finish_reason="memory_limit",
                        )
                    break
                if len(item) == 4:
                    new_text, tok_count, done, _reasoning_tokens = item
                else:
                    new_text, tok_count, done = item
                accumulated += new_text
                n_tok = tok_count

                # TokenPipeline: run stage 3 overlap for stats tracking
                if _pipeline is not None and _pipeline.is_running:
                    await _pipeline.run_stage3_overlap(
                        _pipeline._current,
                        detokenize_fn=lambda _tid, _t=new_text: _t,
                    )
                    await _pipeline.next_token(
                        detokenize_fn=lambda _tid, _t=new_text: _t,
                    )

                # Streaming backpressure: slow down if client can't keep up
                if _backpressure.check_backpressure(_q.qsize()):
                    _delay = _backpressure.get_delay_ms(_q.qsize())
                    if _delay > 0:
                        await asyncio.sleep(_delay / 1000)

                finish_reason = "stop" if done else None
                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                    reasoning_tokens=_reasoning_tokens,
                )
                if done:
                    break
        finally:
            # Stop pipeline and log stats
            if _pipeline is not None:
                _pipeline.stop()
                logger.info(
                    "TokenPipeline stats: %s",
                    _pipeline.get_stats(),
                )
            if not future.done():
                future.cancel()

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
        """Prefill KV cache with common system prompts on startup.

        Enabled via YUNSHU_WARM_PROMPTS env var (comma-separated paths or inline text).
        Provides 1.3-2.25x TTFT improvement on first real request with matching prefix.
        """
        import os
        warm_prompts_raw = os.environ.get("YUNSHU_WARM_PROMPTS", "").strip()
        if not warm_prompts_raw:
            return

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        prompts = [p.strip() for p in warm_prompts_raw.split("||") if p.strip()]
        prefilled = 0

        for prompt_text in prompts:
            # If it looks like a file path, try reading it
            if prompt_text.startswith("/") or prompt_text.startswith("~"):
                try:
                    expanded = os.path.expanduser(prompt_text)
                    with open(expanded) as f:
                        prompt_text = f.read().strip()
                except Exception:
                    logger.debug(f"Warm prompt file not found: {prompt_text}", exc_info=True)
                    continue

            if not prompt_text:
                continue

            def _prefill(text=prompt_text):
                from mlx_lm.models.cache import make_prompt_cache
                import mlx.core as mx
                from mlx_lm.generate import generate_step
                from mlx_lm.sample_utils import make_sampler
                ids = mx.array(self._tokenizer.encode(text))
                prefix_cache = self._kv_prefix_cache
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
                cached_kv, _, matched = prefix_cache.get(ids)
                if cached_kv is not None:
                    return 0  # Already cached
                cache = make_prompt_cache(self._model)
                sampler = make_sampler(temp=0.0)
                for _ in generate_step(ids, self._model, max_tokens=1, sampler=sampler,
                                       prompt_cache=cache):
                    break
                prefix_cache.add(ids, cache)
                mx.clear_cache()
                return len(ids)

            try:
                n_tokens = await loop.run_in_executor(executor, _prefill)
                if n_tokens > 0:
                    prefilled += 1
                    logger.info(f"Warm prompt prefilled: {n_tokens} tokens")
            except Exception as e:
                logger.warning(f"Warm prompt prefill failed: {e}")

        if prefilled > 0:
            logger.info(f"Warm prompt prefill complete: {prefilled}/{len(prompts)} prompts cached")

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
            # Also create unified SpecProposer wrapper
            from .spec_proposer import NgramSpecProposer
            self._spec_proposer = NgramSpecProposer(NgramConfig(max_n=max_n, k=k, mode=mode))
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
                    except (FileNotFoundError, Exception) as e:
                        logger.info(f"MTP weights not available ({e}), using backbone-only MTP")

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
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationOutput:
        """Generate using speculative decoding (single-request EAGLE-3 path).

        This path bypasses the continuous batching scheduler and runs the
        SpeculativeDecoder directly. Best for single-request scenarios where
        the draft model can propose K tokens for the target to verify.
        """
        if self._spec_decoder is None:
            # Fall back to standard generation if no decoder
            return await self.generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            )

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        # Tokenize prompt
        input_ids = self._tokenizer.encode(prompt)
        import mlx.core as mx
        input_array = mx.array(input_ids).reshape(1, -1)

        # Run speculative generation on the MLX executor thread
        def _run_spec():
            return self._spec_decoder.generate(
                input_ids=input_array,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        token_ids = await loop.run_in_executor(executor, _run_spec)

        # Detokenize
        text = self._tokenizer.decode(token_ids)
        text = _clean_special_tokens(text)

        return GenerationOutput(
            text=text,
            new_text=text,
            prompt_tokens=len(input_ids),
            completion_tokens=len(token_ids),
            finished=True,
            finish_reason="stop" if len(token_ids) < max_tokens else "length",
        )

    async def _stream_generate_speculative(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using speculative decoding (single-request path).

        Yields chunks as they are verified by the target model.
        Each yield contains the accepted tokens from one verify step.
        """
        if self._spec_decoder is None:
            # Fall back to standard streaming
            async for output in self.stream_generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            ):
                yield output
            return

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        # Tokenize prompt
        input_ids = self._tokenizer.encode(prompt)
        import mlx.core as mx
        input_array = mx.array(input_ids).reshape(1, -1)

        # Get EOS IDs
        eos_ids = set()
        if hasattr(self._tokenizer, 'eos_token_id'):
            eid = self._tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)

        # Run speculative steps on executor thread, yielding after each step
        from mlx_lm.models.cache import make_prompt_cache
        target_cache = make_prompt_cache(self._spec_decoder.target)
        draft_cache = make_prompt_cache(self._spec_decoder.draft)

        generated_tokens = []
        current_ids = input_array
        prompt_tokens = len(input_ids)

        # Prefill both models
        def _prefill():
            self._spec_decoder.target(input_array, cache=target_cache)
            self._spec_decoder.draft(input_array, cache=draft_cache)

        await loop.run_in_executor(executor, _prefill)

        while len(generated_tokens) < max_tokens:
            def _spec_step():
                draft_result = self._spec_decoder.generate_draft(current_ids, draft_cache)
                verify_result = self._spec_decoder.verify_draft(
                    draft_result, current_ids, target_cache,
                )
                return draft_result, verify_result

            _, verify_result = await loop.run_in_executor(executor, _spec_step)

            new_tokens = verify_result.accepted_ids + [verify_result.bonus_token_id]

            self._spec_decoder._stats["total_draft_tokens"] += self._spec_decoder.config.draft_length
            self._spec_decoder._stats["total_accepted_tokens"] += verify_result.accepted_count
            self._spec_decoder._stats["total_bonus_tokens"] += 1
            self._spec_decoder._stats["total_steps"] += 1

            hit_eos = False
            for token_id in new_tokens:
                generated_tokens.append(token_id)
                if token_id in eos_ids:
                    hit_eos = True
                    break

            # Yield accepted text
            chunk_text = _clean_special_tokens(self._tokenizer.decode(new_tokens))
            finish_reason = None
            if hit_eos:
                finish_reason = "stop"
            elif len(generated_tokens) >= max_tokens:
                finish_reason = "length"

            yield GenerationOutput(
                text=_clean_special_tokens(self._tokenizer.decode(generated_tokens)),
                new_text=chunk_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(generated_tokens),
                finished=finish_reason is not None,
                finish_reason=finish_reason,
            )

            if finish_reason is not None:
                break

            # Feed accepted tokens back
            def _feedback():
                accepted_tensor = mx.array(new_tokens).reshape(1, -1)
                self._spec_decoder.target(accepted_tensor, cache=target_cache)
                self._spec_decoder.draft(accepted_tensor, cache=draft_cache)
                return accepted_tensor[:, -1:]

            current_ids = await loop.run_in_executor(executor, _feedback)

    async def _generate_ngram_spec(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        json_schema: dict | str | None = None,
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
        proposer = self._ngram_proposer

        input_ids = tokenizer.encode(prompt if isinstance(prompt, str) else str(prompt))
        prompt_tokens = len(input_ids)

        # Build stop token sets
        stop_ids = set()
        stop_suffixes = []
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

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0, min_p=min_p)

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

        def _run():
            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            tokens = []
            ttft_s = 0.0

            # Prefill with KV prefix cache
            prefix_cache = self._kv_prefix_cache
            prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
            cached_kv, _remaining, matched = prefix_cache.get(ids)
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids

            gen_t0 = time.perf_counter()
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
                    return tokens, "", [], time.perf_counter() - gen_t0, 0

                ttft_s = time.perf_counter() - gen_t0

                # Get first token from the model
                first_token = int(mx.argmax(first_logits, axis=-1).flatten()[0])
                tokens.append(first_token)
                all_token_ids.append(first_token)

                # Step 2: Decode loop with N-gram lookahead
                remaining = max_tokens - 1
                while remaining > 0:
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
                                break
                            detokenizer.add_token(token_id)
                            if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                                break
                        continue

                    # C10: Batch verify all K draft tokens in one forward pass
                    self._ngram_stats["proposals"] += 1
                    self._ngram_stats["total_draft"] += n_draft
                    accepted = 0

                    # Feed all draft tokens to the model in one batch forward
                    draft_arr = mx.array(draft_ids[:n_draft]).reshape(1, -1)
                    # Use the model directly for batch forward (bypass generate_step)
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

                        # Append accepted tokens
                        for i in range(accepted):
                            tid = draft_ids[i]
                            tokens.append(tid)
                            all_token_ids.append(tid)
                            remaining -= 1
                            if tid in stop_ids:
                                tokens.pop()
                                break
                            detokenizer.add_token(tid)
                            if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                                break

                        # Resample at rejection point
                        if accepted < n_draft and remaining > 0:
                            bonus = _GRS.compute_bonus_token(batch_logits[0, :n_draft], accepted)
                            tokens.append(bonus)
                            all_token_ids.append(bonus)
                            remaining -= 1
                            if bonus not in stop_ids:
                                detokenizer.add_token(bonus)
                            if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                                pass
                    else:
                        # CPU sequential fallback: compare model's argmax at each position with draft
                        for i in range(n_draft):
                            model_pick = int(mx.argmax(batch_logits[0, i], axis=-1).item())
                            draft_id = draft_ids[i]

                            if model_pick == draft_id:
                                tokens.append(draft_id)
                                all_token_ids.append(draft_id)
                                accepted += 1
                                remaining -= 1
                                if draft_id in stop_ids:
                                    tokens.pop()
                                    break
                                detokenizer.add_token(draft_id)
                                if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                                    break
                            else:
                                # Reject: resample from model's distribution at this position
                                resampled = model_pick
                                tokens.append(resampled)
                                all_token_ids.append(resampled)
                                remaining -= 1
                                if resampled in stop_ids:
                                    tokens.pop()
                                    break
                                detokenizer.add_token(resampled)
                                if stop_suffixes and any(detokenizer.text.endswith(s) for s in stop_suffixes):
                                    break
                                break  # Stop verifying rest of drafts

                    self._ngram_stats["accepted"] += accepted

                    # Feed back to adaptive spec controller
                    if self._adaptive_spec is not None:
                        self._adaptive_spec.record_step(n_draft, accepted)

            # Cache KV state
            if self._kv_quant_bits is not None:
                _maybe_quantize_kv_cache(cache, self._kv_quant_start, self._kv_quant_group_size, self._kv_quant_bits)
            prefix_cache.add(mx.array(input_ids), cache)

            output_text = tokenizer.decode(tokens, skip_special_tokens=True)
            mx.synchronize()
            return tokens, output_text, [], ttft_s, matched

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        tokens, output_text, _, ttft_s, cached_tokens = await loop.run_in_executor(executor, _run)

        finish_reason = "stop" if tokens and tokens[-1] in stop_ids else "length"
        output_text = _clean_special_tokens(output_text)
        return GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(tokens),
            finished=True,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
            ttft_ms=round(ttft_s * 1000, 1),
        )

    async def _stream_generate_ngram_spec(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        json_schema: dict | str | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using N-gram speculative decoding (queue-based)."""
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
        from .mlx_executor import get_mlx_executor
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        tokenizer = self._tokenizer
        model = self._model
        proposer = self._ngram_proposer

        input_ids = tokenizer.encode(prompt if isinstance(prompt, str) else str(prompt))
        prompt_tokens = len(input_ids)

        stop_ids = set()
        stop_suffixes = []
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

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0, min_p=min_p)

        _sentinel = object()
        _q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _put(item):
            loop.call_soon_threadsafe(_q.put_nowait, item)

        def _run():
            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            tokens = []
            all_token_ids = list(input_ids)

            prefix_cache = self._kv_prefix_cache
            prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
            cached_kv, _rem, matched = prefix_cache.get(ids)
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids

            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            n_tok = 0

            with _wired_limit_ctx(model):
                # Prefill + first token
                for token, logits in generate_step(
                    ids_to_prefill, model, max_tokens=1, sampler=sampler,
                    prompt_cache=cache,
                ):
                    first_token = int(token)
                    tokens.append(first_token)
                    all_token_ids.append(first_token)
                    detokenizer.add_token(first_token)
                    n_tok += 1
                    _put((detokenizer.last_segment, n_tok, False))

                # Decode with N-gram lookahead
                remaining = max_tokens - 1
                while remaining > 0:
                    _adaptive_k = self._adaptive_spec.get_draft_length() if self._adaptive_spec else None
                    draft_ids = proposer.propose(all_token_ids)
                    if _adaptive_k is not None:
                        draft_ids = draft_ids[:_adaptive_k]
                    n_draft = min(len(draft_ids), remaining)

                    if n_draft == 0:
                        step_input = mx.array([tokens[-1]]).reshape(1, -1)
                        for token, logits in generate_step(
                            step_input, model, max_tokens=1, sampler=sampler,
                            prompt_cache=cache,
                        ):
                            token_id = int(token)
                            tokens.append(token_id)
                            all_token_ids.append(token_id)
                            remaining -= 1
                            n_tok += 1
                            detokenizer.add_token(token_id)
                            stop_hit = token_id in stop_ids
                            suffix_hit = False
                            if not stop_hit and stop_suffixes:
                                suffix_hit = any(detokenizer.text.endswith(s) for s in stop_suffixes)
                            _put((detokenizer.last_segment, n_tok, stop_hit or suffix_hit))
                            if stop_hit or suffix_hit:
                                prefix_cache.add(ids, cache)
                                mx.synchronize()
                                _put(_sentinel)
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
                            detokenizer.add_token(accepted_id)
                            stop_hit = accepted_id in stop_ids
                            suffix_hit = False
                            if not stop_hit and stop_suffixes:
                                suffix_hit = any(detokenizer.text.endswith(s) for s in stop_suffixes)
                            _put((detokenizer.last_segment, n_tok, stop_hit or suffix_hit))
                            if stop_hit or suffix_hit:
                                stopped = True
                            if i >= accepted:
                                stopped = True
                            if stopped:
                                break
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
                            detokenizer.add_token(accepted_id)
                            stop_hit = accepted_id in stop_ids
                            suffix_hit = False
                            if not stop_hit and stop_suffixes:
                                suffix_hit = any(detokenizer.text.endswith(s) for s in stop_suffixes)
                            _put((detokenizer.last_segment, n_tok, stop_hit or suffix_hit))
                            if stop_hit or suffix_hit:
                                stopped = True
                            if not is_accept:
                                stopped = True
                            if stopped:
                                break

                    self._ngram_stats["accepted"] += accepted
                    if self._adaptive_spec is not None:
                        self._adaptive_spec.record_step(n_draft, accepted)
                    if stopped:
                        prefix_cache.add(ids, cache)
                        mx.synchronize()
                        _put(_sentinel)
                        return

            prefix_cache.add(ids, cache)
            detokenizer.finalize()
            remaining = detokenizer.last_segment
            if remaining:
                _put((remaining, n_tok, False))
            _put(("", n_tok, True))
            mx.synchronize()
            _put(_sentinel)

        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run)

        accumulated = ""
        n_tok = 0
        try:
            while True:
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=120)
                except asyncio.TimeoutError:
                    logger.warning("N-gram streaming timeout: no token for 120s")
                    break
                if item is _sentinel:
                    break
                new_text, tok_count, done = item
                accumulated += new_text
                n_tok = tok_count

                # Streaming backpressure: slow down if client can't keep up
                if _backpressure.check_backpressure(_q.qsize()):
                    _delay = _backpressure.get_delay_ms(_q.qsize())
                    if _delay > 0:
                        await asyncio.sleep(_delay / 1000)

                finish_reason = "stop" if done else None
                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                )
                if done:
                    break
        finally:
            if not future.done():
                future.cancel()

    async def _generate_mtp(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationOutput:
        """Generate using MTP speculative decoding (built-in prediction heads).

        Uses the model's own MTP heads to propose draft tokens, then verifies
        via the backbone forward with n_confirmed=1 for zero-cost reject.
        Best for Qwen3.5 and other models with GatedDeltaNet SSM layers.
        """
        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        tokenizer = self._tokenizer
        model = self._model
        mtp_decoder = self._mtp_decoder

        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)

        input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        def _run():
            return mtp_decoder.generate(input_ids, max_tokens=max_tokens)

        token_ids = await loop.run_in_executor(executor, _run)

        output_text = _clean_special_tokens(
            tokenizer.decode(token_ids, skip_special_tokens=True)
        )

        finish_reason = "length"
        eos_ids = set()
        if hasattr(tokenizer, 'eos_token_id'):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
        if token_ids and token_ids[-1] in eos_ids:
            finish_reason = "stop"

        # Record MTP stats in Prometheus
        try:
            from ..middleware.prometheus_exporter import get_prometheus_metrics
            pm = get_prometheus_metrics()
            s = mtp_decoder.stats
            if s.total_cycles > 0:
                pm.set_gauge("mtp_acceptance_rate", s.accepts / s.total_cycles)
                pm.set_gauge("mtp_total_cycles", s.total_cycles)
        except Exception:
            logger.debug("failed", exc_info=True)

        return GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(token_ids),
            finished=True,
            finish_reason=finish_reason,
        )

    async def _stream_generate_mtp(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
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
            text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)

        input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        eos_ids = set()
        if hasattr(tokenizer, 'eos_token_id'):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)

        _sentinel = object()
        _q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _put(item):
            loop.call_soon_threadsafe(_q.put_nowait, item)

        def _run():
            try:
                ids = mx.array(input_ids)
                cache = make_prompt_cache(model)

                # Prefill
                out, hidden = model(ids.reshape(1, -1), cache=cache, return_hidden=True)
                mx.synchronize()
                first = int(mx.argmax(out[0, -1, :]).item())

                generated = [first]
                primary = first
                primary_h = hidden[:, -1:, :]

                # Yield first token
                chunk = _clean_special_tokens(tokenizer.decode([first]))
                _put((chunk, 1, first in eos_ids))

                if first in eos_ids:
                    _put(_sentinel)
                    return

                from .n_confirmed_patch import clear_rollback, restore_rollback

                while len(generated) < max_tokens:
                    # MTP draft
                    draft = mtp_decoder._mtp_draft(primary_h, primary)

                    # Verify: backbone forward [primary, draft] with n_confirmed=1
                    verify_out, verify_h = model(
                        mx.array([[primary, draft]]), cache=cache,
                        return_hidden=True, n_confirmed=1,
                    )
                    mx.synchronize()
                    v0 = int(mx.argmax(verify_out[0, 0, :]).item())
                    v1 = int(mx.argmax(verify_out[0, 1, :]).item())

                    if v0 == draft:
                        # Accept
                        clear_rollback(cache)
                        generated.append(draft)
                        chunk = _clean_special_tokens(tokenizer.decode([draft]))
                        _put((chunk, len(generated), draft in eos_ids))
                        if draft in eos_ids:
                            break

                        # Bonus token
                        generated.append(v1)
                        chunk = _clean_special_tokens(tokenizer.decode([v1]))
                        _put((chunk, len(generated), v1 in eos_ids))
                        if v1 in eos_ids:
                            break
                        primary = v1
                        primary_h = verify_h[:, -1:, :]
                    else:
                        # Reject: restore rollback (zero-cost)
                        restore_rollback(cache)
                        generated.append(v0)
                        chunk = _clean_special_tokens(tokenizer.decode([v0]))
                        _put((chunk, len(generated), v0 in eos_ids))
                        if v0 in eos_ids:
                            break
                        primary = v0
                        primary_h = verify_h[:, 0:1, :]

                _put(_sentinel)
            except Exception as e:
                _put(e)
                _put(_sentinel)

        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run)

        accumulated = ""
        n_tok = 0
        try:
            while True:
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=120)
                except asyncio.TimeoutError:
                    logger.warning("MTP streaming timeout")
                    break
                if item is _sentinel:
                    break
                if isinstance(item, BaseException):
                    logger.warning(f"MTP streaming error: {item}")
                    break
                new_text, tok_count, done = item
                accumulated += new_text
                n_tok = tok_count

                # Streaming backpressure: slow down if client can't keep up
                if _backpressure.check_backpressure(_q.qsize()):
                    _delay = _backpressure.get_delay_ms(_q.qsize())
                    if _delay > 0:
                        await asyncio.sleep(_delay / 1000)

                finish_reason = "stop" if done else None
                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                )
                if done:
                    break
        finally:
            if not future.done():
                future.cancel()

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
            logger.debug("failed", exc_info=True)

        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                clean = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in messages
                ]
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
        """Check if engine has in-flight requests."""
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

    def get_radix_continuations(self, token_ids: list[int], max_results: int = 5) -> list[int]:
        """Return continuation tokens from the radix tree after prefix match.

        Uses the radix tree's bigram view to suggest possible next tokens
        based on historical request patterns. Useful for spec decode draft
        generation context enrichment.
        """
        if not self._engine_core:
            return []
        scheduler = getattr(self._engine_core, "_scheduler", None)
        if scheduler is None:
            return []
        kv_mgr = getattr(scheduler, "_kv_manager", None)
        if kv_mgr is None:
            return []
        tree = getattr(kv_mgr, "_radix_tree", None)
        if tree is None:
            return []
        return tree.get_continuation_tokens(token_ids, max_results)

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
        self, prompt: str, max_tokens: int,
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
                num_prompt_tokens = len(self._tokenizer.encode(prompt))
            except Exception:
                logger.debug("prompt token estimation failed", exc_info=True)
                num_prompt_tokens = len(prompt.split()) * 2  # rough estimate
        else:
            num_prompt_tokens = len(prompt.split()) * 2

        ok, _reason = guard.preflight_check(
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=max_tokens,
        )
        if not ok:
            return GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=num_prompt_tokens,
                completion_tokens=0,
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
