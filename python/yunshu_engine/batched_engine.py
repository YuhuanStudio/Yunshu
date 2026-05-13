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
                pass


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

        # Speculative decoding state (Phase 4)
        self._spec_decoder = None  # SpeculativeDecoder instance
        self._spec_enabled = False

        # N-gram proposer for model-free speculative decoding
        self._ngram_proposer = None  # NgramProposer, created on demand
        self._ngram_stats = {"proposals": 0, "accepted": 0, "total_draft": 0}

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

        # Per-model settings (loaded from model_settings.json + env vars)
        self._settings = None

        # LoRA adapter manager (vLLM pattern)
        self._lora_manager = None

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

        self._model, self._tokenizer = await loop.run_in_executor(executor, _load)
        self._loaded = True

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

        # Warmup generation + cache clear drops RSS from ~3.8GB to ~200MB
        # by forcing OS to reclaim clean mmap pages
        await loop.run_in_executor(executor, _warmup)

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
        self._engine_core = EngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=EngineCoreConfig(stream_interval=self.stream_interval, **arch_kwargs),
            executor=executor,
        )
        self._engine_core.scheduler.config.model_name = self.model_name
        # Wire prefix cache into scheduler for batch-path insert_segments (C16)
        self._engine_core.set_prefix_cache(self._kv_prefix_cache)
        await self._engine_core.start()

        logger.info(f"BatchedEngine started: {self.model_name}")

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
        self._warm_prompts = None
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
        use_engine_loop: bool = False,
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        thinking_budget: int | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
    ) -> GenerationOutput:
        """Non-streaming text generation.

        Args:
            spec_decode: If True, use speculative decoding path when available.
                         Falls back to standard generation if spec decode is
                         not configured or the model lacks spec heads.
            use_engine_loop: If True, route through EngineCore's continuous
                             batching loop. If False (default), use fast path
                             (direct generate_step on executor) for single
                             requests with full GPU utilization.
            logprobs: If True, return log probabilities for each generated token.
            top_logprobs: Number of top logprobs to return per token (max 20).
        """
        if not self._loaded:
            await self.start()

        # Memory guard preflight check
        guard_rejection = self._check_memory_guard(prompt, max_tokens)
        if guard_rejection is not None:
            return guard_rejection

        # Speculative decoding path (Phase 4: single-request EAGLE-3)
        if spec_decode and self._spec_enabled and self._spec_decoder is not None:
            return await self._generate_speculative(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        # N-gram speculative decoding (model-free, CPU-based proposal)
        if spec_decode and self._ngram_proposer is not None and not use_engine_loop:
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
            )

        # Fast path: direct generate_step on executor thread for full GPU utilization
        if not use_engine_loop:
            return await self._generate_fast(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                enable_thinking=enable_thinking,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
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
            seed=seed,
            json_schema=json_schema,
            enable_thinking=enable_thinking,
        )

        if result is None:
            return GenerationOutput(finish_reason="error")

        # Map engine_core finish_reason to OpenAI-compatible finish_reason
        finish_reason = result.finish_reason
        if finish_reason == "memory_exceeded":
            finish_reason = "memory_limit"

        return GenerationOutput(
            text=_clean_special_tokens(result.output_text),
            new_text=_clean_special_tokens(result.output_text),
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
            if thinking_budget is not None:
                try:
                    think_end_token = tokenizer.encode("</think")[-1]
                except Exception:
                    pass

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

            # Cache the completed KV state for future prefix matching
            # Quantize cache layers to save memory (mlx-lm pattern)
            if self._kv_quant_bits is not None:
                _maybe_quantize_kv_cache(
                    cache, self._kv_quant_start,
                    self._kv_quant_group_size, self._kv_quant_bits,
                )
            prefix_cache.add(ids, cache)

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
                    lp_entry["token"] = ""
                    lp_entry["bytes"] = []
                if "top_logprobs" in lp_entry:
                    for tlp in lp_entry["top_logprobs"]:
                        try:
                            tlp["token"] = tokenizer.decode([tlp["token_id"]])
                            tlp["bytes"] = list(tlp["token"].encode("utf-8"))
                        except Exception:
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
        use_engine_loop: bool = False,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Streaming text generation.

        Default uses fast path (direct generate_step on executor) for single
        requests. Set use_engine_loop=True for continuous batching path.
        """
        if not self._loaded:
            await self.start()

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

        # N-gram speculative decoding streaming (model-free)
        if spec_decode and self._ngram_proposer is not None and not use_engine_loop:
            async for output in self._stream_generate_ngram_spec(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty, stop=stop, seed=seed,
            ):
                yield output
            return

        # Fast path: bypass EngineCore for single-request streaming
        if not use_engine_loop:
            async for output in self._stream_generate_fast(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop, stop_token_ids=stop_token_ids,
                seed=seed, enable_thinking=enable_thinking,
            ):
                yield output
            return

        # Engine loop path: continuous batching with scheduler
        await self._ensure_engine_core()
        request_id = await self._engine_core.add_request(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty, logit_bias=logit_bias,
            stop=stop, seed=seed, json_schema=json_schema,
            enable_thinking=enable_thinking,
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
                    pass

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
    ) -> AsyncIterator[GenerationOutput]:
        """Fast streaming: runs generate_step on executor, yields via asyncio.Queue."""
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

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

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0, min_p=min_p)

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
                tid = int(tokens[-1])
                logits[..., tid] -= fp
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
                    new_text = detokenizer.last_segment
                    stop_hit = token in stop_ids
                    suffix_hit = False
                    if not stop_hit and stop_suffixes:
                        if any(detokenizer.text.endswith(s) for s in stop_suffixes):
                            suffix_hit = True
                    _put((new_text, n_tok, stop_hit or suffix_hit))
                    if stop_hit or suffix_hit:
                        prefix_cache.add(ids, cache)
                        mx.synchronize()
                        return
                prefix_cache.add(ids, cache)
                detokenizer.finalize()
                remaining = detokenizer.last_segment
                if remaining:
                    _put((remaining, n_tok, False))
                _put(("", n_tok, True))
                mx.synchronize()

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run)

        accumulated = ""
        n_tok = 0
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
                new_text, tok_count, done = item
                accumulated += new_text
                n_tok = tok_count
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
                    logger.debug(f"Warm prompt file not found: {prompt_text}")
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
            self._ngram_proposer = NgramProposer(NgramConfig(max_n=max_n, k=k))
            # Also create unified SpecProposer wrapper
            from .spec_proposer import NgramSpecProposer
            self._spec_proposer = NgramSpecProposer(NgramConfig(max_n=max_n, k=k))
            logger.info(f"N-gram proposer initialized: max_n={max_n}, k={k}")

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

        # Store config for on-demand decoder creation
        # (actual decoder created when first requested, since it needs a draft model)
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
                    draft_ids = proposer.propose(all_token_ids)
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

                    # Greedy verify: compare model's argmax at each position with draft
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
                            resampled = model_pick  # Use greedy pick (already computed)
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
                    draft_ids = proposer.propose(all_token_ids)
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

        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                clean = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in messages
                ]
                kwargs = {"tokenize": False, "add_generation_prompt": True}
                if thinking is not None:
                    kwargs["enable_thinking"] = thinking
                text = tokenizer.apply_chat_template(clean, **kwargs)
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
            return stats
        return {"model": self.model_name, "loaded": self._loaded}

    def get_kv_cache_stats(self) -> dict:
        """Return KV cache statistics (prefix cache + paged KV)."""
        result = {"prefix_cache": self._kv_prefix_cache.get_stats()}
        if self._engine_core:
            paged = self._engine_core.get_kv_cache_stats()
            result["paged_kv"] = paged
        return result

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
