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
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


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

        # KV prefix cache for multi-turn speedup
        from .kv_prefix_cache import KVPrefixCache
        self._kv_prefix_cache = KVPrefixCache(max_entries=64, min_prefix_length=32)

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
            return load_model(self.model_name)

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
        # Warmup generation + cache clear drops RSS from ~3.8GB to ~200MB
        # by forcing OS to reclaim clean mmap pages
        await loop.run_in_executor(executor, _warmup)

        # Initialize speculative decoding if model supports it (Phase 4)
        self._init_spec_decode()

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
        await self._engine_core.start()

        logger.info(f"BatchedEngine started: {self.model_name}")

    async def stop(self) -> None:
        """Stop engine and release resources."""
        if self._engine_core is not None:
            await self._engine_core.stop()
            self._engine_core = None
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
        seed: int | None = None,
        json_schema: dict | str | None = None,
        spec_decode: bool = False,
        use_engine_loop: bool = False,
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
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

        # Fast path: direct generate_step on executor thread for full GPU utilization
        if not use_engine_loop:
            return await self._generate_fast(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop,
                seed=seed,
                enable_thinking=enable_thinking,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
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
            finish_reason = "context_length_exceeded"

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
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
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

        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else 0,
        )

        def _run():
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
            ids = mx.array(input_ids)
            tokens = []
            token_logprobs = []
            ttft_s = 0.0
            cached_tokens = 0
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()

            # Try KV prefix cache hit
            prefix_cache = self._kv_prefix_cache
            cached_kv, remaining, matched = prefix_cache.get(ids)
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)

            if cached_kv is not None:
                # Only prefill remaining tokens after cache hit
                cached_tokens = matched
                ids_to_prefill = ids[matched:]
            else:
                ids_to_prefill = ids

            gen_t0 = time.perf_counter()
            first = True

            for token, logits in generate_step(
                ids_to_prefill, model, max_tokens=max_tokens, sampler=sampler,
                prompt_cache=cache,
            ):
                if first:
                    ttft_s = time.perf_counter() - gen_t0
                    first = False
                tokens.append(token)
                if logprobs:
                    import numpy as np
                    log_probs_arr = np.array(logits, copy=False)
                    tok_lp = float(log_probs_arr[token])
                    entry = {"token_id": int(token), "logprob": tok_lp}
                    if top_logprobs and top_logprobs > 0:
                        k = min(top_logprobs, len(log_probs_arr))
                        top_indices = np.argpartition(log_probs_arr, -k)[-k:]
                        top_indices = top_indices[np.argsort(log_probs_arr[top_indices])[::-1]]
                        entry["top_logprobs"] = [
                            {"token_id": int(idx), "logprob": float(log_probs_arr[idx])}
                            for idx in top_indices
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
            prefix_cache.add(ids, cache)

            output_text = tokenizer.decode(tokens, skip_special_tokens=True)
            mx.synchronize()
            return tokens, output_text, token_logprobs, ttft_s, cached_tokens

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        tokens, output_text, token_logprobs, ttft_s, cached_tokens = await loop.run_in_executor(executor, _run)

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
        seed: int | None = None,
        json_schema: dict | str | None = None,
        spec_decode: bool = False,
        use_engine_loop: bool = False,
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

        # Fast path: bypass EngineCore for single-request streaming
        if not use_engine_loop:
            async for output in self._stream_generate_fast(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                stop=stop, seed=seed,
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
        )

        finished_normally = False
        try:
            async for output in self._engine_core.stream_outputs(request_id):
                cleaned = _clean_special_tokens(output.new_text)
                finish_reason = output.finish_reason
                if finish_reason == "memory_exceeded":
                    finish_reason = "context_length_exceeded"
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
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Fast streaming: runs generate_step on executor, yields via asyncio.Queue."""
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        tokenizer = self._tokenizer
        model = self._model

        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            prompt = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True,
            )

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

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0)

        # Thread-safe bridge: executor puts via call_soon_threadsafe so the
        # event loop's async consumer is woken for every token.
        _sentinel = object()
        _q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _put(item):
            loop.call_soon_threadsafe(_q.put_nowait, item)

        def _run():
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
            ids = mx.array(input_ids)
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            n_tok = 0

            # KV prefix cache for streaming
            prefix_cache = self._kv_prefix_cache
            cached_kv, remaining, matched = prefix_cache.get(ids)
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids

            for token, _logits in generate_step(
                ids_to_prefill, model, max_tokens=max_tokens, sampler=sampler,
                prompt_cache=cache,
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
            _put(("", n_tok, True))
            mx.synchronize()
            _put(_sentinel)

        from .mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run)

        accumulated = ""
        n_tok = 0
        try:
            while True:
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=300)
                except asyncio.TimeoutError:
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

    def _init_spec_decode(self) -> None:
        """Initialize speculative decoding if the model supports it.

        Called during start() after model loading. Checks for spec heads in the
        model config and creates a SpeculativeDecoder if detected.
        """
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

            draft_result, verify_result = await loop.run_in_executor(executor, _spec_step)

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
                pass

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

        Returns a GenerationOutput with finish_reason="context_length_exceeded"
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

        ok, reason = guard.preflight_check(
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=max_tokens,
        )
        if not ok:
            return GenerationOutput(
                finished=True,
                finish_reason="context_length_exceeded",
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
