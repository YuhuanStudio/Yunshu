from __future__ import annotations

"""Engine stream extracted from batched_engine.

Patchable helpers resolve through the compatibility facade. Concrete self types
retain the shared BatchedEngine state; misc ignores allow that mixin self type.
"""

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from contextlib import suppress

from .context_window import ContextBudgetError, reject_overlong_prompt
from .fast_path_stats import FastPathStats
from .stream_bridge import StreamBridge, make_stream_queue
from .text_utils import StopHoldbackBuffer


class EngineStreamMixin:
    async def _stream_generate_fast(  # type: ignore[misc]
        self: _engine.BatchedEngine,
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
        lora_adapter: str | None = None,
        kv_cache_breakpoints: list[int] | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        suppress_tokens: list[int] | None = None,
    ) -> AsyncIterator[_engine.GenerationOutput]:
        """Fast streaming: runs generate_step on executor, yields via asyncio.Queue.

        Wraps generation with a StreamingBackpressureController to prevent OOM
        on slow clients.

        """
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        # Streaming optimizer components
        from .streaming_optimizer import StreamingBackpressureController

        _backpressure = StreamingBackpressureController(max_queue_size=100)

        from .grammar_compile import prepare_constraint

        await prepare_constraint(json_schema)
        tokenizer = self._tokenizer
        model = self._model

        # Model-specific preprocessing
        if self._preprocessor_registry is not None:
            try:
                model_config = {"model_type": self.model_name or ""}
                if hasattr(model, "config") and hasattr(model.config, "model_type"):
                    model_config["model_type"] = model.config.model_type
                preprocessor = self._preprocessor_registry.detect(model_config)
                if preprocessor is not None:
                    processed = preprocessor.preprocess(prompt, tokenizer=tokenizer)
                    if processed.token_ids:
                        prompt = processed.token_ids
            except Exception:
                _engine.logger.debug(
                    "model preprocessor failed in streaming", exc_info=True
                )

        # ── Pre-encoding context window truncation (message-level) ──
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            try:
                _max_ctx_pre = getattr(model, "max_seq_len", None)
                if _max_ctx_pre is None:
                    _max_ctx_pre = getattr(
                        getattr(model, "config", None), "max_seq_len", None
                    ) or getattr(getattr(model, "args", None), "max_seq_len", None)
                if _max_ctx_pre and _max_ctx_pre > 0:
                    _thinking_overhead = (
                        thinking_budget if (thinking_budget and enable_thinking) else 0
                    )
                    _generation_budget = max_tokens + _thinking_overhead
                    _est_tokens = sum(
                        len(str(m.get("content", ""))) // 4 + 4 for m in prompt
                    )
                    if _est_tokens + _generation_budget > _max_ctx_pre:
                        from .context_window import ContextWindowManager

                        ctx_mgr = ContextWindowManager(
                            token_counter=lambda text: len(tokenizer.encode(text)),
                        )
                        result = ctx_mgr.compute_truncation(
                            messages=prompt,
                            max_tokens=_max_ctx_pre - _generation_budget,
                            strategy="importance_aware",
                        )
                        prompt = result.messages
                        result.publish()
                        result.raise_if_cannot_fit()
                        _engine.logger.debug(
                            "Streaming fast path pre-encode truncation: estimated %d → %d tokens",
                            _est_tokens,
                            result.truncated_token_count,
                        )
            except ContextBudgetError:
                raise
            except Exception:
                _engine.logger.debug(
                    "context window truncation skipped in streaming fast path",
                    exc_info=True,
                )

        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # Route through _apply_chat_template (NOT raw) — same reason as the
            # non-streaming fast path: tool_calls.arguments str→dict normalization,
            # the family message adapter, dangling-<think> closing, and the
            # enable_thinking-unsupported retry. The default streaming path otherwise
            # broke multi-turn tool conversations / Mistral-Gemma role rules.
            prompt = self._apply_chat_template(prompt, enable_thinking=enable_thinking)

        input_ids = self._encode_prompt(tokenizer, prompt)
        prompt_tokens = len(input_ids)

        # Guard against empty prompt (same as _generate_fast)
        if not input_ids:
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is not None:
                input_ids = [bos_id]
            else:
                eos_id = getattr(tokenizer, "eos_token_id", 1)
                input_ids = [eos_id]
            prompt_tokens = len(input_ids)

        # ── Context window: clamp generation + truncate over-long prompt ──
        # Same fix as the non-streaming fast path — resolve the REAL
        # context window (mlx-lm uses max_position_embeddings, not max_seq_len, so
        # the old resolution was dead for most models) and clamp max_tokens so
        # prompt+generation can't decode past the window into RoPE garbage.
        _max_ctx = _engine._resolve_model_max_ctx(model)
        if _max_ctx and _max_ctx > 0:
            reject_overlong_prompt(prompt_tokens, _max_ctx)
            _room = _max_ctx - prompt_tokens
            if _room >= 1 and max_tokens > _room:
                _engine.logger.info(
                    "Streaming fast path clamped max_tokens %d → %d to fit context "
                    "(prompt=%d, ctx=%d)",
                    max_tokens,
                    _room,
                    prompt_tokens,
                    _max_ctx,
                )
                max_tokens = _room

        # Convert KV cache breakpoint char offsets to token positions.
        _stream_kv_breakpoints: list[int] = []
        if kv_cache_breakpoints and isinstance(prompt, str) and len(prompt) > 0:
            for char_off in kv_cache_breakpoints:
                if char_off <= 0 or char_off > len(prompt):
                    continue
                try:
                    prefix_ids = tokenizer.encode(prompt[:char_off])
                    _stream_kv_breakpoints.append(len(prefix_ids))
                except Exception:
                    _engine.logger.debug(
                        "KV breakpoint char->token conversion failed at offset %d",
                        char_off,
                        exc_info=True,
                    )

        # normalize eos ids — some tokenizers (Qwen3.6-27B) expose eos_token_ids
        # as a BARE INT, which crashed stop_ids.update(...) with "'int' object is not
        # iterable". The non-streaming _generate_fast got this fix; the streaming
        # twin was missed → every streaming request on that tokenizer raised → client got
        # finish_reason="error", no content. Accept both shapes.
        stop_ids = set()
        _user_stop_ids: set[int] = set()  # ids of single-token USER stop strings
        _user_stop_hit = [False]  # a user stop (not EOS) ended generation
        # collect EOS separately (ignore_eos / min_tokens) — see _generate_fast.
        _eos_ids: set[int] = set()
        _eid = getattr(tokenizer, "eos_token_id", None)
        if _eid is not None:
            _eos_ids.update(_eid if isinstance(_eid, (list, tuple, set)) else (_eid,))
        _eids = getattr(tokenizer, "eos_token_ids", None)
        if _eids is not None:
            _eos_ids.update(
                _eids if isinstance(_eids, (list, tuple, set)) else (_eids,)
            )
        _eos_ids.update(_engine._read_config_eos_ids(self.model_name))
        if not ignore_eos:
            stop_ids.update(_eos_ids)

        # : always string-match user stops (see _generate_fast)
        # — a bare stop's encoded id rarely equals the space-prefixed token the
        # model emits, so token-id matching alone silently missed most stops.
        stop_suffixes = []
        if stop:
            for s in stop:
                if not s:
                    continue
                ids = tokenizer.encode(s)
                if len(ids) == 1:
                    stop_ids.add(ids[0])
                    _user_stop_ids.add(ids[0])
                stop_suffixes.append(s)
        if stop_token_ids:
            stop_ids.update(stop_token_ids)

        # Multi-token stop hold-back : withhold any streamed text that
        # could be the start of a multi-token stop string so its prefix never
        # leaks before the match completes (SSE is append-only). Active whenever
        # string stops exist. When logprobs is off the steady-state emit uses
        # feed() (per-token lp_entry is None anyway). When logprobs is ON we
        # route through feed_lp() instead, which keeps each token's logprob
        # aligned to its text. Previously hold-back was disabled under logprobs
        # (`and not logprobs`), which let a multi-token stop prefix LEAK into the
        # streamed output when logprobs was on . When inactive
        # (no stops), the buffer is pure passthrough.
        _hb_active = bool(stop_suffixes)
        _hb = StopHoldbackBuffer(stop_suffixes if _hb_active else None)

        # route temp>0 through _build_temp_sampler, like the non-streaming
        # fast path (3160). The bare make_sampler uses mlx-lm's categorical_sampling,
        # which is @mx.compile(inputs=mx.random.state) — the PRNG-cache trap that
        # makes `seed` a no-op (non-reproducible) and collapses concurrent/n>1 temp>0
        # STREAMING requests to identical token streams. This streaming path was the
        # un-propagated sibling of the non-streaming fix.
        if temperature is not None and temperature > 1e-6:
            sampler = _engine._build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k and top_k > 0 else 0,
                min_p=min_p if min_p else 0.0,
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

        # Grammar constraint for streaming fast path
        if json_schema is not None:
            try:
                sampler = _engine._build_constrained_sampler(
                    sampler, json_schema, tokenizer
                )
            except Exception:
                _engine.logger.warning(
                    "Grammar constraint setup failed in streaming", exc_info=True
                )

        # Build logits processors for penalty/bias params
        # length-gated KV-quant bits for THIS request (stream path).
        _req_kv_bits = self._effective_kv_quant_bits(prompt_tokens + max_tokens)
        _custom_logits_processors = logits_processors or []
        logits_processors = []
        _tool_processor = (
            self._tool_call_processor(input_ids) if json_schema is None else None
        )
        if repetition_penalty != 1.0:

            def _repetition_penalty(tokens, logits, rp=repetition_penalty, ctx=20):
                if len(tokens) > 0:
                    recent = tokens[-ctx:]
                    import mlx.core as _mx

                    sel = logits[..., recent]
                    sel = _mx.where(sel < 0, sel * rp, sel / rp)
                    logits[..., _mx.array(recent)] = sel
                return logits

            logits_processors.append(_repetition_penalty)
        if frequency_penalty != 0.0 or presence_penalty != 0.0:
            # Maintain incremental counts dict instead of rebuilding
            # from full tokens list every step (O(T²) → O(T) total). The
            # closure captures a mutable state dict so successive calls only
            # observe the newest token.
            _fp_state: dict[str, object] = {"counts": {}, "last_len": -1}

            # The prompt offset inside the processor is 1, NOT the full
            # prompt length. mlx-lm prefills all-but-the-last prompt token OUTSIDE
            # _step, so the `tokens` accumulator handed to logits_processors is
            # [last_prompt_token, gen1, gen2, ...]. With n_prompt=prompt_tokens the
            # guard `cur_len <= n_prompt` made frequency/presence penalty INERT for
            # the first ~prompt_len generated tokens (short completions never
            # penalized at all) and then tokens[n_prompt:] under-counted repeats.
            def _freq_pres_penalty(
                tokens,
                logits,
                fp=frequency_penalty,
                pp=presence_penalty,
                n_prompt=1,
                _st=_fp_state,
            ):
                counts: dict[int, int] = _st["counts"]  # type: ignore[assignment]
                last_len = int(_st["last_len"])  # type: ignore[arg-type]
                cur_len = len(tokens)
                if cur_len <= n_prompt:
                    _st["last_len"] = cur_len
                    return logits
                # Reset/rebuild if tokens shrank (spec-decode rollback) or
                # we have not yet started incremental accounting.
                if cur_len < last_len or last_len < n_prompt:
                    counts = {}
                    for t in tokens[n_prompt:]:
                        counts[int(t)] = counts.get(int(t), 0) + 1
                    _st["counts"] = counts
                else:
                    start = max(last_len, n_prompt)
                    for t in tokens[start:]:
                        counts[int(t)] = counts.get(int(t), 0) + 1
                _st["last_len"] = cur_len
                for tid, cnt in counts.items():
                    # OpenAI permits frequency/presence penalty in [-2.0, 2.0];
                    # negative values BOOST the token (encourage repetition).
                    # Guarding with `> 0` silently dropped negative penalties
                    # (verified: fp=-2.0 produced output identical to fp=0).
                    if fp != 0.0:
                        logits[..., tid] = logits[..., tid] - fp * cnt
                    if pp != 0.0 and cnt > 0:
                        logits[..., tid] = logits[..., tid] - pp
                return logits

            logits_processors.append(_freq_pres_penalty)
        if logit_bias:

            def _logit_bias_proc(_tokens, logits, biases=logit_bias):
                # Skip token ids outside [0, vocab) — a user-supplied out-of-range
                # or negative id would otherwise index out of bounds and crash the
                # request (MLX: "Cannot squeeze axis 1 with size 0"). OpenAI ignores
                # invalid logit_bias token ids rather than erroring.
                vocab = logits.shape[-1]
                for tid, bias in biases.items():
                    if 0 <= tid < vocab:
                        logits[..., tid] = logits[..., tid] + bias
                return logits

            logits_processors.append(_logit_bias_proc)

        # suppress_tokens + min_tokens (mirror _generate_fast).
        if suppress_tokens:
            _sup = [int(t) for t in suppress_tokens]

            def _suppress_proc(_tokens, logits, sup=_sup):
                vocab = logits.shape[-1]
                for tid in sup:
                    if 0 <= tid < vocab:
                        logits[..., tid] = -float("inf")
                return logits

            logits_processors.append(_suppress_proc)
        # skip min_tokens EOS-masking when a JSON/grammar constraint is
        # active — the un-propagated streaming sibling of the non-streaming
        # fix (3336). The constraint already enforces a structural minimum, and
        # masking EOS at its DONE state leaves the constrained sampler an
        # all-(-inf) allowed set → its argmax fallback emits an INVALID non-EOS
        # token after a complete JSON value.
        if min_tokens and min_tokens > 0 and stop_ids and json_schema is None:
            _mask_ids = list(stop_ids)

            def _min_tokens_proc(tokens, logits, ids=_mask_ids, floor=int(min_tokens)):
                if (len(tokens) - 1) < floor:
                    vocab = logits.shape[-1]
                    for tid in ids:
                        if 0 <= tid < vocab:
                            logits[..., tid] = -float("inf")
                return logits

            logits_processors.append(_min_tokens_proc)

        # SAMP-2: Wrap user-provided custom logits processors to adapt signature.
        if _custom_logits_processors:
            logits_processors.extend(
                _engine._wrap_custom_logits_processor(p)
                for p in _custom_logits_processors
            )
        if _tool_processor is not None:
            logits_processors.append(_tool_processor)

        # Thread-safe bridge: executor puts via call_soon_threadsafe so the
        # event loop's async consumer is woken for every token.
        _sentinel = object()
        _q: asyncio.Queue = make_stream_queue(512)
        loop = asyncio.get_running_loop()
        # Cross-thread cancel: set by the async consumer on timeout so the
        # GPU generation loop in _run_inner stops producing tokens.
        _timeout_cancel = threading.Event()

        _bridge = StreamBridge(
            loop,
            _q,
            lambda it: it is _sentinel or isinstance(it, BaseException),
            on_overflow=_timeout_cancel.set,
        )

        def _put(item):
            # Atomic reserve-then-schedule; terminals bypass capacity (B12).
            return _bridge.put(item)

        # (LoRA concurrency keystone): acquire+apply / release+restore moved into
        # _run_with_lora on the executor thread (serialized with generate_step), not here on
        # the event loop. See the non-streaming _generate_fast for the rationale.

        # Inflight prefix sharing: defined at _run level so it's accessible
        # from exception handlers even if _run_inner crashes early
        _inflight_req_id = f"fp-s-{int(time.monotonic() * 1e6)}"

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_inflight_req_id)
            except Exception:
                _engine.logger.debug(
                    "inflight prefix unregister failed in streaming", exc_info=True
                )

        _stream_gen_t0 = time.perf_counter()  # TTFT timing for streaming fast path
        _fp_stats = FastPathStats(cancel_event, prompt_tokens)
        # a TOTAL-generation deadline for the streaming path. The consumer's
        # asyncio.wait_for(_q.get(), timeout_seconds) only catches an INACTIVITY gap
        # (no token for timeout_seconds); a stream that keeps emitting tokens steadily
        # would run to max_tokens, blowing far past the user's timeout. The non-streaming
        # path enforces gen_t0 + timeout_seconds as a hard total deadline — mirror it in
        # the streaming GPU loop so `timeout` means the same (total wall time) for both.
        _stream_timeout_deadline = (
            _stream_gen_t0 + timeout_seconds if timeout_seconds else None
        )
        _stream_ttft_recorded = [False]  # mutable box to track first-token observation
        _stream_ttft_box = [0.0]  # mutable box for TTFT value
        # define BEFORE the _run_inner closure is submitted to the
        # executor — the closure reads/writes _cached_tokens_box[0] (line ~4377),
        # and if the executor thread reached it before the old definition (which
        # sat AFTER run_in_executor), it raised NameError. Defining it here
        # closes the ordering race.
        _cached_tokens_box = [0]  # mutable box for inner _run_inner to set
        _stream_itl_samples = []  # ITL samples for streaming fast path

        # Create detokenizer at _run scope so the error handler can finalize
        # it even if _run_inner() crashes before its own cleanup paths run.
        _detokenizer_ref = [None]  # mutable box shared with _run_inner
        # Hybrid-model guard : the prefix cache relies on trimming the
        # cached KV; hybrid/recurrent models (Qwen3.5 ArraysCache) are NOT
        # trimmable and reusing such a cache silently corrupts the recurrent
        # layers → grossly wrong logits. The non-streaming path disables the
        # cache for them; the streaming path must do the same (was missing).
        # Also bypass for seeded sampling (reproducibility needs a fresh cache).
        _stream_bypass_cache = (
            (seed is not None and temperature > 0)
            or not self._cache_supports_trim(self._model)
            # bypass when a LoRA adapter is active — the prefix cache is not
            # adapter-keyed, so reusing KV computed under a different adapter (or base)
            # silently serves the wrong adapter's state. Mirrors _generate_fast.
            or (lora_adapter is not None)
        )
        prefix_cache = None if _stream_bypass_cache else self._kv_prefix_cache

        def _save_breakpoint_prefixes(token_ids, kv_cache):
            """Save KV prefix cache entries at breakpoint positions."""
            if not _stream_kv_breakpoints or prefix_cache is None:
                return
            from .kv_prefix_cache import cache_length as _cache_len_bp

            for bp_pos in _stream_kv_breakpoints:
                if bp_pos < len(token_ids) and bp_pos >= 32:
                    bp_tokens = token_ids[:bp_pos]
                    # trim relative to the LIVE cache length (which
                    # includes generated tokens), not len(token_ids) — see the
                    # non-streaming twin. Storing bp_pos+num_generated tokens under a
                    # bp_pos-token key leaked the prior completion into a future hit.
                    trim_count = max(0, _cache_len_bp(kv_cache) - bp_pos)
                    try:
                        bp_cache = prefix_cache._snapshot_cache(
                            kv_cache, trim=trim_count
                        )
                        prefix_cache.add(bp_tokens, bp_cache)
                    except Exception:
                        _engine.logger.debug(
                            "KV breakpoint prefix add failed at pos %d",
                            bp_pos,
                            exc_info=True,
                        )

        def _run_inner():
            import mlx.core as mx

            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            if self._lookahead_reasoning is not None:
                self._lookahead_reasoning._in_thinking = False
                self._lookahead_reasoning._thinking_tokens = []
                self._lookahead_reasoning._recent_accepts = []
            _detokenizer_ref[0] = detokenizer
            n_tok = 0
            thinking_tokens_used = 0
            think_end_token = None
            think_start_token = None
            _first_token = True
            _in_thinking = False
            _thinking_tokens: list[int] = []

            # Prefill progress tracking for streaming fast path
            _prefill_req_id = f"fp-s-{id(_run_inner)}-{int(time.monotonic() * 1e6)}"
            _prefill_tracker = None
            try:
                from .prefill_progress import get_prefill_tracker

                _prefill_tracker = get_prefill_tracker()
                _prefill_tracker.update(
                    _prefill_req_id, 0, prompt_tokens, self.model_name or "default"
                )
            except Exception:
                _engine.logger.debug("prefill tracker setup failed", exc_info=True)
                _prefill_tracker = None

            # resolve the think tokens UNCONDITIONALLY (was gated on
            # `thinking_budget is not None or enable_thinking`). enable_thinking defaults
            # to None end-to-end, yet Qwen3/Qwen3.5/DeepSeek-R1 chat templates are
            # default-ON: they inject the OPENING <think> into the PROMPT, so a plain
            # default-param request still generates a chain-of-thought. With the old gate,
            # think_end_token stayed None → the pre-seed below was skipped →
            # _in_thinking never flipped → the ENTIRE CoT leaked into visible delta.content
            # and reasoning_tokens stayed 0 (streaming diverged from the non-streaming fast
            # path, which recovers via the post-hoc closing-only reasoning parser). The
            # single-token guard already prevents false positives; the pre-seed is gated on
            # the prompt actually ending with an open <think>, so non-thinking models are
            # unaffected.
            # resolve via the bracketed-form helper (the bare "</think" encoded to
            # 2 tokens for Qwen3/DeepSeek-R1 → guard failed → CoT leaked into content).
            think_start_token, think_end_token = _engine._resolve_think_token_ids(
                tokenizer
            )

            # chat-template-injected <think> models (Qwen3/Qwen3.5/
            # DeepSeek-R1) put the OPENING <think> in the PROMPT, so the model
            # OUTPUT contains only </think>. Without pre-seeding, _in_thinking
            # never flipped True → the ENTIRE chain-of-thought leaked into visible
            # delta.content and reasoning_tokens stayed 0 (streaming diverged from
            # the non-streaming path, which uses the closing-only reasoning parser).
            # Pre-seed when the rendered prompt ends with an open <think>.
            if think_end_token is not None and not _in_thinking:
                try:
                    from .thinking_budget import detect_needs_think_prefix

                    if detect_needs_think_prefix(list(input_ids), tokenizer):
                        _in_thinking = True
                        _thinking_tokens = []
                except Exception:
                    _engine.logger.debug("think-prefix detection failed", exc_info=True)

            # KV prefix cache for streaming
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
                        _engine.logger.debug(
                            "paged KV pressure eviction failed", exc_info=True
                        )
            try:
                cached_kv, _, matched = (
                    prefix_cache.get(ids)
                    if prefix_cache is not None
                    else (None, None, 0)
                )
            except Exception:
                _engine.logger.warning(
                    "KV prefix cache get failed in streaming — falling back to full prefill",
                    exc_info=True,
                )
                cached_kv, _, matched = None, None, 0
            cache = (
                cached_kv
                if cached_kv is not None
                else _engine._create_prompt_cache_with_quant(
                    model, _req_kv_bits, self._kv_quant_group_size
                )
            )
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids
            if len(ids_to_prefill) == 0 and len(ids) > 0:
                ids_to_prefill = ids[-1:]
            _stream_cached_tokens = matched
            _cached_tokens_box[0] = _stream_cached_tokens

            # Inflight prefix sharing ()
            _inflight_entry = None
            if cached_kv is None:
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    _tracker = get_inflight_tracker()
                    _inflight_entry = _tracker.find_prefix(
                        [int(t) for t in ids], self.model_name or ""
                    )
                    # BUG #1: live-borrow reuse disabled — see the
                    # non-streaming site for the full rationale (empty-prefill
                    # crash + cross-request corruption against a growing donor
                    # cache, unsafe under the now-working engine loop).
                except Exception:
                    _engine.logger.debug(
                        "inflight prefix lookup failed in streaming", exc_info=True
                    )

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
                _engine.logger.debug(
                    "inflight prefix register failed in streaming", exc_info=True
                )

            _lprocs = logits_processors if logits_processors else None
            _last_stream_tok_time = 0.0
            # mlx-lm's generate_step quantizes the cache per-step,
            # but RotatingKVCache.to_quantized() is NYI → passing kv_bits crashes the
            # stream for sliding-window models (Gemma-3/Gemma-4/gpt-oss/Cohere2). Suppress
            # KV quant for those (matches mlx-lm's CLI + the non-streaming guard above).
            _stream_kv_bits = _req_kv_bits
            if _stream_kv_bits is not None:
                try:
                    from mlx_lm.models.cache import RotatingKVCache

                    if any(isinstance(c, RotatingKVCache) for c in cache):
                        _stream_kv_bits = None
                except Exception:
                    pass
            _fp_stats.admit(
                _stream_cached_tokens,
                len(ids_to_prefill),
                *_engine._prefix_cache_provenance(
                    prefix_cache, False, _stream_cached_tokens
                ),
            )
            with _engine._wired_limit_ctx(model):
                for token, logits in generate_step(
                    ids_to_prefill,
                    model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=cache,
                    logits_processors=_lprocs,
                    prefill_step_size=_engine._prefill_step_size(),
                    prompt_progress_callback=_fp_stats.progress,
                    # the streaming path had NO KV-quant — mlx-lm
                    # quantizes the cache per-step internally, but only when these
                    # are passed, so YUNSHU_KV_QUANT_BITS gave zero in-flight memory
                    # benefit on the (default) streaming path. None → mlx-lm no-ops.
                    kv_bits=_stream_kv_bits,
                    kv_group_size=self._kv_quant_group_size,
                    quantized_kv_start=self._kv_quant_start,
                ):
                    n_tok += 1
                    _fp_stats.token(n_tok)
                    # Check stop_ids BEFORE adding to detokenizer to avoid emitting stop text
                    stop_hit = token in stop_ids
                    suffix_hit = False
                    if stop_hit and token in _user_stop_ids:
                        _user_stop_hit[0] = True
                    if not stop_hit:
                        detokenizer.add_token(token)
                        if stop_suffixes:
                            suffix_hit = any(
                                detokenizer.text.endswith(s) for s in stop_suffixes
                            )
                            if suffix_hit:
                                _user_stop_hit[0] = True
                    # Compute per-token logprobs (same pattern as _generate_fast)
                    _lp_entry = None
                    if logprobs and logits is not None:
                        import mlx.core as _mx

                        _lp_logits = logits.astype(_mx.float32)
                        _log_probs = _lp_logits - _mx.logsumexp(
                            _lp_logits, axis=-1, keepdims=True
                        )
                        _tok_lp = float(_log_probs[token])
                        if _tok_lp != _tok_lp or _tok_lp == float("-inf"):
                            _tok_lp = -100.0
                        _lp_entry = {"token_id": int(token), "logprob": _tok_lp}
                        if top_logprobs and top_logprobs > 0:
                            _k = min(top_logprobs, _log_probs.shape[0])
                            _sorted_idx = _mx.argsort(-_log_probs)
                            _top_k_idx = _sorted_idx[:_k]
                            _top_entries = []
                            for j in range(_k):
                                _tlp = float(_log_probs[int(_top_k_idx[j])])
                                if _tlp != _tlp or _tlp == float("-inf"):
                                    _tlp = -100.0
                                _top_entries.append(
                                    {"token_id": int(_top_k_idx[j]), "logprob": _tlp}
                                )
                            _lp_entry["top_logprobs"] = _top_entries
                    # Check cancellation
                    if _engine._is_cancelled(cancel_event):
                        mx.synchronize()
                        # Flush remaining detokenizer bytes before cancelling
                        try:
                            detokenizer.finalize()
                            remaining = detokenizer.last_segment
                            if _hb_active:
                                # Not a stop match — flush held-back text (real
                                # output that merely looked like a stop-prefix).
                                remaining = _hb.feed(remaining) + _hb.flush()
                            if remaining:
                                _put(
                                    (
                                        remaining,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        "reasoning" if _in_thinking else "normal",
                                    )
                                )
                        except Exception:
                            _engine.logger.debug(
                                "detokenizer finalize in cancel handler failed",
                                exc_info=True,
                            )
                        # Emit terminal stop chunk so consumer sees finished=True
                        _put(
                            (
                                "",
                                n_tok,
                                "stop",
                                len(_thinking_tokens),
                                None,
                                "reasoning" if _in_thinking else "normal",
                            )
                        )
                        if _prefill_tracker is not None:
                            _prefill_tracker.remove(_prefill_req_id)
                        _unregister_inflight()
                        return
                    # Check timeout-driven cancel from consumer (inactivity) OR the total
                    # generation deadline — both finalize + emit a "timeout"
                    # terminal chunk and stop the GPU loop.
                    if _timeout_cancel.is_set() or (
                        _stream_timeout_deadline is not None
                        and time.perf_counter() > _stream_timeout_deadline
                    ):
                        mx.synchronize()
                        try:
                            detokenizer.finalize()
                            remaining = detokenizer.last_segment
                            if _hb_active:
                                # Not a stop match — flush held-back text (real
                                # output that merely looked like a stop-prefix).
                                remaining = _hb.feed(remaining) + _hb.flush()
                            if remaining:
                                _put(
                                    (
                                        remaining,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        "reasoning" if _in_thinking else "normal",
                                    )
                                )
                        except Exception:
                            _engine.logger.debug(
                                "detokenizer finalize in timeout cancel failed",
                                exc_info=True,
                            )
                        _put(
                            (
                                "",
                                n_tok,
                                "timeout",
                                len(_thinking_tokens),
                                None,
                                "reasoning" if _in_thinking else "normal",
                            )
                        )
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
                                _prefill_req_id,
                                prompt_tokens,
                                prompt_tokens,
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
                    if suffix_hit and not stop_hit and not _hb_active:
                        # (Non-hold-back path) Trim ONLY the matched stop suffix
                        # from this segment so leading non-stop text is still
                        # streamed (e.g. emit "done" from "doneSTOP"). When
                        # _hb_active, the hold-back buffer handles this via
                        # take_stopped() instead, so skip the in-segment trim.
                        for _s in stop_suffixes:
                            if new_text.endswith(_s):
                                new_text = new_text[: -len(_s)]
                                break
                        else:
                            new_text = ""
                    # Exclude stop/suffix-triggering token from completion count
                    if stop_hit or suffix_hit:
                        n_tok -= 1
                    # Track thinking segment boundaries BEFORE budget check
                    # so that a natural </think token is detected first and
                    # the budget enforcement does not append a duplicate.
                    # Skip stop/suffix tokens from thinking tracking since they
                    # are excluded from completion_tokens (invariant must hold).
                    if think_start_token is not None and not (stop_hit or suffix_hit):
                        if not _in_thinking and token == think_start_token:
                            _in_thinking = True
                            _thinking_tokens = []
                            self._lookahead_reasoning.check_thinking_state_text(
                                "<think"
                            )
                        elif _in_thinking:
                            _thinking_tokens.append(token)
                            if token == think_end_token:
                                _in_thinking = False
                                self._lookahead_reasoning.check_thinking_state_text(
                                    "</think"
                                )
                    # Thinking budget enforcement in streaming.
                    # Only force-append think_end_token + add to detokenizer
                    # if the current token is NOT already the natural closing tag.
                    if thinking_budget is not None and _in_thinking:
                        thinking_tokens_used += 1
                        if (
                            thinking_tokens_used >= thinking_budget
                            and think_end_token is not None
                        ):
                            # Token was already appended at line ~3298 — don't
                            # duplicate it. Only force-emit the closing tag.
                            _in_thinking = False
                            # Emit current token's text first (still reasoning content)
                            if new_text:
                                _put(
                                    (
                                        new_text,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        _lp_entry,
                                        "reasoning",
                                    )
                                )
                            # Only force-emit closing tag if the token isn't already it
                            if token != think_end_token:
                                n_tok += 1  # Count the forced closing tag token
                                detokenizer.add_token(think_end_token)
                                _thinking_tokens.append(think_end_token)
                                _end_text = detokenizer.last_segment
                                if _end_text:
                                    _put(
                                        (
                                            _end_text,
                                            n_tok,
                                            None,
                                            len(_thinking_tokens),
                                            None,
                                            "reasoning",
                                        )
                                    )
                            detokenizer.finalize()
                            _remaining = detokenizer.last_segment
                            if _hb_active:
                                _remaining = _hb.feed(_remaining) + _hb.flush()
                            if _remaining:
                                _put(
                                    (
                                        _remaining,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        "normal",
                                    )
                                )
                            # thinking_budget exhausted → "length" (matches fast-path and
                            # OpenAI semantics: budget == token limit)
                            _put(
                                (
                                    "",
                                    n_tok,
                                    "length",
                                    len(_thinking_tokens),
                                    None,
                                    "normal",
                                )
                            )
                            if prefix_cache is not None:
                                prefix_cache.add(ids, cache)
                            _save_breakpoint_prefixes(ids, cache)
                            mx.synchronize()
                            if _prefill_tracker is not None:
                                _prefill_tracker.remove(_prefill_req_id)
                            _unregister_inflight()
                            return
                    _is_stopping = stop_hit or suffix_hit
                    # the closing </think> token flipped _in_thinking to
                    # False at line ~5293 BEFORE this state is computed, so its own
                    # text ("</think>") would leak into visible content. Keep the
                    # boundary token classified as reasoning so the tag stays out of
                    # delta.content (matches the non-streaming parser stripping it).
                    _just_closed = (
                        think_end_token is not None
                        and token == think_end_token
                        and not (stop_hit or suffix_hit)
                    )
                    _cur_state = (
                        "reasoning" if (_in_thinking or _just_closed) else "normal"
                    )
                    if _is_stopping:
                        # Emit current text without finish_reason so the consumer
                        # reads it before breaking on done=True below.
                        if _hb_active and logprobs:
                            # Logprob-aware hold-back: each token's lp rides the
                            # chunk that first reveals its text; stop prefixes are
                            # withheld so they never leak with logprobs on.
                            for _t, _lp in _hb.feed_lp(new_text, _lp_entry):
                                if _t or _lp is not None:
                                    _put(
                                        (
                                            _t,
                                            n_tok,
                                            None,
                                            len(_thinking_tokens),
                                            _lp,
                                            _cur_state,
                                        )
                                    )
                        elif _hb_active:
                            _emit = _hb.feed(new_text)
                            if _emit:
                                _put(
                                    (
                                        _emit,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        _lp_entry,
                                        _cur_state,
                                    )
                                )
                        elif new_text:
                            _put(
                                (
                                    new_text,
                                    n_tok,
                                    None,
                                    len(_thinking_tokens),
                                    _lp_entry,
                                    _cur_state,
                                )
                            )
                    else:
                        if _hb_active and logprobs:
                            for _t, _lp in _hb.feed_lp(new_text, _lp_entry):
                                if _t or _lp is not None:
                                    _put(
                                        (
                                            _t,
                                            n_tok,
                                            None,
                                            len(_thinking_tokens),
                                            _lp,
                                            _cur_state,
                                        )
                                    )
                        elif _hb_active:
                            _emit = _hb.feed(new_text)
                            if _emit:
                                _put(
                                    (
                                        _emit,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        _lp_entry,
                                        _cur_state,
                                    )
                                )
                        else:
                            _put(
                                (
                                    new_text,
                                    n_tok,
                                    None,
                                    len(_thinking_tokens),
                                    _lp_entry,
                                    _cur_state,
                                )
                            )
                    # A complete stop that landed MID-segment (e.g. one token decodes to
                    # "aSTOPb") is missed by the endswith() suffix_hit but is now held in
                    # the buffer — fire the stop so generation ends and take_stopped()
                    # drops it (the streaming backstop the non-streaming find() path had).
                    if _hb_active and not _is_stopping and _hb.contains_stop():
                        _is_stopping = True
                    if _is_stopping:
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        if _hb_active:
                            # Feed the final bytes through the buffer, then flush
                            # with the matched stop trimmed. This correctly
                            # discards a multi-token stop whose prefix was held
                            # back across earlier chunks (e.g. "\n\n" as two "\n"
                            # tokens) without leaking it.
                            _emit = _hb.feed(_remaining)
                            if _emit:
                                _put(
                                    (
                                        _emit,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        _cur_state,
                                    )
                                )
                            _tail = _hb.take_stopped()
                            if _tail:
                                _put(
                                    (
                                        _tail,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        _cur_state,
                                    )
                                )
                        else:
                            # Trim stop suffix from remaining text — the suffix may
                            # span multiple tokens, so detokenizer.text still contains
                            # it even after finalize(). Without this, the partial
                            # suffix text leaks into the output.
                            if suffix_hit and stop_suffixes and _remaining:
                                for s in stop_suffixes:
                                    if _remaining.endswith(s):
                                        _remaining = _remaining[: -len(s)]
                                        break
                            if _remaining:
                                _put(
                                    (
                                        _remaining,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        _cur_state,
                                    )
                                )
                        # Final stop chunk — consumer breaks on this
                        _put(
                            ("", n_tok, "stop", len(_thinking_tokens), None, _cur_state)
                        )
                        if prefix_cache is not None:
                            prefix_cache.add(ids, cache)
                        _save_breakpoint_prefixes(ids, cache)
                        mx.synchronize()
                        _unregister_inflight()
                        return
                if prefix_cache is not None:
                    prefix_cache.add(ids, cache)
                _save_breakpoint_prefixes(ids, cache)
                detokenizer.finalize()
                remaining = detokenizer.last_segment
                if _hb_active:
                    # Generation ended without a stop match — flush held-back
                    # text (it was real output that looked like a stop-prefix).
                    remaining = _hb.feed(remaining) + _hb.flush()
                _final_state = "reasoning" if _in_thinking else "normal"
                if remaining:
                    _put(
                        (
                            remaining,
                            n_tok,
                            None,
                            len(_thinking_tokens),
                            None,
                            _final_state,
                        )
                    )
                _put(("", n_tok, "length", len(_thinking_tokens), None, _final_state))
                mx.synchronize()
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
                    _engine.logger.debug(
                        "detokenizer finalize in error handler failed", exc_info=True
                    )

        def _run():
            try:
                _run_inner()
                # Normal exit: clear Metal cache to prevent GPU memory
                # accumulation across sequential streaming requests.
                try:
                    import mlx.core as _cleanup_mx

                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                except Exception:
                    pass
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
                    _engine.logger.debug(
                        "GPU cache cleanup failed in streaming OOM handler",
                        exc_info=True,
                    )
                if isinstance(e, MemoryError) or "memory" in str(e).lower():
                    _engine.logger.warning(f"OOM during streaming: {e}")
                _put(e)
            except Exception as e:
                _unregister_inflight()
                _finalize_detokenizer()
                try:
                    import mlx.core as _cleanup_mx

                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                except Exception:
                    _engine.logger.debug(
                        "GPU cache cleanup failed in streaming error handler",
                        exc_info=True,
                    )
                _put(e)
            finally:
                _put(_sentinel)

        def _run_with_lora():
            # (LoRA concurrency keystone): adapter lifecycle on the executor thread,
            # serialized with generate_step (see the non-streaming twin).
            _lora_applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
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
            try:
                return _run()
            finally:
                if _lora_applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        _engine.logger.debug(
                            "LoRA release failed (stream executor)", exc_info=True
                        )

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run_with_lora)

        accumulated = ""
        n_tok = 0
        _reasoning_tokens = 0
        # _cached_tokens_box now defined above, before run_in_executor .
        _fp_lock = getattr(self, "_fast_path_lock", None)
        if _fp_lock is not None:
            with _fp_lock:
                self._active_fast_path_count += 1
        try:
            while True:
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=timeout_seconds)
                except TimeoutError:
                    _engine.logger.warning(
                        f"Streaming fast path timeout: no token for {timeout_seconds}s"
                    )
                    _timeout_cancel.set()  # Signal GPU loop to stop
                    # Yield terminal output so consumer sees finished=True
                    yield _engine.GenerationOutput(
                        text=_engine._clean_special_tokens(accumulated)
                        if accumulated
                        else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="timeout",
                        error=f"Streaming timeout: no token for {timeout_seconds}s",
                        ttft_ms=round(_stream_ttft_box[0] * 1000, 1)
                        if _stream_ttft_box[0] > 0
                        else 0.0,
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
                        yield _engine.GenerationOutput(
                            text=_engine._clean_special_tokens(accumulated)
                            if accumulated
                            else "",
                            new_text="",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=n_tok,
                            finished=True,
                            finish_reason="memory_limit",
                            error=str(item),
                            ttft_ms=round(_stream_ttft_box[0] * 1000, 1)
                            if _stream_ttft_box[0] > 0
                            else 0.0,
                            cached_tokens=_cached_tokens_box[0],
                            reasoning_tokens=_reasoning_tokens,
                        )
                    else:
                        # Non-OOM exception: yield terminal error output
                        yield _engine.GenerationOutput(
                            text=_engine._clean_special_tokens(accumulated)
                            if accumulated
                            else "",
                            new_text="",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=n_tok,
                            finished=True,
                            finish_reason="error",
                            error=str(item),
                            ttft_ms=round(_stream_ttft_box[0] * 1000, 1)
                            if _stream_ttft_box[0] > 0
                            else 0.0,
                            cached_tokens=_cached_tokens_box[0],
                            reasoning_tokens=_reasoning_tokens,
                        )
                    break
                if len(item) >= 6:
                    (
                        new_text,
                        tok_count,
                        _fr_val,
                        _reasoning_tokens,
                        _lp_entry,
                        _cur_state,
                    ) = item[:6]
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
                if len(accumulated) > _engine._MAX_STREAMING_TEXT_BUFFER:
                    _engine.logger.error(
                        "Streaming text buffer exceeded 1MB limit (%d bytes) — truncating",
                        len(accumulated),
                    )
                    yield _engine.GenerationOutput(
                        text=_engine._clean_special_tokens(accumulated),
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=tok_count,
                        finished=True,
                        finish_reason="length",
                        error="Streaming text buffer exceeded 1MB limit",
                        cached_tokens=_cached_tokens_box[0],
                        reasoning_tokens=_reasoning_tokens,
                        ttft_ms=round(_stream_ttft_box[0] * 1000, 1)
                        if _stream_ttft_box[0] > 0
                        else 0.0,
                    )
                    break
                n_tok = tok_count

                # Streaming backpressure: slow down if client can't keep up
                if _backpressure.check_backpressure(_q.qsize()):
                    _delay = _backpressure.get_delay_ms(_q.qsize())
                    if _delay > 0:
                        await asyncio.sleep(_delay / 1000)

                # Record TTFT in Prometheus on first token
                if _stream_ttft_recorded[0] and _stream_ttft_box[0] > 0:
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _stream_ttft_box[0],
                            labels={"model_id": self.model_label},
                        )
                    except Exception:
                        _engine.logger.debug(
                            "streaming TTFT prometheus recording failed", exc_info=True
                        )
                    _stream_ttft_recorded[0] = False  # only observe once
                # Attach logprobs to output if computed for this token
                _lp_list = None
                if _lp_entry is not None:
                    # Decode token string for the logprob entry
                    try:
                        _lp_entry["token"] = tokenizer.decode([_lp_entry["token_id"]])
                        from .text_utils import token_id_to_bytes  # raw bytes

                        _lp_entry["bytes"] = token_id_to_bytes(
                            tokenizer, _lp_entry["token_id"], _lp_entry["token"]
                        )
                    except Exception:
                        _engine.logger.debug(
                            "logprob token decode failed in streaming", exc_info=True
                        )
                        _lp_entry["token"] = ""
                        _lp_entry["bytes"] = []
                    _lp_list = [_lp_entry]
                yield _engine.GenerationOutput(
                    text=_engine._clean_special_tokens(accumulated),
                    new_text=_engine._clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                    stopped_by_stop_sequence=bool(
                        done and finish_reason == "stop" and _user_stop_hit[0]
                    ),
                    reasoning_tokens=_reasoning_tokens,
                    logprobs=_lp_list,
                    cached_tokens=_cached_tokens_box[0],
                    ttft_ms=round(_stream_ttft_box[0] * 1000, 1)
                    if _stream_ttft_box[0] > 0
                    else 0.0,
                    current_state=_cur_state,
                )
                if done:
                    break
        finally:
            _fp_stats.finish("stop")
            # LoRA release+restore happens inside _run_with_lora on the executor .
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
                            from yunshu_gateway.middleware.prometheus_exporter import (
                                get_prometheus_metrics,
                            )

                            pm = get_prometheus_metrics()
                            avg_itl = sum(_stream_itl_samples) / len(
                                _stream_itl_samples
                            )
                            pm.observe_histogram(
                                "itl_seconds",
                                avg_itl,
                                labels={"model_id": self.model_label},
                            )
                        except Exception:
                            _engine.logger.debug(
                                "streaming ITL prometheus recording failed",
                                exc_info=True,
                            )
                except Exception:
                    _engine.logger.debug(
                        "ServerMetrics recording failed in streaming fast path",
                        exc_info=True,
                    )
            if not future.done():
                future.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await future
            # Drain remaining queue items to unblock the executor thread's
            # call_soon_threadsafe calls, preventing GPU work from continuing
            # after the consumer has stopped iterating.
            while not _q.empty():
                try:
                    _q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            # Decrement active fast path count
            _fp_lock = getattr(self, "_fast_path_lock", None)
            if _fp_lock is not None:
                with _fp_lock:
                    self._active_fast_path_count -= 1


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
