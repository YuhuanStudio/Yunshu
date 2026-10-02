from __future__ import annotations

"""Engine speculative extracted from batched_engine.

Patchable helpers resolve through the compatibility facade. Concrete self types
retain the shared BatchedEngine state; misc ignores allow that mixin self type.
"""

import asyncio
import time
from collections.abc import AsyncIterator


class EngineSpeculativeMixin:
    async def _warm_prompt_prefill(self: _engine.BatchedEngine) -> None:  # type: ignore[misc]
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
                _engine.logger.info(
                    f"Warm prompt prefill: {result.prompts_prefilled} prefilled, "
                    f"{result.prompts_skipped_cached} cached, "
                    f"{result.total_tokens_prefilled} tokens, "
                    f"{result.prefill_time_s:.3f}s"
                )
        except Exception as e:
            _engine.logger.warning(f"Warm prompt prefill failed: {e}")

    def gemma4_spec_generate(  # type: ignore[misc]
        self: _engine.BatchedEngine,
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

    def _gemma4_spec_eligible(  # type: ignore[misc]
        self: _engine.BatchedEngine,
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

    async def _generate_gemma4_assistant_spec(  # type: ignore[misc]
        self: _engine.BatchedEngine,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        stop_token_ids: list[int] | None = None,
        k: int = 4,
    ) -> _engine.GenerationOutput:
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
        out_text = _engine._clean_special_tokens(out_text)
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
            _engine.logger.debug(
                "ServerMetrics record failed (gemma4 assistant spec)", exc_info=True
            )
        return _engine.GenerationOutput(
            text=out_text,
            new_text=out_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(visible_ids),
            finished=True,
            finish_reason="stop" if finished_by_eos else "length",
            ttft_ms=ttft_ms,
            cached_tokens=0,
        )

    async def _generate_speculative(  # type: ignore[misc]
        self: _engine.BatchedEngine,
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
    ) -> _engine.GenerationOutput:
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
                    _engine.logger.debug(
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
                _spec_constraint = _engine._build_grammar_constraint(
                    json_schema, self._tokenizer
                )
            except Exception:
                _engine.logger.warning(
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
                        _engine.logger.debug(
                            "LoRA release failed (spec executor)", exc_info=True
                        )

        _spec_gen_t0 = time.perf_counter()
        try:
            token_ids, hit_stop = await asyncio.wait_for(
                loop.run_in_executor(executor, _run_spec_with_lora),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            _engine.logger.warning(
                f"Speculative generation timed out after {timeout_seconds}s"
            )
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                _engine.logger.debug(
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
                    _engine.logger.warning(
                        "LoRA release failed after spec decode timeout", exc_info=True
                    )
            return _engine.GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error=f"Speculative generation timed out after {timeout_seconds}s",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except MemoryError:
            _engine.logger.warning(
                "OOM during speculative generation — returning memory_limit finish reason"
            )
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                _engine.logger.debug(
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
                    _engine.logger.warning(
                        "LoRA release failed after spec decode OOM", exc_info=True
                    )
            return _engine.GenerationOutput(
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
                _engine.logger.warning(f"MLX OOM during speculative generation: {e}")
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    _engine.logger.debug(
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
                        _engine.logger.warning(
                            "LoRA release failed after spec decode OOM (RuntimeError)",
                            exc_info=True,
                        )
                return _engine.GenerationOutput(
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
                    _engine.logger.warning(
                        "LoRA release failed after spec decode RuntimeError",
                        exc_info=True,
                    )
            # Return error output for non-OOM RuntimeError instead of
            # propagating to caller (which expects GenerationOutput).
            return _engine.GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error=f"RuntimeError during speculative generation: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except Exception as e:
            _engine.logger.error(
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
                    _engine.logger.warning(
                        "LoRA release failed after spec decode unexpected error",
                        exc_info=True,
                    )
            # Return error output instead of propagating exception to caller.
            return _engine.GenerationOutput(
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

        text = _engine._clean_special_tokens(detokenizer.text)

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
                _engine.logger.debug(
                    "TTFT prometheus recording failed in spec decode path",
                    exc_info=True,
                )

        # Determine finish_reason with cancel awareness
        _cancelled = _engine._is_cancelled(cancel_event)
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
                _engine.logger.debug(
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
                _engine.logger.warning(
                    "LoRA release failed after spec decode normal completion",
                    exc_info=True,
                )
        return _engine.GenerationOutput(
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

    async def _stream_generate_speculative(  # type: ignore[misc]
        self: _engine.BatchedEngine,
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
    ) -> AsyncIterator[_engine.GenerationOutput]:
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
                    _engine.logger.debug(
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
                _spec_constraint = _engine._build_grammar_constraint(
                    json_schema, self._tokenizer
                )
            except Exception:
                _engine.logger.warning(
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
            _spec_think_start_token, _spec_think_end_token = (
                _engine._resolve_think_token_ids(self._tokenizer)
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
                _engine.logger.debug(
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
            _engine.logger.error(
                f"Spec decode streaming prefill failed: {e}", exc_info=True
            )
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_state["applied"]
                and getattr(self, "_lora_manager", None) is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    _engine.logger.warning(
                        "LoRA release failed after spec decode prefill error",
                        exc_info=True,
                    )
            raise
        _spec_gen_t0 = time.perf_counter()  # TTFT timing starts after prefill
        try:
            _spec_ttft_ms_val = 0.0
            _spec_ttft_recorded = False
            while len(generated_tokens) < max_tokens:
                if _engine._is_cancelled(cancel_event):
                    _engine.logger.debug(
                        "Cancel event triggered during spec decode streaming"
                    )
                    # Yield terminal stop chunk so consumer sees finished=True
                    if generated_tokens:
                        detokenizer.finalize()
                        _final_text = _engine._clean_special_tokens(detokenizer.text)
                    else:
                        _final_text = ""
                    yield _engine.GenerationOutput(
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
                                _engine.logger.debug(
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
                            _engine.logger.debug(
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
                        _bonus_logits = _engine._apply_spec_bonus_penalties(
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
                        _engine.logger.debug(
                            "spec streaming TTFT prometheus recording failed",
                            exc_info=True,
                        )

                # Yield accepted text via incremental detokenizer
                chunk_text = _engine._clean_special_tokens(detokenizer.last_segment)
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
                        tok_text = _engine._clean_special_tokens(
                            self._tokenizer.decode([tid])
                        )
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

                yield _engine.GenerationOutput(
                    text=_engine._clean_special_tokens(detokenizer.text),
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
                                _engine.logger.debug(
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
            _engine.logger.debug("Client disconnected during spec decode streaming")
        except Exception as e:
            _engine.logger.error(f"Spec decode streaming error: {e}", exc_info=True)
            # Yield a terminal error output so the consumer sees finished=True
            # instead of a broken stream (exception without terminal output).
            try:
                detokenizer.finalize()
                _err_final_text = _engine._clean_special_tokens(detokenizer.text)
            except Exception:
                _err_final_text = ""
            yield _engine.GenerationOutput(
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
                    _engine.logger.warning(
                        "LoRA release failed in spec decode streaming finally",
                        exc_info=True,
                    )


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
