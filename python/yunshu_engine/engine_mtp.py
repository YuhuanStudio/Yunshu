from __future__ import annotations

"""Engine mtp extracted from batched_engine.

Patchable helpers resolve through the compatibility facade. Concrete self types
retain the shared BatchedEngine state; misc ignores allow that mixin self type.
"""

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from contextlib import suppress

from .stream_bridge import StreamBridge, make_stream_queue
from .text_utils import StopHoldbackBuffer


class EngineMtpMixin:
    async def _generate_mtp(  # type: ignore[misc]
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
        cancel_event: asyncio.Event | None = None,
        json_schema: dict | str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logits_processors: list | None = None,
        timeout_seconds: float = 300.0,
        lora_adapter: str | None = None,
    ) -> _engine.GenerationOutput:
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
                _constrained_sampler = _engine._build_constrained_sampler(
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
                _engine.logger.warning(
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
        eos_ids.update(_engine._read_config_eos_ids(self.model_name))
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
                    _engine.logger.debug(
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
            _mtp_sampler = _engine._build_temp_sampler(
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
                    _engine.logger.warning(
                        "LoRA release failed in MTP path", exc_info=True
                    )

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
                        _engine.logger.debug(
                            "LoRA release failed (MTP executor)", exc_info=True
                        )

        _mtp_gen_t0 = time.perf_counter()
        try:
            token_ids = await asyncio.wait_for(
                loop.run_in_executor(executor, _run_with_lora),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            _engine.logger.warning(f"MTP generation timed out after {timeout_seconds}s")
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                _engine.logger.debug(
                    "GPU cache cleanup failed after MTP timeout", exc_info=True
                )
            _lora_release()
            return _engine.GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"MTP generation timed out after {timeout_seconds}s",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except MemoryError:
            _engine.logger.warning(
                "OOM during MTP generation — returning memory_limit finish reason"
            )
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                _engine.logger.debug(
                    "GPU cache cleanup failed after MTP OOM", exc_info=True
                )
            _lora_release()
            return _engine.GenerationOutput(
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
                _engine.logger.warning(f"MLX OOM during MTP generation: {e}")
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    _engine.logger.debug(
                        "GPU cache cleanup failed after MTP OOM (RuntimeError)",
                        exc_info=True,
                    )
                _lora_release()
                return _engine.GenerationOutput(
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
            return _engine.GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"RuntimeError during MTP generation: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except Exception as e:
            _engine.logger.error(
                f"Unexpected error during MTP generation: {e}", exc_info=True
            )
            _lora_release()
            # Return error output instead of propagating exception to caller.
            return _engine.GenerationOutput(
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
            _mtp_think_start_token, _mtp_think_end_token = (
                _engine._resolve_think_token_ids(tokenizer)
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
        output_text = _engine._clean_special_tokens(detokenizer.text)

        # Determine finish_reason with cancel awareness
        _cancelled = _engine._is_cancelled(cancel_event)
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
            _engine.logger.debug("MTP metrics export failed", exc_info=True)

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
                _engine.logger.debug(
                    "reasoning_parser failed in MTP path", exc_info=True
                )

        _lora_release()
        return _engine.GenerationOutput(
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

    async def _stream_generate_mtp(  # type: ignore[misc]
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
        cancel_event: asyncio.Event | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        timeout_seconds: float = 300.0,
        json_schema: dict | str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logits_processors: list | None = None,
        lora_adapter: str | None = None,
    ) -> AsyncIterator[_engine.GenerationOutput]:
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
                _constrained_sampler = _engine._build_constrained_sampler(
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
                _engine.logger.warning(
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
        eos_ids.update(_engine._read_config_eos_ids(self.model_name))
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
                    _engine.logger.debug(
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
            _mtp_sampler = _engine._build_temp_sampler(
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
            _engine.logger.debug("MTP inflight prefix register failed", exc_info=True)

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_inflight_req_id)
            except Exception:
                _engine.logger.debug(
                    "MTP inflight prefix unregister failed", exc_info=True
                )

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
                        _engine._resolve_think_token_ids(tokenizer)
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
                _first_emit = _hb.feed(
                    _engine._clean_special_tokens(detokenizer.last_segment)
                )
                if _first_emit:
                    _put((_first_emit, 1, None, first))

                from .n_confirmed_patch import clear_rollback, restore_rollback

                # Emit an accepted token's cleaned text through the stop hold-back
                # buffer. Returns True if a multi-token string stop completed
                # (caller must set _early_stop and break). On an EOS token, call
                # _mtp_flush_held() instead to release genuine held text.
                def _mtp_emit(tok):
                    _seg = _engine._clean_special_tokens(detokenizer.last_segment)
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
                    if _engine._is_cancelled(cancel_event):
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
                            _engine.logger.debug(
                                "MTP grammar checkpoint failed", exc_info=True
                            )
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
                        _mtp_bonus_logits = _engine._apply_spec_bonus_penalties(
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
                                _engine.logger.debug(
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
                                _engine.logger.debug(
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
                                _engine.logger.debug(
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
                                _engine.logger.debug(
                                    "MTP grammar rollback failed", exc_info=True
                                )
                        # MTP-PEN: Apply penalty/bias to rejection correction logits (v0).
                        # The correction token is the first new token after the rejection.
                        if _has_mtp_pen:
                            _mtp_corr_logits = verify_out[0, 0, :]
                            _mtp_token_hist_corr = list(input_ids) + generated
                            _mtp_corr_logits = _engine._apply_spec_bonus_penalties(
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
                                _engine.logger.debug(
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
                _engine.logger.error(
                    f"MTP streaming generation failed: {e}", exc_info=True
                )
                # Finalize detokenizer to flush partial UTF-8 bytes before
                # reporting the error — without this, any bytes buffered in
                # the detokenizer's internal state are silently lost.
                try:
                    detokenizer.finalize()
                    _final_segment = detokenizer.last_segment
                    if _final_segment:
                        _put((_final_segment, len(generated), None, None))
                except Exception:
                    _engine.logger.debug(
                        "detokenizer finalize in MTP error handler failed",
                        exc_info=True,
                    )
                try:
                    mx.synchronize()
                    mx.clear_cache()
                except Exception:
                    _engine.logger.debug(
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
                _engine._resolve_think_token_ids(tokenizer)
            )
        _mtp_fp_lock = getattr(self, "_fast_path_lock", None)
        if _mtp_fp_lock is not None:
            with _mtp_fp_lock:
                self._active_fast_path_count += 1
        try:
            while True:
                # Check cancel_event from consumer side
                if _engine._is_cancelled(cancel_event):
                    # Yield terminal stop chunk so consumer sees finished=True
                    yield _engine.GenerationOutput(
                        text=_engine._clean_special_tokens(accumulated)
                        if accumulated
                        else "",
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
                    _engine.logger.warning(
                        f"MTP streaming timeout: no token for {timeout_seconds}s"
                    )
                    _mtp_timeout_cancel.set()  # Signal GPU loop to stop
                    # Yield terminal output so consumer sees finished=True
                    yield _engine.GenerationOutput(
                        text=_engine._clean_special_tokens(accumulated)
                        if accumulated
                        else "",
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
                    _engine.logger.warning(f"MTP streaming error: {item}")
                    # Yield terminal error output so consumer sees finished=True
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
                        _engine.logger.debug(
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
                        _engine.logger.debug(
                            "MTP streaming path: real logprobs unavailable "
                            "(per-token logits not exposed via queue). "
                            "Returning empty logprobs list."
                        )

                yield _engine.GenerationOutput(
                    text=_engine._clean_special_tokens(accumulated),
                    new_text=_engine._clean_special_tokens(new_text),
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
                    _engine.logger.warning(
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


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
