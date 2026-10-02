from __future__ import annotations

"""Engine speculative extracted from batched_engine.

Patchable helpers resolve through the compatibility facade. Concrete self types
retain the shared BatchedEngine state; misc ignores allow that mixin self type.
"""

import asyncio
import time


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


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
