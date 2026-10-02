from __future__ import annotations

"""Engine ngram extracted from batched_engine.

Patchable helpers resolve through the compatibility facade. Concrete self types
retain the shared BatchedEngine state; misc ignores allow that mixin self type.
"""

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from contextlib import suppress

from .stream_bridge import StreamBridge, make_stream_queue


class EngineNgramMixin:
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

    async def _generate_ngram_spec(  # type: ignore[misc]
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
    ) -> _engine.GenerationOutput:
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
            _engine.logger.info(
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
        stop_ids.update(_engine._read_config_eos_ids(self.model_name))
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
                    _engine.logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        # route temp>0 off mlx-lm's PRNG-trapped make_sampler (its
        # categorical_sampling @mx.compile cache traps the global PRNG state, so
        # sequential/concurrent temp>0 spec requests collapse + seed is a no-op).
        if temperature is not None and temperature > 1e-6:
            sampler = _engine._build_temp_sampler(
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
                sampler = _engine._build_constrained_sampler(
                    sampler, json_schema, tokenizer
                )
                _grammar_constraint = (
                    sampler.constraint if hasattr(sampler, "constraint") else None
                )
            except Exception:
                _engine.logger.warning(
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
                        _engine.logger.debug(
                            "grammar rollback failed in n-gram filter", exc_info=True
                        )
                except Exception:
                    _engine.logger.debug(
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
                        _engine.logger.debug(
                            "grammar constraint advance failed", exc_info=True
                        )
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
                _engine.logger.debug(
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
                _engine.logger.debug(
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

                with _engine._wired_limit_ctx(model):
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
                        if _engine._is_cancelled(cancel_event):
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
                                _engine.logger.warning(
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
                                    _engine.logger.debug(
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
                                    _engine.logger.debug(
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
                    _engine._maybe_quantize_kv_cache(
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
                    _engine.logger.debug(
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
                    output_text = _engine._clean_special_tokens(output_text)
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
                        _engine.logger.debug(
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
            _engine.logger.warning(
                "OOM during N-gram spec generation — returning memory_limit finish reason"
            )
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                _engine.logger.debug(
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
                    _engine.logger.warning(
                        "LoRA release failed after n-gram spec OOM", exc_info=True
                    )
            return _engine.GenerationOutput(
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
                _engine.logger.warning(f"MLX OOM during N-gram spec generation: {e}")
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    _engine.logger.debug(
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
                        _engine.logger.warning(
                            "LoRA release failed after n-gram spec OOM (RuntimeError)",
                            exc_info=True,
                        )
                return _engine.GenerationOutput(
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
                    _engine.logger.warning(
                        "LoRA release failed after n-gram spec RuntimeError",
                        exc_info=True,
                    )
            # Return error output for non-OOM RuntimeError instead of
            # propagating to caller (which expects GenerationOutput).
            return _engine.GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"RuntimeError during N-gram spec generation: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except Exception as e:
            _engine.logger.error(
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
                    _engine.logger.warning(
                        "LoRA release failed after n-gram spec unexpected error",
                        exc_info=True,
                    )
            # Return error output instead of propagating exception to caller.
            return _engine.GenerationOutput(
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
        _cancelled = _engine._is_cancelled(cancel_event)
        if _cancelled or _stopped_by_suffix or _stopped_by_stop_id:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        output_text = _engine._clean_special_tokens(output_text)

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
                _engine.logger.debug(
                    "TTFT prometheus recording failed in n-gram spec path",
                    exc_info=True,
                )

        # Channel-style reasoning recovery (Gemma-4 <|channel>…<channel|>): the
        # greedy default routes here (n-gram spec), so the same recovery the fast
        # path does must run here too, else gemma reasoning leaks into content.
        output_text, _ch_reason = _engine._recover_channel_reasoning(
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
                _engine.logger.debug(
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
                _engine.logger.warning(
                    "LoRA release failed after n-gram spec normal completion",
                    exc_info=True,
                )
        return _engine.GenerationOutput(
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

    async def _stream_generate_ngram_spec(  # type: ignore[misc]
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
    ) -> AsyncIterator[_engine.GenerationOutput]:
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
            _engine.logger.info(
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
        stop_ids.update(_engine._read_config_eos_ids(self.model_name))
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
                    _engine.logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        # route temp>0 off mlx-lm's PRNG-trapped make_sampler (see the
        # non-streaming sibling). Greedy (temp==0) stays on argmax make_sampler.
        if temperature is not None and temperature > 1e-6:
            sampler = _engine._build_temp_sampler(
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
                sampler = _engine._build_constrained_sampler(
                    sampler, json_schema, tokenizer
                )
                _stream_grammar_constraint = (
                    sampler.constraint if hasattr(sampler, "constraint") else None
                )
            except Exception:
                _engine.logger.warning(
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
                        _engine.logger.debug(
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
                _engine.logger.debug(
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
                _engine.logger.error(
                    f"N-gram streaming generation failed: {e}", exc_info=True
                )
                try:
                    import mlx.core as _cleanup_mx

                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                except Exception:
                    _engine.logger.debug(
                        "GPU cache cleanup failed in n-gram streaming error handler",
                        exc_info=True,
                    )
                _put(e)
            finally:
                if _applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        _engine.logger.debug(
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
                _engine.logger.debug(
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
                _ng_think_start_token, _ng_think_end_token = (
                    _engine._resolve_think_token_ids(tokenizer)
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

            with _engine._wired_limit_ctx(model):
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
                    if _engine._is_cancelled(cancel_event):
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
                            _ng_bonus_logits_cpu = _engine._apply_spec_bonus_penalties(
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
                                _engine.logger.debug(
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
                            _engine.logger.debug(
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
            _ng_consumer_think_start, _ng_consumer_think_end = (
                _engine._resolve_think_token_ids(tokenizer)
            )
        _ng_fp_lock = getattr(self, "_fast_path_lock", None)
        if _ng_fp_lock is not None:
            with _ng_fp_lock:
                self._active_fast_path_count += 1
        try:
            while True:
                # Check cancel_event from consumer side (mirrors MTP streaming path)
                if _engine._is_cancelled(cancel_event):
                    yield _engine.GenerationOutput(
                        text=_engine._clean_special_tokens(accumulated)
                        if accumulated
                        else "",
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
                    _engine.logger.warning(
                        f"N-gram streaming timeout: no token for {timeout_seconds}s"
                    )
                    _ng_timeout_cancel.set()  # Signal GPU loop to stop
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
                        error=f"N-gram streaming timeout: no token for {timeout_seconds}s",
                        ttft_ms=_ng_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                if item is _sentinel:
                    break
                if isinstance(item, BaseException):
                    _engine.logger.warning(f"N-gram streaming error: {item}")
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
                        _engine.logger.debug(
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
                        _engine.logger.debug(
                            "N-gram streaming path: real logprobs unavailable "
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
                    _engine.logger.debug(
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


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
