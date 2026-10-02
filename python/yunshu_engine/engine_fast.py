from __future__ import annotations

"""Engine fast extracted from batched_engine.

Patchable helpers resolve through the compatibility facade. Concrete self types
retain the shared BatchedEngine state; misc ignores allow that mixin self type.
"""

import asyncio
import time

from . import settings
from .context_window import ContextBudgetError, reject_overlong_prompt
from .fast_path_stats import FastPathStats


class EngineFastMixin:
    def _jump_forward_generate_sync(
        self, input_ids, constraint, max_tokens, eos_ids, stop=None
    ):
        """Jump-forward decoding (opt-in YUNSHU_JUMP_FORWARD=1) for
        JSON-schema/grammar-constrained generation. Runs on the MLX executor.

        Custom constrained decode loop (mlx-lm's generate_step is a black box that
        can't skip forwards). Each iteration runs ONE model forward over
        [sampled_token] + [the structural tokens the FSM FORCES next] — so the
        forced JSON structure (':', ',', '}', forced key names, 'true'/'false'/
        'null') is emitted WITHOUT a per-token forward. Forward count = number of
        BRANCH points, not total tokens → a big decode-latency cut for structured
        output on bandwidth-bound Apple Silicon. Greedy (argmax of the
        constraint-masked logits) for deterministic, schema-valid output.

        Returns (text, generated_token_ids, n_forwards, stop_hit). Constraint
        guarantees the TEXT is schema-valid; the forced run's exact token boundaries
        may differ from natural tokenization (no backward token-healing), which only
        affects the model's free-value continuation, never the forced structure.
        stop_hit is True iff generation terminated because a `stop` substring appeared
        (so the caller can report finish_reason="stop" and an accurate token count).
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        from .json_schema import apply_json_constraint

        model, tok = self._model, self._tokenizer
        eos = set(eos_ids or [])
        stops = [s for s in (stop or []) if s]

        def _first_stop(text: str) -> int:
            # earliest index at which any stop substring begins, or -1
            best = -1
            for s in stops:
                p = text.find(s)
                if p != -1 and (best == -1 or p < best):
                    best = p
            return best

        cache = make_prompt_cache(model)
        logits = model(mx.array([input_ids]), cache=cache)[:, -1, :]
        gen_ids: list[int] = []
        pieces: list[str] = []
        acc = ""  # running detokenized text, for incremental stop scanning
        stop_hit = False
        n_fwd = 1
        max_tokens = int(max_tokens)
        while len(gen_ids) < max_tokens:
            allowed = constraint.get_allowed_tokens(tok, gen_ids)
            if not allowed:
                # FSM is in a terminal/DONE state with no further legal tokens. If
                # eos is empty (e.g. ignore_eos) there is nothing left to sample —
                # stop instead of looping forever on an all-masked distribution.
                if not eos:
                    break
            masked = apply_json_constraint(logits, allowed if allowed else list(eos))
            tid = int(mx.argmax(masked, axis=-1).item())
            if tid in eos:
                break
            gen_ids.append(tid)
            ttext = tok.decode([tid]) if tok else ""
            pieces.append(ttext)
            acc += ttext
            # Detect the stop substring INCREMENTALLY (like _generate_fast),
            # so we break at the token that completes it instead of running to
            # max_tokens and only truncating `text` post-hoc — which left gen_ids
            # over-counted (wrong completion_tokens) and finish_reason="length".
            if stops and _first_stop(acc) != -1:
                stop_hit = True
                break
            try:
                constraint.advance(ttext)
            except Exception:
                _engine.logger.debug(
                    "jump-forward: constraint.advance(token) failed", exc_info=True
                )
                break
            batch = [tid]
            # Jump-forward: emit the grammar-forced continuation in this same forward.
            try:
                fc = constraint.forced_continuation()
            except Exception:
                fc = ""
            if fc and len(gen_ids) < max_tokens:
                fids = tok.encode(fc, add_special_tokens=False) if tok else []
                # Don't overshoot max_tokens: a long forced continuation could push
                # gen_ids past the budget in one jump. Truncate to the remaining room.
                room = max_tokens - len(gen_ids)
                truncated = len(fids) > room
                if truncated:
                    fids = fids[:room]
                if fids:
                    batch = [tid] + fids
                    gen_ids.extend(fids)
                    # On truncation the emitted text must reflect the tokens we
                    # actually fed, not the full forced string (which won't be
                    # finished). Decode the kept tokens.
                    femit = fc if not truncated else (tok.decode(fids) if tok else "")
                    pieces.append(femit)
                    acc += femit
                    try:
                        # Advance the FSM only by the text we
                        # actually emitted — advancing by the full `fc` on truncation
                        # desynced the constraint from gen_ids.
                        constraint.advance(femit)
                    except Exception:
                        _engine.logger.debug(
                            "jump-forward: advance(forced) failed", exc_info=True
                        )
                    if stops and _first_stop(acc) != -1:
                        stop_hit = True
                        break
                    if truncated:
                        # gen_ids is now at max_tokens; the loop is about to exit.
                        # Skip the otherwise-wasted final forward (its logits would
                        # be discarded).
                        break
            out = model(mx.array([batch]), cache=cache)
            logits = out[:, -1, :]
            mx.eval(logits)
            n_fwd += 1
        text = "".join(pieces)
        if stops:
            p = _first_stop(text)
            if p != -1:
                text = text[:p]
                stop_hit = True
        return text, gen_ids, n_fwd, stop_hit

    def _capture_hybrid_prefix(
        self, model, full_ids, cache, prefix_cache, start_offset, block
    ):
        """Advance `cache` through full_ids[start_offset:-1] in block-sized
        chunks, storing a trim=0 block-boundary snapshot into `prefix_cache` at
        each boundary. The last token is intentionally NOT consumed — it is left
        for generate_step to prefill and produce the first decode logits.

        This is how the fast path reuses prefixes for HYBRID
        models (Qwen3.5). A boundary snapshot holds the EXACT recurrent state
        after N tokens, so a later request sharing that N-token prefix reuses it
        with zero trim — never slicing the non-trimmable ArraysCache layers.
        Returns the absolute number of prompt tokens now resident in `cache`.
        """
        n = len(full_ids)
        if n <= 1:
            return start_offset
        p = int(start_offset)
        end = n - 1  # leave the final token for generate_step
        while p < end:
            chunk = full_ids[p : min(p + block, end)]
            cn = int(chunk.shape[0])
            if cn == 0:
                break
            _ = model(chunk[None], cache=cache)
            p += cn
            # NB: no per-chunk mx.eval. MLX arrays are immutable — each model()
            # call produces NEW key/value arrays — so a detached snapshot taken
            # now stays valid even though the prefill graph is still lazy. Forcing
            # eval here would serialize the prefill (a large cold-latency penalty);
            # leaving it lazy keeps prefill pipelined and materializes once at the
            # first decode token.
            if p % block == 0 and p >= prefix_cache._min_prefix:
                try:
                    prefix_cache.add(full_ids[:p], cache)
                except Exception:
                    _engine.logger.debug(
                        "hybrid boundary snapshot add failed", exc_info=True
                    )
        return p

    def _tool_call_processor(self: _engine.BatchedEngine, input_ids: list[int]):  # type: ignore[misc]
        """Structural-tag logits processor for the request's native tools (free until
        the tool-call marker, then the call body is masked to this request's tool
        grammar), or None: no native tools, ``YUNSHU_TOOL_GRAMMAR`` off, or a model /
        tool set the grammar cannot cover."""
        tools = _engine._REQUEST_TOOLS.get()
        if not tools or not settings.get_bool("YUNSHU_TOOL_GRAMMAR"):
            return None
        from . import tool_call_grammar as tcg

        use = _engine._REQUEST_TOOL_USE.get() or {}
        choice = tcg.normalize_tool_choice(use.get("tool_choice"))
        if choice == "none":
            return None
        parallel = use.get("parallel", True) is not False
        cache = self.__dict__.setdefault("_tool_grammars", {})
        key = tcg.grammar_key(tools, choice, parallel)
        if key not in cache:
            args = getattr(self._model, "args", None)
            vocab = getattr(args, "vocab_size", None) or len(
                getattr(self._tokenizer, "_tokenizer", self._tokenizer)
            )
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[key] = tcg.compile_tool_grammar(
                tools,
                self._tokenizer,
                int(vocab),
                tool_choice=choice,
                parallel=parallel,
            )
        grammar = cache[key]
        if grammar is None:
            return None
        guide = grammar.guide(
            thinking_open=tcg.prompt_opens_thinking(input_ids, grammar, self._tokenizer)
        )
        return tcg.TokenStreamToolCallProcessor(guide)

    async def _generate_fast(  # type: ignore[misc]
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
        timeout_seconds: float = 300.0,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        json_schema: dict | str | None = None,
        cancel_event: asyncio.Event | None = None,
        logits_processors: list | None = None,
        priority: int = 0,
        lora_adapter: str | None = None,
        kv_cache_breakpoints: list[int] | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        suppress_tokens: list[int] | None = None,
    ) -> _engine.GenerationOutput:
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

        # ── Model preprocessor for multimodal input ──
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
                    if hasattr(model, "config") and hasattr(model.config, "model_type"):
                        model_config["model_type"] = model.config.model_type
                    preprocessor = self._preprocessor_registry.detect(model_config)
                    if preprocessor is not None:
                        processed = preprocessor.preprocess(prompt, tokenizer=tokenizer)
                        if processed.token_ids:
                            prompt = processed.token_ids
                except Exception:
                    _engine.logger.debug(
                        "model preprocessor failed, using raw prompt", exc_info=True
                    )

        # ── Pre-encoding context window truncation (message-level) ──
        # When prompt is a list of message dicts, use ContextWindowManager to
        # truncate at the message level BEFORE applying the chat template.
        # This preserves system prompts — the engine-loop path does the same.
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
                            "Fast path pre-encode truncation: estimated %d → %d tokens",
                            _est_tokens,
                            result.truncated_token_count,
                        )
            except ContextBudgetError:
                raise
            except Exception:
                _engine.logger.debug(
                    "context window truncation skipped in fast path", exc_info=True
                )

        # Encode prompt
        if isinstance(prompt, str):
            text = prompt
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # Route through _apply_chat_template (NOT raw tokenizer.apply_chat_template)
            # so the fast path — the DEFAULT serving path — also gets: tool_calls.
            # arguments str→dict normalization (multi-turn tool history otherwise
            # double-encodes or raises in Jinja → silent degraded fallback), the
            # family message adapter (Gemma4/Mistral/Harmony role rules), dangling
            # <think> closing, and the enable_thinking-unsupported retry.
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
        else:
            text = str(prompt)

        input_ids = self._encode_prompt(tokenizer, text)
        prompt_tokens = len(input_ids)

        if not input_ids:
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is not None:
                input_ids = [bos_id]
            else:
                eos_id = getattr(tokenizer, "eos_token_id", 1)
                input_ids = [eos_id]
            prompt_tokens = len(input_ids)

        # ── Context window: clamp generation + truncate over-long prompt ──
        # Resolve the REAL context window. The old code read ONLY the
        # `max_seq_len` attribute, which mlx-lm models don't expose (Qwen/Llama use
        # `max_position_embeddings`) → this safety net was DEAD for most models
        # (same root cause as the gateway-guard bug). And it only handled
        # prompt-alone-too-long; a prompt that FITS but whose prompt+max_tokens
        # exceeds the window decoded past max_position_embeddings into
        # RoPE-extrapolated garbage. Resolve correctly, then clamp max_tokens.
        _max_ctx = _engine._resolve_model_max_ctx(model)
        if _max_ctx and _max_ctx > 0:
            # Prompt alone fills the window: reject (the gateway guard normally
            # does this first); never cut the token stream from the left.
            reject_overlong_prompt(prompt_tokens, _max_ctx)
            # Clamp max_tokens so prompt + generation stays within the window
            # (prevents RoPE-extrapolated garbage past max_position_embeddings).
            _room = _max_ctx - prompt_tokens
            if _room >= 1 and max_tokens > _room:
                _engine.logger.info(
                    "Fast path clamped max_tokens %d → %d to fit context "
                    "(prompt=%d, ctx=%d)",
                    max_tokens,
                    _room,
                    prompt_tokens,
                    _max_ctx,
                )
                max_tokens = _room

        stop_ids = set()
        _user_stop_ids: set[int] = set()  # ids of single-token USER stop strings
        _user_stop_hit = [False]  # a user stop (not EOS) ended generation
        # eos_token_id / eos_token_ids may be a single int OR an
        # iterable depending on the tokenizer (Qwen3.6-27B exposes eos_token_ids
        # as a bare int, which crashed `stop_ids.update(...)` with "'int' object
        # is not iterable"). Accept both shapes.
        # Collect EOS ids SEPARATELY so ignore_eos can suppress only the
        # model EOS (not user stop_token_ids), and min_tokens can mask them.
        _eos_ids: set[int] = set()
        _eid = getattr(tokenizer, "eos_token_id", None)
        if _eid is not None:
            _eos_ids.update(_eid if isinstance(_eid, (list, tuple, set)) else (_eid,))
        _eids = getattr(tokenizer, "eos_token_ids", None)
        if _eids is not None:
            _eos_ids.update(
                _eids if isinstance(_eids, (list, tuple, set)) else (_eids,)
            )
        # Multi-eos from the model's (generation_)config — e.g. Gemma-4's turn-end
        # token 106, which the tokenizer omits, so the model would never stop.
        _eos_ids.update(_engine._read_config_eos_ids(self.model_name))
        if not ignore_eos:
            stop_ids.update(_eos_ids)

        # Pre-encode stop strings for suffix matching.
        # A user stop string must ALWAYS be string-matched.
        # The previous code routed single-token stops to stop_ids via
        # tokenizer.encode(s), but the encoded id of a BARE stop ("C", "3")
        # rarely equals the SPACE-PREFIXED token the model actually emits (" C"),
        # so token-id matching silently missed most single-token stops (the stop
        # text leaked with finish_reason="stop" coming from natural EOS). We keep
        # the token-id as a cheap early-stop hint but ALSO always add the string
        # to stop_suffixes so the string matcher + final truncation enforce it.
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

        # For temp>0, bypass mlx-lm's @mx.compile-wrapped categorical_sampling
        # which traps the FIRST call's PRNG state in its compile cache (so
        # re-seeding via mx.random.seed() between requests is a no-op). Use
        # a numpy.random.Generator backed sampler instead — same shape as
        # the vlm_engine fix. Greedy (temp=0) keeps mlx-lm's
        # compiled argmax (deterministic anyway, faster).
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

        # JSON Schema / grammar constraint: wrap sampler with ConstrainedSampler
        if json_schema is not None:
            try:
                sampler = _engine._build_constrained_sampler(
                    sampler, json_schema, tokenizer
                )
            except Exception:
                _engine.logger.warning(
                    "Grammar constraint setup failed, falling back to unconstrained",
                    exc_info=True,
                )

        # Jump-forward decoding (opt-in YUNSHU_JUMP_FORWARD=1) for
        # JSON-schema-constrained GREEDY generation — emits the grammar-forced
        # structure with one forward per BRANCH instead of per token. Bypasses the
        # normal generate_step loop; gated + greedy + JSON-schema only (not
        # regex/choice/cfg, not logprobs) so the default path is untouched.
        # NOTE: the custom loop honors only greedy JSON-schema decoding — it does
        # NOT apply lora_adapter, cancel_event, logit_bias, suppress_tokens, or
        # min_tokens. Gate OFF whenever any of those are set so we never silently
        # ignore a request constraint; those requests fall through to the normal
        # constrained decode below.
        if (
            settings.get_bool("YUNSHU_JUMP_FORWARD")
            and json_schema is not None
            and not logprobs
            and (temperature is None or temperature <= 1e-6)
            and lora_adapter is None
            and cancel_event is None
            and not logit_bias
            and not suppress_tokens
            and not min_tokens
            and not (
                isinstance(json_schema, dict)
                and json_schema.get("type") in ("regex", "choice", "cfg")
            )
        ):
            try:
                import json as _json
                import time as _jf_time

                from .grammar_constraint import build_json_constraint
                from .mlx_executor import get_mlx_executor

                if json_schema == "json_object":
                    _jf_schema = None
                elif isinstance(json_schema, str):
                    _jf_schema = _json.loads(json_schema)
                else:
                    _jf_schema = json_schema
                _jf_constraint = build_json_constraint(
                    _jf_schema, self._tokenizer, compact=True
                )
                _jf_t0 = _jf_time.perf_counter()
                _jf_loop = asyncio.get_running_loop()
                # Count this against the fast-path concurrency gauge like the normal
                # path (it occupies the same single MLX executor thread).
                self._active_fast_path_count += 1
                try:
                    (
                        _jf_text,
                        _jf_ids,
                        _jf_nfwd,
                        _jf_stop,
                    ) = await _jf_loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: self._jump_forward_generate_sync(
                            input_ids, _jf_constraint, max_tokens, stop_ids, stop
                        ),
                    )
                finally:
                    self._active_fast_path_count -= 1
                _engine.logger.debug(
                    "jump-forward: %d tokens in %d forwards (%.1fx)",
                    len(_jf_ids),
                    _jf_nfwd,
                    len(_jf_ids) / max(_jf_nfwd, 1),
                )
                # A stop-substring termination is finish_reason="stop"
                # (and _jf_ids is now truncated at the stop, so completion_tokens is
                # accurate) — only report "length" when we genuinely hit the budget
                # without a stop.
                _jf_fr = (
                    "length"
                    if (not _jf_stop and len(_jf_ids) >= max_tokens)
                    else "stop"
                )
                return _engine.GenerationOutput(
                    text=_jf_text,
                    new_text=_jf_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(_jf_ids),
                    finished=True,
                    finish_reason=_jf_fr,
                    ttft_ms=(_jf_time.perf_counter() - _jf_t0) * 1000.0,
                )
            except Exception:
                _engine.logger.warning(
                    "jump-forward path failed; falling back to normal "
                    "constrained decode",
                    exc_info=True,
                )

        # SAMP-2: Save user-provided custom logits processors before building internal list
        # Length-gated KV-quant bits for THIS request (8-bit default at
        # long context; explicit config wins; None below the threshold).
        _req_kv_bits = self._effective_kv_quant_bits(prompt_tokens + max_tokens)
        _custom_logits_processors = logits_processors or []
        logits_processors = []
        _tool_processor = (
            self._tool_call_processor(input_ids) if json_schema is None else None
        )
        if repetition_penalty != 1.0:

            def _rep_penalty(tokens, logits, rp=repetition_penalty, ctx=20):
                if len(tokens) > 0:
                    recent = tokens[-ctx:]
                    import mlx.core as _mx

                    sel = logits[..., recent]
                    sel = _mx.where(sel < 0, sel * rp, sel / rp)
                    logits[..., _mx.array(recent)] = sel
                return logits

            logits_processors.append(_rep_penalty)
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

        # suppress_tokens — hard-ban specific token ids (logits → -inf).
        if suppress_tokens:
            _sup = [int(t) for t in suppress_tokens]

            def _suppress_proc(_tokens, logits, sup=_sup):
                vocab = logits.shape[-1]
                for tid in sup:
                    if 0 <= tid < vocab:
                        logits[..., tid] = -float("inf")
                return logits

            logits_processors.append(_suppress_proc)

        # min_tokens — mask EOS + single-token stops to -inf until at
        # least min_tokens have been GENERATED, so generation can't end early.
        # mlx-lm's tokens accumulator is [last_prompt_token, gen1, ...] so the
        # generated count is len(tokens) - 1 (the prompt-offset insight).
        # SKIP when a JSON/grammar constraint is active. The constraint
        # only permits EOS once the document is structurally COMPLETE (so it
        # already enforces a minimum), and masking EOS at the DONE state leaves the
        # constrained sampler with an all-(-inf) allowed set → its argmax fallback
        # emits an INVALID non-EOS token after a complete JSON value. Valid JSON
        # must win over min_tokens (which can't lengthen structured output anyway).
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
        # User processors take (token_ids: list[int], logits: mx.array) -> mx.array
        # but generate_step passes (tokens: mx.array, logits: mx.array).
        if _custom_logits_processors:
            logits_processors.extend(
                _engine._wrap_custom_logits_processor(p)
                for p in _custom_logits_processors
            )
        if _tool_processor is not None:
            logits_processors.append(_tool_processor)

        # Convert KV cache breakpoint char offsets to token positions.
        _kv_breakpoint_token_positions: list[int] = []
        if kv_cache_breakpoints and isinstance(text, str) and len(text) > 0:
            for char_off in kv_cache_breakpoints:
                if char_off <= 0 or char_off > len(text):
                    continue
                try:
                    prefix_ids = tokenizer.encode(text[:char_off])
                    _kv_breakpoint_token_positions.append(len(prefix_ids))
                except Exception:
                    _engine.logger.debug(
                        "KV breakpoint char->token conversion failed at offset %d",
                        char_off,
                        exc_info=True,
                    )

        # Generate inflight request ID outside the closure so it is
        # accessible in the outer exception handlers below. Previously
        # this was defined inside _run() which caused a NameError when
        # an exception fired before the executor ran the closure.
        _inflight_req_id = f"fp-{id(generate_step)}-{int(time.monotonic() * 1e6)}"

        # LoRA concurrency keystone: acquire+apply and release+restore are NO
        # LONGER done here on the event loop. They run inside `_run_with_lora` on the
        # max_workers=1 MLX executor (below), serialized with generate_step — so a
        # release-triggered _restore_base can never mutate the shared model while another
        # request's generation reads it (the cross-thread race the concurrency audit found).

        _fp_stats = FastPathStats(cancel_event, prompt_tokens)

        def _run():
            import mlx.core as mx

            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            tokens = []
            token_logprobs = []
            ttft_s = 0.0
            cached_tokens = 0
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            if self._lookahead_reasoning is not None:
                self._lookahead_reasoning._in_thinking = False
                self._lookahead_reasoning._thinking_tokens = []
                self._lookahead_reasoning._recent_accepts = []
            thinking_tokens_used = 0
            think_end_token = None
            think_start_token = None
            _in_thinking = False
            _thinking_tokens: list[int] = []
            _stopped_by_suffix = False
            _stopped_by_stop_id = False

            # Prefill progress tracking for the fast path
            _prefill_req_id = f"fp-{id(_run)}-{int(time.monotonic() * 1e6)}"
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

            if thinking_budget is not None or enable_thinking:
                # Resolve via the bracketed-form helper (the bare "</think"
                # encoded to 2 tokens for Qwen3/DeepSeek-R1 → guard failed → state machine
                # never engaged). Multi-token markers still resolve to (None, None). The
                # helper swallows its own errors, so no surrounding try/except is needed.
                think_start_token, think_end_token = _engine._resolve_think_token_ids(
                    tokenizer
                )

            # Seed reproducibility: when caller pins a seed for sampling
            # determinism, bypass KV prefix / prompt caches. Cached KV blocks
            # came from an earlier prefill whose intermediate accumulation
            # order is not bit-exact to a fresh prefill (bf16 / async eval),
            # so the first-decode logits drift by a few ULPs and a seeded
            # categorical sample lands on a different token. Disabling both
            # caches when seed is set restores cold==warm reproducibility at
            # the cost of TTFT on repeat prompts (acceptable for a seeded
            # request).
            _bypass_cache_for_seed = seed is not None and temperature > 0

            # Hybrid-model guard: prompt/prefix KV reuse both rely on trimming
            # the cached KV (snapshot trim at store, trim=1 + re-feed at lookup,
            # arbitrary-prefix trim for the prefix cache). Hybrid models mix
            # full-attention KVCache layers (trimmable) with linear/recurrent
            # ArraysCache layers (NOT trimmable — their recurrent state encodes
            # every token seen and cannot be sliced back). Reusing such a cache
            # silently corrupts the recurrent layers and produces grossly wrong
            # logits (verified: cold 'Thinking' -> warm garbage, logit Δ≈13).
            # mlx_lm.can_trim_prompt_cache reports False for these caches, so
            # disable ALL KV caching for them — correctness over TTFT.
            _cache_trimmable = self._cache_supports_trim(model)
            # Hybrid models are normally bypassed (not
            # trimmable), so route them through the
            # boundary-snapshot path instead: keep the cache enabled, force
            # trim=0-only reuse (no_trim mode), and chunk-prefill with snapshots.
            _hybrid_mode = (
                (not _cache_trimmable)
                and not _bypass_cache_for_seed
                and self._kv_prefix_cache is not None
            )
            if _hybrid_mode:
                self._kv_prefix_cache._no_trim_mode = True
            _bypass_cache = _bypass_cache_for_seed or (
                (not _cache_trimmable) and not _hybrid_mode
            )
            # The KV prefix cache AND the exact-match prompt cache are
            # keyed on token ids (+ model name) but NOT on the active LoRA adapter. The
            # cached KV is computed with whatever adapter was applied when it was stored,
            # so a later request with the SAME prompt prefix but a DIFFERENT adapter (or
            # base) would get a prefix/exact hit and decode on the WRONG adapter's KV —
            # silently serving a model the caller didn't ask for (the "LoRA-fail must
            # not silently serve base" class, reopened through the cache). Until the caches
            # are adapter-keyed, bypass both whenever an adapter is active. (Adapter-keyed
            # reuse is a future enhancement; correctness first.)
            if lora_adapter is not None:
                _bypass_cache = True

            # Prompt cache: try exact-match KV lookup by messages hash.
            # Skip for hybrid — the prompt cache reuses full KV
            # with a trim=1 refeed (and stores post-generation state), which
            # corrupts non-trimmable recurrent layers. Hybrid uses boundary
            # snapshots only.
            _pc_hit = False
            if (
                not _bypass_cache
                and not _hybrid_mode
                and hasattr(self, "_prompt_cache")
                and self._prompt_cache is not None
            ):
                try:
                    from .prompt_cache import compute_messages_hash

                    _pc_hash = compute_messages_hash(
                        [{"role": "user", "content": text}],
                        model=self.model_name,
                    )
                    _pc_entry = self._prompt_cache.lookup(_pc_hash)
                    if _pc_entry is not None and _pc_entry.kv_state is not None:
                        # CRITICAL: snapshot the stored cache before using
                        # it for decode. Decoding mutates KV layers in place
                        # (c.keys = mx.concat(c.keys, new_keys) replaces the
                        # attribute), so a shared reference would corrupt
                        # the stored entry — next lookup gets a cache with
                        # offset advanced past prompt length, ids_to_prefill
                        # becomes empty, and the model continues the prior
                        # response (BUG-A cross-request leakage).
                        # FULL-HIT RE-FEED: the prompt cache stores exactly
                        # prompt-length KV (token_count == len(ids)), so a hit
                        # leaves ids_to_prefill empty and the engine re-feeds
                        # the last prompt token to start decoding. If the cache
                        # still holds that token (trim=0), re-feeding DUPLICATES
                        # it (offset -> len+1, wrong RoPE positions) and the
                        # model echoes the prompt / emits garbage. Trim the last
                        # token from the snapshot so the re-fed ids[-1:] lands at
                        # the correct position. (BUG-A2 full-hit duplication.)
                        _pc_count = int(_pc_entry.token_count)
                        _refeed = 1 if (_pc_count >= len(ids) and len(ids) > 0) else 0
                        if self._kv_prefix_cache is not None:
                            try:
                                cache = self._kv_prefix_cache._snapshot_cache(
                                    _pc_entry.kv_state,
                                    trim=_refeed,
                                )
                            except Exception:
                                _engine.logger.debug(
                                    "prompt cache snapshot failed; falling back to direct ref",
                                    exc_info=True,
                                )
                                cache = _pc_entry.kv_state
                                _refeed = 0
                        else:
                            cache = _pc_entry.kv_state
                            _refeed = 0
                        _pc_hit = True
                        cached_tokens = _pc_count - _refeed
                        _engine.logger.debug(
                            f"Prompt cache hit: hash={_pc_hash[:12]}, "
                            f"tokens={cached_tokens} (refeed={_refeed})"
                        )
                except Exception:
                    _engine.logger.debug("prompt cache lookup failed", exc_info=True)

            # Try KV prefix cache hit (skip when prompt cache already hit —
            # prompt cache provides full KV state which is always better,
            # and skip when seed-bypass is active for reproducibility).
            prefix_cache = None if _bypass_cache else self._kv_prefix_cache
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
            if not _pc_hit:
                try:
                    cached_kv, _, matched = (
                        prefix_cache.get(ids)
                        if prefix_cache is not None
                        else (None, None, 0)
                    )
                except Exception:
                    _engine.logger.warning(
                        "KV prefix cache get failed — falling back to full prefill",
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
                if cached_kv is not None:
                    cached_tokens = matched
                    ids_to_prefill = ids[matched:]
                else:
                    ids_to_prefill = ids
                # Full prefix match: generate_step needs at least one token
                # to start decoding. Re-feed the last cached token.
                if len(ids_to_prefill) == 0 and len(ids) > 0:
                    ids_to_prefill = ids[-1:]
            else:
                # Prompt cache provided full KV — skip prefix cache lookup.
                cached_kv = None
                ids_to_prefill = ids[cached_tokens:]
                if len(ids_to_prefill) == 0 and len(ids) > 0:
                    ids_to_prefill = ids[-1:]

            # Inflight prefix sharing: check for in-flight
            # prefills with matching prefix to share partial KV blocks.
            # Skip when seed-bypass is active so seeded requests stay
            # bit-reproducible regardless of concurrent in-flight prompts.
            # Inflight sharing borrows a donor's live cache and
            # trims it to the matched prefix — not safe for hybrid (non-trimmable
            # recurrent layers). Hybrid reuse goes through boundary snapshots only.
            _inflight_entry = None
            if cached_kv is None and not _bypass_cache and not _hybrid_mode:
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    _tracker = get_inflight_tracker()
                    _inflight_entry = _tracker.find_prefix(
                        [int(t) for t in ids], self.model_name or ""
                    )
                    # : DO NOT borrow the donor's LIVE kv_cache_ref. It was
                    # unsnapshotted+untrimmed: a full-length match yields an empty
                    # ids_to_prefill (generate_step on a (1,0) tensor → IndexError),
                    # and a partial match decodes into the donor's still-growing
                    # cache → wrong RoPE offsets / cross-request corruption. Safe
                    # only when the donor unregisters before any other fast-path
                    # _run — which the now-working engine loop breaks (a long-lived
                    # engine-loop donor stays registered). The snapshot-based
                    # persistent prefix cache already covers safe sharing, so the
                    # live-borrow reuse is disabled (registration below is kept and
                    # is harmless).
                except Exception:
                    _engine.logger.debug("inflight prefix lookup failed", exc_info=True)

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
                _engine.logger.debug("inflight prefix register failed", exc_info=True)

            gen_t0 = time.perf_counter()
            _timeout_deadline = gen_t0 + timeout_seconds
            first = True
            _itl_samples: list[float] = []
            _last_tok_time = 0.0
            _lprocs = logits_processors if logits_processors else None

            # HYBRID boundary-snapshot capture. Chunk-prefill
            # the prompt (minus its last token) into `cache`, storing a trim=0
            # snapshot at each block boundary so future shared-prefix requests
            # reuse losslessly. generate_step then prefills only the final token
            # and starts decoding. Any failure falls through to a normal full
            # prefill below (ids_to_prefill unchanged).
            if _hybrid_mode and len(ids_to_prefill) > 1:
                try:
                    _resident = self._capture_hybrid_prefix(
                        model,
                        ids,
                        cache,
                        self._kv_prefix_cache,
                        start_offset=cached_tokens,
                        block=self._hybrid_prefix_block,
                    )
                    if _resident >= len(ids) - 1:
                        ids_to_prefill = ids[-1:]
                except Exception:
                    _engine.logger.warning(
                        "hybrid prefix capture failed — full prefill",
                        exc_info=True,
                    )

            _timeout_check_interval = 32
            _fp_stats.admit(
                cached_tokens,
                len(ids_to_prefill),
                *_engine._prefix_cache_provenance(prefix_cache, _pc_hit, cached_tokens),
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
                ):
                    if first:
                        ttft_s = time.perf_counter() - gen_t0
                        first = False
                        # Feed cold-prefill throughput to the KV cache so the
                        # SSD tier can auto-gate fast-prefill models.
                        # Only on a cold prefill (cached==0) of a non-trivial
                        # prompt, where ttft_s ≈ pure prefill of ids_to_prefill.
                        if cached_tokens == 0 and ttft_s > 0:
                            try:
                                _n_pf = (
                                    int(ids_to_prefill.shape[0])
                                    if hasattr(ids_to_prefill, "shape")
                                    else len(ids_to_prefill)
                                )
                                if _n_pf >= 256:
                                    self._kv_prefix_cache.note_prefill_tps(
                                        _n_pf / ttft_s
                                    )
                            except Exception:
                                pass
                        # Prefill complete — remove from progress tracker
                        if _prefill_tracker is not None:
                            _prefill_tracker.update(
                                _prefill_req_id,
                                prompt_tokens,
                                prompt_tokens,
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
                    _fp_stats.token(len(tokens))
                    # Request-level timeout: check every N tokens
                    if len(tokens) % _timeout_check_interval == 0:
                        if time.perf_counter() > _timeout_deadline:
                            _engine.logger.warning(
                                f"Generation timed out after {timeout_seconds}s ({len(tokens)} tokens)"
                            )
                            break
                    # Progressive KV quantization (C6: keep memory flat during generation)
                    if _req_kv_bits is not None:
                        _engine._progressive_quantize_kv_cache(
                            cache,
                            self._kv_quant_start,
                            self._kv_quant_group_size,
                            _req_kv_bits,
                            len(tokens),
                        )
                    # Compute logprobs BEFORE stop checks — logprobs for stop
                    # tokens are trimmed later via lp_result[:len(tokens)].
                    if logprobs:
                        import mlx.core as mx

                        _lp_logits = logits.astype(mx.float32)
                        log_probs = _lp_logits - mx.logsumexp(
                            _lp_logits, axis=-1, keepdims=True
                        )
                        log_probs = mx.where(
                            mx.isnan(log_probs),
                            mx.array(-100.0, dtype=log_probs.dtype),
                            log_probs,
                        )
                        tok_lp = float(log_probs[token])
                        entry = {"token_id": int(token), "logprob": tok_lp}
                        if top_logprobs and top_logprobs > 0:
                            k = min(top_logprobs, log_probs.shape[0])
                            sorted_idx = mx.argsort(-log_probs)
                            top_k_idx = sorted_idx[:k]
                            entry["top_logprobs"] = [
                                {
                                    "token_id": int(top_k_idx[j]),
                                    "logprob": float(log_probs[int(top_k_idx[j])]),
                                }
                                for j in range(k)
                            ]
                        token_logprobs.append(entry)
                    # Stop ID check — must happen before thinking budget so that
                    # a stop token gets finish_reason="stop" even during thinking.
                    if token in stop_ids:
                        tokens.pop()  # Exclude stop token from output
                        _stopped_by_stop_id = True
                        if token in _user_stop_ids:
                            _user_stop_hit[0] = True
                        break
                    # Always add token to detokenizer for incremental state
                    # consistency — previously only added when stop_suffixes
                    # was non-empty, leaving the detokenizer empty and its
                    # state stale when no suffix matching was requested.
                    detokenizer.add_token(token)
                    if stop_suffixes and any(
                        detokenizer.text.endswith(s) for s in stop_suffixes
                    ):
                        tokens.pop()  # Exclude suffix-triggering token from count
                        _stopped_by_suffix = True
                        break
                    # Cancellation check
                    if _engine._is_cancelled(cancel_event):
                        mx.synchronize()
                        break
                    # Track thinking segment boundaries BEFORE budget check
                    # so that a natural </think token is detected first and
                    # the budget enforcement does not append a duplicate.
                    if think_start_token is not None:
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

                    # Thinking budget enforcement: cap thinking tokens.
                    # Only force-append think_end_token if the current token
                    # is NOT already the natural closing tag (avoids duplicate).
                    if thinking_budget is not None and _in_thinking:
                        thinking_tokens_used += 1
                        if (
                            thinking_tokens_used >= thinking_budget
                            and think_end_token is not None
                        ):
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
            if _req_kv_bits is not None:
                _engine._maybe_quantize_kv_cache(
                    cache,
                    self._kv_quant_start,
                    self._kv_quant_group_size,
                    _req_kv_bits,
                )
            # For HYBRID models the boundary snapshots were
            # already stored DURING prefill (before any generation). The post-
            # generation cache holds prompt+generated tokens, and a hybrid
            # snapshot CANNOT be trimmed back to prompt length (ArraysCache
            # recurrent state isn't sliceable) — adding it would store a state
            # polluted by generated tokens and corrupt future reuse. So skip the
            # trim-based adds for hybrid; the boundary snapshots are correct.
            if prefix_cache is not None and not _hybrid_mode:
                prefix_cache.add(ids, cache)

            # KV cache breakpoints: save prefix entries at Anthropic
            # cache_control positions so future requests can reuse the
            # KV state up to each breakpoint (multi-turn speedup).
            if (
                _kv_breakpoint_token_positions
                and prefix_cache is not None
                and not _hybrid_mode
            ):
                from .kv_prefix_cache import cache_length as _cache_len_bp

                for bp_pos in _kv_breakpoint_token_positions:
                    if bp_pos < len(ids) and bp_pos >= 32:
                        bp_tokens = ids[:bp_pos]
                        # Trim relative to the LIVE cache length,
                        # not len(ids). This runs AFTER generation, so `cache` holds
                        # len(ids)+num_generated tokens; trim=len(ids)-bp_pos retained
                        # bp_pos+num_generated tokens under a bp_pos-token key → the entry
                        # carried the prior request's generated continuation and corrupted
                        # a future prefix-hit (BUG-A cross-request leak). The main
                        # prefix_cache.add(ids, cache) self-corrects via cache_len-prompt_len;
                        # this hand-computed snapshot bypassed that.
                        trim_count = max(0, _cache_len_bp(cache) - bp_pos)
                        try:
                            bp_cache = prefix_cache._snapshot_cache(
                                cache, trim=trim_count
                            )
                            prefix_cache.add(bp_tokens, bp_cache)
                        except Exception:
                            _engine.logger.debug(
                                "KV breakpoint prefix add failed at pos %d",
                                bp_pos,
                                exc_info=True,
                            )

            # Prompt cache: store KV state for exact-match reuse.
            # CRITICAL: store the cache trimmed to prompt-length and report
            # token_count=len(ids). Storing prompt+completion length means
            # the next request with same prompt sees cached_tokens > prompt
            # length, ids_to_prefill becomes empty, the engine re-feeds the
            # last prompt token against KV state that contains *prior*
            # completion tokens, and produces gibberish that continues the
            # earlier response (BUG-A cross-request leakage).
            if (
                not _pc_hit
                and not _bypass_cache
                and not _hybrid_mode
                and hasattr(self, "_prompt_cache")
                and self._prompt_cache is not None
            ):
                try:
                    from .kv_prefix_cache import cache_length as _cache_len
                    from .prompt_cache import compute_messages_hash

                    _pc_hash = compute_messages_hash(
                        [{"role": "user", "content": text}],
                        model=self.model_name,
                    )
                    _prompt_only_len = int(len(ids))
                    _full_cache_len = int(_cache_len(cache))
                    _trim = max(0, _full_cache_len - _prompt_only_len)
                    if _trim > 0 and prefix_cache is not None:
                        # Reuse prefix_cache's snapshot+trim (it knows how to
                        # detach and slice KV layers safely).
                        _trimmed_cache = prefix_cache._snapshot_cache(cache, trim=_trim)
                    else:
                        _trimmed_cache = cache
                    self._prompt_cache.store(
                        _pc_hash,
                        _trimmed_cache,
                        token_count=_prompt_only_len,
                    )
                except Exception:
                    _engine.logger.debug("prompt cache store failed", exc_info=True)

            # Finalize detokenizer to flush any remaining partial UTF-8 bytes
            # before assembling final output text.
            try:
                detokenizer.finalize()
            except Exception:
                _engine.logger.debug(
                    "detokenizer finalize failed in fast path", exc_info=True
                )

            # When stop_suffix matching is active, use detokenizer text for
            # output because tokenizer.decode(tokens) may contain a partial
            # suffix that leaked across token boundaries. The detokenizer
            # has the complete incremental text including the suffix, which
            # we trim below. When no suffix matching, tokenizer.decode is
            # authoritative and avoids detokenizer state issues.
            if _stopped_by_suffix and stop_suffixes:
                output_text = detokenizer.text
                # Trim the matched suffix from detokenizer text
                for s in stop_suffixes:
                    if output_text.endswith(s):
                        output_text = output_text[: -len(s)]
                        break
                output_text = _engine._clean_special_tokens(output_text)
            else:
                output_text = tokenizer.decode(tokens, skip_special_tokens=True)
            # GUARANTEED stop-suffix truncation. The per-token
            # `detokenizer.text.endswith(s)` check misses stops that don't land
            # exactly at the (space-padded, one-token-lagging) detokenizer
            # boundary — e.g. bare words like "banana"/"END"/"User:" — so the stop
            # string leaked into the output with finish_reason="stop" coming from
            # natural EOS. Truncate the final text at the FIRST occurrence of any
            # stop string (OpenAI semantics) so it can never leak.
            if stop_suffixes and output_text:
                _cut = len(output_text)
                for s in stop_suffixes:
                    if s:
                        _i = output_text.find(s)
                        if _i != -1:
                            _cut = min(_cut, _i)
                if _cut < len(output_text):
                    output_text = output_text[:_cut]
                    _stopped_by_suffix = True
            mx.synchronize()
            _fp_stats.finish("stop")

            # Unregister from inflight prefix tracker
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_inflight_req_id)
            except Exception:
                _engine.logger.debug("inflight prefix unregister failed", exc_info=True)

            return (
                tokens,
                output_text,
                token_logprobs,
                ttft_s,
                cached_tokens,
                _stopped_by_suffix,
                _stopped_by_stop_id,
                _itl_samples,
                _thinking_tokens,
            )

        def _run_with_lora():
            # LoRA concurrency keystone: acquire+apply / release+restore on the
            # EXECUTOR thread so the whole adapter lifecycle is serialized with this
            # request's generate_step — and therefore with every other request's, since the
            # executor is max_workers=1. This removes the cross-thread race where an
            # event-loop _restore_base mutated the shared model while a different request's
            # generation read it. (Gateway-side apply is deferred for self-managing engines.)
            _lora_applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _lora_applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _lora_err:
                    # A registered adapter can still fail to apply at
                    # load time (arch mismatch / bad weights / max_loras exhausted).
                    # Fall-through would silently serve BASE-model output with a 200
                    # for a request that explicitly asked for the fine-tuned model —
                    # raise so it fails loudly instead.
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _lora_err
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
                            "LoRA release failed (fast-path executor)", exc_info=True
                        )

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        _fp_lock = getattr(self, "_fast_path_lock", None)
        if _fp_lock is not None:
            with _fp_lock:
                self._active_fast_path_count += 1
        try:
            try:
                (
                    tokens,
                    output_text,
                    token_logprobs,
                    ttft_s,
                    cached_tokens,
                    _stopped_by_suffix,
                    _stopped_by_stop_id,
                    _itl_samples,
                    _thinking_tokens,
                ) = await loop.run_in_executor(executor, _run_with_lora)
            except MemoryError:
                _engine.logger.warning(
                    "OOM during generation — returning memory_limit finish reason"
                )
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    get_inflight_tracker().unregister(_inflight_req_id)
                except Exception:
                    _engine.logger.debug(
                        "inflight prefix unregister failed in OOM handler",
                        exc_info=True,
                    )
                # Clear Metal buffers left behind by the OOM
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    _engine.logger.debug(
                        "GPU cache cleanup failed after OOM", exc_info=True
                    )
                return _engine.GenerationOutput(
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
                    _engine.logger.warning(f"MLX OOM during generation: {e}")
                    try:
                        from .inflight_prefix_sharing import get_inflight_tracker

                        get_inflight_tracker().unregister(_inflight_req_id)
                    except Exception:
                        _engine.logger.debug(
                            "inflight prefix unregister failed in OOM handler",
                            exc_info=True,
                        )
                    # Clear Metal buffers left behind by the OOM
                    try:
                        import mlx.core as _mx

                        await loop.run_in_executor(
                            executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                        )
                    except Exception:
                        _engine.logger.debug(
                            "GPU cache cleanup failed after OOM (RuntimeError path)",
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
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    get_inflight_tracker().unregister(_inflight_req_id)
                except Exception:
                    _engine.logger.debug(
                        "inflight prefix unregister failed in error handler",
                        exc_info=True,
                    )
                # Return error output for non-OOM RuntimeError too (e.g. shape
                # mismatch, unsupported op) instead of propagating to caller.
                return _engine.GenerationOutput(
                    finished=True,
                    finish_reason="error",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=f"RuntimeError during generation: {e}",
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            except Exception as e:
                _engine.logger.error(
                    f"Unexpected error during generation: {e}", exc_info=True
                )
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    get_inflight_tracker().unregister(_inflight_req_id)
                except Exception:
                    _engine.logger.debug(
                        "inflight prefix unregister failed in error handler",
                        exc_info=True,
                    )
                # Return an error GenerationOutput instead of propagating the
                # exception to the caller (which expects GenerationOutput, not
                # an exception). Previously this re-raised, causing unhandled
                # exceptions in the gateway handler.
                return _engine.GenerationOutput(
                    finished=True,
                    finish_reason="error",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=f"Generation failed: {e}",
                    ttft_ms=0.0,
                    cached_tokens=0,
                )

            # Stop-suffix truncation can leave completion_tokens (len(tokens))
            # counting tokens PAST the returned text — the per-token stop check missed
            # (stop not on a detokenizer boundary), generation ran to max_tokens, then
            # the text was find()-truncated at the stop. Re-encode the (truncated,
            # pre-reasoning-parse → TOTAL) text and report it when it's materially
            # fewer tokens, so we don't over-bill the discarded tail. None → len(tokens).
            _ct_override = None
            if _stopped_by_suffix and stop_suffixes and output_text is not None:
                try:
                    _ct_re = len(
                        tokenizer.encode(output_text, add_special_tokens=False)
                    )
                    if _ct_re < len(tokens):
                        _ct_override = _ct_re
                except Exception:
                    _ct_override = None

            # Decode token strings for logprobs
            lp_result = None
            if logprobs and token_logprobs:
                for lp_entry in token_logprobs:
                    tid = lp_entry["token_id"]
                    try:
                        lp_entry["token"] = tokenizer.decode([tid])
                        # Raw token bytes (NOT the decoded-string bytes, which
                        # are the U+FFFD replacement for a split multi-byte char).
                        from .text_utils import token_id_to_bytes

                        lp_entry["bytes"] = token_id_to_bytes(
                            tokenizer, tid, lp_entry["token"]
                        )
                    except Exception:
                        _engine.logger.debug(
                            "logprob token decode failed", exc_info=True
                        )
                        lp_entry["token"] = ""
                        lp_entry["bytes"] = []
                    if "top_logprobs" in lp_entry:
                        for tlp in lp_entry["top_logprobs"]:
                            try:
                                tlp["token"] = tokenizer.decode([tlp["token_id"]])
                                from .text_utils import token_id_to_bytes

                                tlp["bytes"] = token_id_to_bytes(
                                    tokenizer, tlp["token_id"], tlp["token"]
                                )
                            except Exception:
                                _engine.logger.debug(
                                    "top_logprob token decode failed", exc_info=True
                                )
                                tlp["token"] = ""
                                tlp["bytes"] = []
                lp_result = token_logprobs

            output_text = _engine._clean_special_tokens(output_text)

            # Trim stop suffix from output text when matched during generation
            if _stopped_by_suffix and stop_suffixes:
                for s in stop_suffixes:
                    if output_text.endswith(s):
                        output_text = output_text[: -len(s)]
                        break

            # Determine finish_reason.
            # Priority: cancel > stop (suffix or stop_id) > length
            # When cancel_event or timeout triggers, the loop breaks without
            # setting _stopped_by_suffix or _stopped_by_stop_id, so those
            # tokens correctly show up as "stop" only when genuinely stopped.
            _cancelled = _engine._is_cancelled(cancel_event)
            if _cancelled or _stopped_by_suffix or _stopped_by_stop_id:
                finish_reason = "stop"
            else:
                finish_reason = "length"

            # BUG FIX: When the first token is a stop_id (SpecPrefill path),
            # it is popped from `tokens` but its logprob entry remains in
            # `token_logprobs`. Trim the stale entry so logprobs count matches
            # `completion_tokens`.
            if lp_result is not None:
                lp_result = lp_result[: len(tokens)]

            # Record TTFT + ITL in Prometheus
            _ttft_ms_val = round(ttft_s * 1000, 1)
            if ttft_s > 0:
                try:
                    from yunshu_gateway.middleware.prometheus_exporter import (
                        get_prometheus_metrics,
                    )

                    pm = get_prometheus_metrics()
                    _ml = {"model_id": self.model_label}
                    pm.observe_histogram("ttft_seconds", ttft_s, labels=_ml)
                    if cached_tokens > 0:
                        pm.inc_counter("kv_prefix_cache_hits", labels=_ml)
                    else:
                        pm.inc_counter("kv_prefix_cache_misses", labels=_ml)
                    # ITL: record individual inter-token latency samples into histogram
                    if _itl_samples:
                        for _itl_sample in _itl_samples:
                            pm.observe_histogram("itl_seconds", _itl_sample, labels=_ml)
                except Exception:
                    _engine.logger.debug(
                        "TTFT/ITL prometheus recording failed", exc_info=True
                    )

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
                _engine.logger.debug(
                    "ServerMetrics recording failed in fast path", exc_info=True
                )

            self._total_reasoning_tokens += len(_thinking_tokens)

            # Channel-style reasoning recovery (Gemma-4 <|channel>…<channel|>).
            output_text, _ch_reason = _engine._recover_channel_reasoning(
                tokens, tokenizer, output_text
            )
            if _ch_reason:
                _thinking_tokens = _ch_reason

            # Reasoning parser: supplement token-level tracking with model-specific
            # reasoning extraction when thinking tokens were not explicitly tracked
            _reasoning_tok = len(_thinking_tokens)
            if output_text:
                try:
                    from .reasoning_parser import get_reasoning_parser

                    rp = get_reasoning_parser(self.model_name)
                    rp_out = rp.parse(output_text)
                    if rp_out.reasoning:
                        _reasoning_tok = rp_out.reasoning_tokens
                        if rp_out.content != output_text:
                            output_text = rp_out.content
                except Exception:
                    _engine.logger.debug(
                        "reasoning_parser failed in fast path", exc_info=True
                    )

            return _engine.GenerationOutput(
                text=output_text,
                new_text=output_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=_ct_override
                if _ct_override is not None
                else len(tokens),
                finished=True,
                finish_reason=finish_reason,
                cached_tokens=cached_tokens,
                logprobs=lp_result,
                ttft_ms=_ttft_ms_val,
                reasoning_tokens=_reasoning_tok,
                # A user stop sequence fired (vs natural EOS) — both map to "stop".
                stopped_by_stop_sequence=bool(_stopped_by_suffix or _user_stop_hit[0]),
            )
        finally:
            # LoRA release+restore now happens inside _run_with_lora on the executor
            # thread — not here on the event loop.
            _fp_lock = getattr(self, "_fast_path_lock", None)
            if _fp_lock is not None:
                with _fp_lock:
                    self._active_fast_path_count -= 1


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
