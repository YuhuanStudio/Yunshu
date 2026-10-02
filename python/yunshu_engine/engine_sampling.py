from __future__ import annotations

"""Engine sampling extracted from batched_engine.

Runtime dependencies stay on the compatibility facade so existing patches apply.
"""

from typing import Any


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


def _apply_spec_bonus_penalties(
    logits: Any,
    all_token_ids: list[int],
    prompt_token_count: int,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logit_bias: dict[int, float] | None = None,
    recent_ctx: int = 20,
) -> Any:
    """Apply penalty and logit_bias processors to bonus token logits.

    For speculative decode paths, penalties are only applied to the BONUS
    token (first new token after acceptance) since draft tokens are already
    committed.

    Args:
        logits: 1-D or 2-D logits array (vocab_size,) or (1, vocab_size).
        all_token_ids: Full token history (prompt + generated).
        prompt_token_count: Number of prompt tokens (for freq/presence penalty).
        repetition_penalty: Multiplier for repeated tokens (>1 = penalize).
        frequency_penalty: Subtractive penalty proportional to token count.
        presence_penalty: Subtractive penalty for any token seen >0 times.
        logit_bias: Additive bias per token ID.
        recent_ctx: How many recent tokens to check for repetition penalty.

    Returns:
        Modified logits array (same shape).
    """
    import mlx.core as _mx

    # Generated tokens only (exclude prompt)
    gen_tokens = (
        all_token_ids[prompt_token_count:]
        if len(all_token_ids) > prompt_token_count
        else []
    )

    if repetition_penalty != 1.0 and len(gen_tokens) > 0:
        recent = gen_tokens[-recent_ctx:]
        unique_recent = list(dict.fromkeys(recent))
        sel = logits[..., unique_recent]
        sel = _mx.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)
        logits[..., _mx.array(unique_recent)] = sel

    if (frequency_penalty != 0.0 or presence_penalty != 0.0) and len(gen_tokens) > 0:
        counts: dict[int, int] = {}
        for t in gen_tokens:
            counts[t] = counts.get(t, 0) + 1
        for tid, cnt in counts.items():
            # OpenAI permits frequency/presence penalty in [-2.0, 2.0].
            # Negative values BOOST the token (encourage repetition); the
            # earlier code only applied when > 0 which silently dropped
            # negative penalties — confirmed via curl with pp=-2.0 producing
            # identical output to pp=0. Apply for any non-zero value.
            if frequency_penalty != 0.0:
                logits[..., tid] = logits[..., tid] - frequency_penalty * cnt
            if presence_penalty != 0.0 and cnt > 0:
                logits[..., tid] = logits[..., tid] - presence_penalty

    if logit_bias:
        # Guard against out-of-range / negative token ids (see _logit_bias_proc):
        # an invalid user-supplied id would index out of bounds and crash.
        vocab = logits.shape[-1]
        for tid, bias in logit_bias.items():
            if 0 <= tid < vocab:
                logits[..., tid] = logits[..., tid] + bias

    return logits


def _build_noncached_sampler_text(
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    seed: int | None,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    top_n_sigma: float = 0.0,
):
    """Numpy-backed sampler that bypasses mlx-lm's @mx.compile cache.

    mlx-lm's `categorical_sampling` is wrapped with
    `@mx.compile(inputs=mx.random.state, outputs=mx.random.state)`. The
    compile cache traps the first call's PRNG state, so subsequent
    requests with temperature>0 produce identical token streams even
    after `mx.random.seed()` between them — the cause of the repeated-output
    bug for VLM and the n>1 collapse seen on /v1/chat/completions.

    For temperature>0 use `numpy.random.Generator` (independent RNG per
    request). Always returns an mx.array compatible with mlx-lm's
    generate_step.
    """
    import time as _t

    import numpy as _np

    base = (
        int(seed) & ((1 << 63) - 1)
        if seed is not None
        else _t.time_ns() & ((1 << 63) - 1)
    )
    rng = _np.random.default_rng(base)
    _t_ = float(temperature)
    _tp = float(top_p) if top_p else 1.0
    _tk = int(top_k) if top_k and top_k > 0 else 0
    _mp = float(min_p) if min_p else 0.0
    # XTC (eXclude Top Choices) was silently dropped on this
    # default non-streaming fast path — the request was accepted but produced non-XTC
    # output (the streaming path applied it, so behavior diverged by the `stream` flag).
    _xtc_p = float(xtc_probability) if xtc_probability else 0.0
    _xtc_thr = float(xtc_threshold) if xtc_threshold else 0.0
    # top-nσ (ACL 2025): keep only tokens whose RAW logit is within n·σ of the max
    # logit (σ = std of the logit row). Temperature-invariant (operates on logits,
    # not the tempered distribution); a pure-quality reasoning filter applied FIRST,
    # before the prob-based nucleus/min_p filters.
    _nsig = float(top_n_sigma) if top_n_sigma and top_n_sigma > 0 else 0.0

    def _sampler(logits):
        import mlx.core as _mx

        arr = _np.asarray(logits.astype(_mx.float32))
        flat = arr.reshape(-1, arr.shape[-1])
        out = _np.empty(flat.shape[0], dtype=_np.int64)
        for i in range(flat.shape[0]):
            # Build the surviving-token set on the UN-tempered softmax,
            # then apply temperature LAST (only when forming the final categorical
            # distribution). mlx-lm's make_sampler runs apply_top_p/min_p/xtc/top_k
            # on the raw logprobs and applies temp inside categorical_sampling
            # (`logprobs * (1/temp)`) — temp LAST. The old code divided by temp
            # FIRST, so for temp!=1 + any filter the nucleus/min_p/XTC threshold
            # selected a DIFFERENT token set than the streaming path → same request
            # sampled from a materially different distribution depending on the
            # `stream` flag. An earlier fix only corrected the order AMONG the filters.
            raw = flat[i].astype(_np.float64)
            l = raw - _np.max(raw)
            p = _np.exp(l)
            p = p / p.sum()
            # top-nσ FIRST (logit-statistics filter): drop tokens whose raw logit
            # is more than n·σ below the max logit. σ is the std of the full logit
            # row; the threshold is temperature-independent.
            if _nsig > 0:
                thr = raw.max() - _nsig * raw.std()
                m = (raw >= thr).astype(_np.float64)
                p = p * m
                _s = p.sum()
                if _s > 0:
                    p = p / _s
            # Apply filters in mlx-lm make_sampler's ORDER — top_p → min_p →
            # XTC → top_k — so this non-streaming temp>0 sampler produces the SAME
            # distribution as the streaming path (which uses make_sampler) for the same
            # params. The old order (top_k first) gave a different surviving token set
            # when both top_k and top_p were set → stream/non-stream diverged.
            if 0 < _tp < 1:
                order = _np.argsort(-p)
                cum = _np.cumsum(p[order])
                # Nucleus = smallest set whose cumulative prob >= top_p, INCLUDING
                # the token that crosses the threshold (matches mlx-lm apply_top_p).
                # `order[cum <= _tp]` dropped the crossing token → nucleus too narrow.
                k = int(_np.searchsorted(cum, _tp, side="left")) + 1
                keep = order[: max(1, min(k, len(order)))]
                m = _np.zeros_like(p)
                m[keep] = 1.0
                p = p * m
                p = p / p.sum()
            if _mp > 0:
                pmax = p.max()
                m = (p >= _mp * pmax).astype(_np.float64)
                p = p * m
                p = p / p.sum()
            # XTC: with probability _xtc_p, remove every token whose prob is above the
            # threshold EXCEPT the least-probable one among them (matches mlx-lm
            # apply_xtc: mask = probs > min(probs where probs > threshold)). Keeps the
            # low-confidence tail, dropping the high-prob cluster → more diverse output.
            if _xtc_p > 0.0 and rng.random() < _xtc_p:
                above = p > _xtc_thr
                if above.any():
                    min_above = p[above].min()
                    remove = p > min_above
                    if remove.any() and not remove.all():
                        p = p * (~remove).astype(_np.float64)
                        _s = p.sum()
                        if _s > 0:
                            p = p / _s
            if _tk and _tk < len(p):
                idx = _np.argpartition(p, -_tk)[-_tk:]
                m = _np.zeros_like(p)
                m[idx] = 1.0
                p = p * m
                p = p / p.sum()
            # Apply temperature LAST, restricted to the survivor set (p>0), exactly
            # like mlx-lm's categorical_sampling(masked_logprobs, temp). Sampling
            # the un-tempered survivor `p` directly would ignore temperature; sample
            # from softmax(raw_logits/temp) over the kept tokens instead.
            keep = p > 0
            lf = _np.where(keep, raw / _t_, -_np.inf)
            lf = lf - _np.max(lf)
            pf = _np.exp(lf)
            _sf = pf.sum()
            pf = pf / _sf if _sf > 0 else p
            out[i] = int(rng.choice(len(pf), p=pf))
        return _mx.array(out.reshape(arr.shape[:-1]).astype(_np.int64))

    return _sampler


def _build_gpu_sampler_text(
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    seed: int | None,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    top_n_sigma: float = 0.0,
):
    """On-GPU temp>0 sampler (opt-in via YUNSHU_GPU_SAMPLER=1).

    The default numpy sampler (_build_noncached_sampler_text) does
    `np.asarray(logits)` INSIDE mlx-lm's _step → a forced GPU→CPU sync EVERY
    token, which breaks generate_step's async pipeline (it can no longer overlap
    the next forward with token consumption) and runs a full-vocab numpy softmax
    on the CPU (~1100µs/token on a 152k vocab). This sampler stays on the GPU:

    - Uses mlx-lm's OWN filter functions (apply_top_p/min_p/xtc/top_k) — the exact
      chain the streaming make_sampler uses — so the distribution is identical to
      both the streaming path and the numpy path.
    - Replaces mlx-lm's categorical_sampling (the @mx.compile(inputs=mx.random.state)
      one whose compile-cache traps the PRNG → the n>1 collapse the numpy sampler
      exists to avoid) with Gumbel-max keyed by an EXPLICIT per-request mx.random.key
      that is split per step. Gumbel-max(x) is distributionally identical to
      categorical(softmax(x)), and the explicit key gives per-request
      reproducibility WITHOUT the compile-cache trap.

    Returns an mx.array (no .item()/np sync), preserving the async decode pipeline.
    """
    import time as _t

    import mlx.core as mx
    from mlx_lm.sample_utils import (
        apply_min_p,
        apply_top_k,
        apply_top_p,
    )

    _t_ = float(temperature)
    _nsig = float(top_n_sigma) if top_n_sigma and float(top_n_sigma) > 0 else 0.0
    methods = []
    if top_p and 0 < float(top_p) < 1.0:
        _tp = float(top_p)
        methods.append(lambda x: apply_top_p(x, _tp))
    if min_p and float(min_p) != 0.0:
        _mp = float(min_p)
        methods.append(lambda x: apply_min_p(x, _mp))
    # top_k must be applied AFTER XTC (mlx-lm order: top_p→min_p→XTC→top_k,
    # matching the default numpy sampler). Keeping it in `methods` applied it BEFORE the
    # XTC block below, so XTC computed its cutoff over a top_k-truncated distribution —
    # masking a different token set than mlx-lm and breaking this sampler's stated
    # "identical to the numpy path" invariant. Apply it last instead.
    _top_k_n = int(top_k) if (top_k and int(top_k) > 0) else 0

    # XTC handled INLINE (not via mlx-lm's apply_xtc) because that function is
    # @mx.compile(inputs=mx.random.state) → its compile-cache traps the global
    # PRNG exactly like categorical_sampling does (the n>1-collapse this sampler
    # exists to avoid). We replicate its math but draw the coin from our explicit
    # per-request split key so XTC stays reproducible, on-GPU, and trap-free.
    _xtc_on = bool(xtc_probability and float(xtc_probability) > 0.0)
    _xp, _xt = float(xtc_probability), float(xtc_threshold)
    if _xtc_on and not (0 <= _xt <= 0.5):
        raise ValueError(f"xtc_threshold must be in [0, 0.5], got {_xt}")

    base = (
        int(seed) & ((1 << 63) - 1)
        if seed is not None
        else _t.time_ns() & ((1 << 63) - 1)
    )
    state = {"key": mx.random.key(base)}

    def _sampler(logprobs):
        lp = logprobs
        # top-nσ FIRST: mask tokens > n·σ below the max. log_softmax is logits minus
        # a per-row constant, so max/std (hence the mask) match the raw-logit form.
        if _nsig > 0:
            _mu = lp.mean(axis=-1, keepdims=True)
            _sd = (((lp - _mu) ** 2).mean(axis=-1, keepdims=True)) ** 0.5
            _thr = lp.max(axis=-1, keepdims=True) - _nsig * _sd
            lp = mx.where(lp >= _thr, lp, -mx.inf)
        for m in methods:
            lp = m(lp)
        if _xtc_on:
            # Inline XTC: drop every token whose prob exceeds the smallest prob
            # above the threshold, gated by a per-step coin from the explicit key.
            probs = mx.softmax(lp, axis=-1)
            cutoff = mx.where(probs > _xt, probs, mx.inf).min(axis=-1, keepdims=True)
            mask = probs > cutoff
            state["key"], coin_key = mx.random.split(state["key"])
            coin = mx.random.uniform(0, 1, key=coin_key)
            lp = mx.where(coin > _xp, lp, mx.where(mask, -mx.inf, lp))
        # top_k LAST (after XTC), matching mlx-lm / the numpy sampler. Guard
        # top_k >= vocab — mlx-lm's apply_top_k raises there (numpy no-ops; keep parity).
        if _top_k_n and _top_k_n < lp.shape[-1]:
            lp = apply_top_k(lp, _top_k_n)
        # Temperature LAST (matches categorical_sampling: logprobs * 1/temp), then
        # Gumbel-max with a freshly-split explicit key (no global PRNG state).
        state["key"], sub = mx.random.split(state["key"])
        u = mx.random.uniform(shape=lp.shape, key=sub)
        g = -mx.log(-mx.log(u))
        return mx.argmax(lp * (1.0 / _t_) + g, axis=-1)

    return _sampler


def _build_temp_sampler(
    temperature,
    top_p,
    top_k,
    min_p,
    seed,
    xtc_probability=0.0,
    xtc_threshold=0.0,
    top_n_sigma=0.0,
):
    """Pick the temp>0 sampler. Default = numpy (proven, avoids the
    @mx.compile PRNG trap). YUNSHU_GPU_SAMPLER=1 = on-GPU Gumbel-max (no per-token
    GPU→CPU sync, preserves mlx-lm's async pipeline; same distribution)."""
    # top-nσ resolution order: explicit call arg → per-request value (set on the
    # ContextVar by generate()/stream_generate()).
    # The sampler is built on the request's event-loop task BEFORE the executor
    # decode, so the ContextVar set in the entrypoint is visible here and gets
    # baked into the returned closure — no need to thread the value through the
    # deep fast-path call chain.
    if not top_n_sigma or float(top_n_sigma) <= 0:
        _req_nsig = _engine._REQUEST_TOP_N_SIGMA.get()
        top_n_sigma = _req_nsig if _req_nsig and _req_nsig > 0 else 0.0

    if _engine.settings.get_bool("YUNSHU_GPU_SAMPLER"):
        return _engine._build_gpu_sampler_text(
            temperature,
            top_p,
            top_k,
            min_p,
            seed,
            xtc_probability,
            xtc_threshold,
            top_n_sigma,
        )
    return _engine._build_noncached_sampler_text(
        temperature,
        top_p,
        top_k,
        min_p,
        seed,
        xtc_probability,
        xtc_threshold,
        top_n_sigma,
    )


def _build_constrained_sampler(sampler, json_schema, tokenizer):
    """Build a constrained sampler from a grammar specification.

    Handles:
    - JSON schema dict → JsonSchemaConstraint
    - "json_object" string → generic JSON constraint
    - {"type": "regex", "pattern": "..."} → RegexConstraint
    - {"type": "choice", "choices": [...]} → ChoiceConstraint
    - {"type": "cfg", "grammar": "..."} → CfgGrammarConstraint

    When YUNSHU_GRAMMAR_BITMASK=1 is set, uses the bitmask engine instead
    of the allowlist-based ConstrainedSampler (xgrammar-style approach).
    """
    # Determine grammar type and payload
    grammar_type = None
    grammar = None

    if isinstance(json_schema, dict) and json_schema.get("type") in (
        "regex",
        "choice",
        "cfg",
    ):
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
        from .grammar_bitmask import (
            BitmaskConstrainedSampler,
            build_bitmask_engine,
            is_bitmask_enabled,
        )

        if is_bitmask_enabled():
            try:
                engine = build_bitmask_engine(grammar_type, grammar)
                return BitmaskConstrainedSampler(sampler, engine, tokenizer)
            except Exception:
                _engine.logger.debug(
                    "bitmask engine setup failed, falling back to allowlist",
                    exc_info=True,
                )
    except ImportError:
        _engine.logger.debug(
            "grammar_bitmask module not available, using allowlist path", exc_info=True
        )

    # ── Standard allowlist path ──
    if grammar_type in ("regex", "choice", "cfg"):
        from .grammar_constraint import ConstraintFactory

        try:
            constraint = ConstraintFactory.create(grammar_type, grammar, tokenizer)
            from .json_schema import ConstrainedSampler

            return ConstrainedSampler(sampler, constraint, tokenizer)
        except Exception:
            _engine.logger.debug(
                "grammar constraint setup failed, returning unconstrained sampler",
                exc_info=True,
            )
            return sampler

    # Standard JSON schema path
    from .grammar_constraint import build_json_constraint
    from .json_schema import ConstrainedSampler

    if isinstance(json_schema, str):
        if json_schema == "json_object":
            # Generic JSON object mode — no specific schema
            constraint = build_json_constraint(None, tokenizer)
        else:
            import json as _json

            schema = _json.loads(json_schema)
            constraint = build_json_constraint(schema, tokenizer)
    else:
        constraint = build_json_constraint(json_schema, tokenizer)
    return ConstrainedSampler(sampler, constraint, tokenizer)


def _build_grammar_constraint(json_schema, tokenizer):
    """Build the RAW constraint object (not a sampler wrapper) from a grammar spec.

    Routes {"type": "regex"|"choice"|"cfg"} to ConstraintFactory and everything else
    (a JSON-schema dict, the "json_object" string, or a JSON-schema string) to
    JsonSchemaConstraint. The speculative decoder drives the constraint directly
    (advance / rollback / get_allowed_tokens), so it needs the bare constraint, not the
    ConstrainedSampler wrapper that _build_constrained_sampler returns.

    The cross-model spec paths previously wrapped EVERY grammar dict in
    JsonSchemaConstraint, so a {"type":"regex"/"choice"/"cfg"} spec was silently turned
    into a JSON-object constraint (_get_type_from_schema → "regex" falls through to the
    default object branch) — the model was forced to emit JSON and the user's regex /
    choice / cfg constraint was dropped with no error. This mirrors the correct routing
    in _build_constrained_sampler.
    """
    if isinstance(json_schema, dict) and json_schema.get("type") in (
        "regex",
        "choice",
        "cfg",
    ):
        gtype = json_schema["type"]
        if gtype == "regex":
            grammar = json_schema.get("pattern", "")
        elif gtype == "choice":
            grammar = json_schema.get("choices", [])
        else:  # cfg
            grammar = json_schema.get("grammar", "")
        from .grammar_constraint import ConstraintFactory

        return ConstraintFactory.create(gtype, grammar, tokenizer)

    from .grammar_constraint import build_json_constraint

    if isinstance(json_schema, str) and json_schema != "json_object":
        import json as _json

        return build_json_constraint(_json.loads(json_schema), tokenizer)
    if json_schema == "json_object":
        return build_json_constraint(None, tokenizer)
    return build_json_constraint(json_schema, tokenizer)


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
