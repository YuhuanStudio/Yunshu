"""Fused chunked prefill: the decoding rows and a prefill chunk in one forward.

The runner's shared batch (upstream mlx-vlm ``BatchGenerator``) runs a waiting
prompt's prefill and the decoding rows as separate forwards, so while a long
prompt prefills every decoding row gets one token per prefill chunk (27B, 4 rows
decoding, 16K prompt in 2048-token chunks: ~0.7 tok/s per row for 17 s). Smaller
chunks alternating with decode steps barely help (256: 2.9 tok/s, +21% TTFT),
because each extra forward reads all the weights again.

Here, when one scheduling step would run both a decode step and a prefill step
(upstream's ``_next`` does both, decode first), the two model calls become one
forward over the packed tokens ``[decode rows (1 token each) | prefill chunk]``:

- every weight-bearing op runs once over the packed tokens — the attention and
  GatedDeltaNet in/out projections, the MLP, the norms — so the decode rows ride
  along a prefill forward that reads the weights anyway (decode is bandwidth
  bound; a few hundred prefill tokens are compute the step has to spare);
- only the sequence mixers run per segment, through the model's own modules and
  caches (each row's KV / recurrent state, masks, rotary positions, ragged or
  stock attention, the fused GDN decode kernels), so cache bookkeeping, APC
  checkpoints and multimodal positions stay upstream's.

Plumbing, without re-implementing upstream's batching:

1. ``plan`` (before ``BatchGenerator.next``) sets the prefill chunk to the token
   budget while rows decode, and when this ``next`` will run a decode step and
   a prefill step it *captures* both model calls: it calls the language model
   with each call's exact arguments while ``Qwen3_5Model.__call__`` raises before
   touching any cache, after the language model's own position / mRoPE
   bookkeeping ran in the same order upstream runs it;
2. inside ``next``, the first real call runs the fused forward for both segments
   and each call gets back its own hidden states (``FusedStep.take``). Upstream
   then samples, applies logits processors / logprobs, stores APC checkpoints and
   advances the prompt exactly as it would have.

Numerics: a row's arithmetic is the same as unfused except the projections'
matmul shape (packed rows instead of the segment's own rows), i.e. the same
class of difference as batch composition already causes in the shared batch.
The speculative lane never fuses.

Qwen3.5 family only (hybrid GatedDeltaNet + attention; dense and MoE).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Per mixer: the projections of the layer input (computed once over the packed
# tokens) and the output projection (applied once to the packed mixer outputs).
_IN = {
    "linear_attn": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
    "self_attn": ("q_proj", "k_proj", "v_proj"),
}
_OUT = {"linear_attn": "out_proj", "self_attn": "o_proj"}

_STATE: dict = {
    "installed": False,
    "capture": False,
    "step": None,
    "logged": False,
    "steps": 0,
    "tokens": 0,
}


class _CapturedCallError(Exception):
    """Raised by the patched model call in capture mode, carrying its args."""

    def __init__(self, call: dict):
        super().__init__("captured")
        self.call = call


@dataclass
class _Segment:
    cache: list
    inputs: mx.array
    inputs_embeds: mx.array | None
    position_ids: mx.array | None
    hidden: mx.array | None = None
    taken: bool = False
    # set by the forward
    view: list | None = None
    fa_mask: Any = None
    ssm_mask: Any = None
    position_embeddings: Any = None
    rows: int = 0
    length: int = 0


class FusedStep:
    """The captured decode + prefill calls of one ``BatchGenerator.next``."""

    def __init__(self, model: Any, segments: list[_Segment]):
        self.model = model
        self.segments = segments
        self.computed = False

    def take(self, model: Any, inputs: mx.array, cache: list) -> mx.array | None:
        seg = next((s for s in self.segments if s.cache is cache), None)
        if seg is None:
            return None
        if model is not self.model or seg.taken or inputs.shape != seg.inputs.shape:
            raise RuntimeError(
                "fused prefill: model call does not match the planned step "
                f"(inputs {inputs.shape} vs {seg.inputs.shape}, taken={seg.taken})"
            )
        if not self.computed:
            outs = forward(self.model, self.segments)
            for s, h in zip(self.segments, outs, strict=True):
                s.hidden = h
            self.computed = True
        seg.taken = True
        return seg.hidden

    def done(self) -> bool:
        return all(s.taken for s in self.segments)


def supports(language_model: Any) -> bool:
    """Qwen3.5-family language model (hybrid GDN + attention decoder)."""
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:  # pragma: no cover - mlx-vlm without qwen3_5
        return False
    inner = getattr(language_model, "model", None)
    if not isinstance(inner, q35.Qwen3_5Model):
        return False
    return all(
        hasattr(layer, "is_linear")
        and hasattr(layer, "input_layernorm")
        and hasattr(layer, "post_attention_layernorm")
        and hasattr(layer, "mlp")
        for layer in inner.layers
    )


def install() -> None:
    """Patch ``Qwen3_5Model.__call__`` (once) for capture and replay; with no
    step planned it is the stock call."""
    if _STATE["installed"]:
        return
    from mlx_vlm.models.qwen3_5 import language as q35

    cls = q35.Qwen3_5Model
    orig = cls.__call__

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        position_ids=None,
        capture_layer_ids=None,
        hidden_sink=None,
    ):
        if _STATE["capture"]:
            raise _CapturedCallError(
                {
                    "model": self,
                    "inputs": inputs,
                    "inputs_embeds": inputs_embeds,
                    "mask": mask,
                    "cache": cache,
                    "position_ids": position_ids,
                    "capture_layer_ids": capture_layer_ids,
                    "hidden_sink": hidden_sink,
                }
            )
        step = _STATE["step"]
        if step is not None and cache is not None:
            out = step.take(self, inputs, cache)
            if out is not None:
                return out
        return orig(
            self,
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
        )

    cls.__call__ = __call__
    _STATE["installed"] = True


# ── planning ─────────────────────────────────────────────────────────────────


def prefill_chunk(pb: Any) -> int:
    """Tokens upstream's ``PromptProcessingBatch.prompt_step`` processes next
    (mirrors its chunk arithmetic)."""
    remaining = int(pb._inputs_embeds.shape[1])
    step = pb.prefill_step_size or remaining
    n = min(step, remaining - 1)
    if pb._right_pad_per_row is not None:
        start = pb._processed_prompt_columns
        pending = [n_ for n_ in pb._suffix_lens if n_ > start]
        if pending:
            n = min(n, min(pending) - start)
    col = pb._next_apc_checkpoint_column()
    if col is not None:
        n = min(n, col - pb._processed_prompt_columns)
    return n


def set_chunk(gen: Any, budget: int, full: int) -> bool:
    """Prefill chunk: ``budget`` tokens while rows decode (so each fused step
    stays short), ``full`` otherwise. Returns whether rows are decoding."""
    gb = gen._generation_batch
    decoding = gb is not None and len(gb) > 0
    step = budget if decoding else full
    gen.prefill_step_size = step
    pb = gen._prompt_batch
    if pb is not None and pb.prefill_step_size is not None:
        pb.prefill_step_size = step
    return decoding


def _capture(lm: Any, *args, **kwargs) -> dict:
    _STATE["capture"] = True
    try:
        lm(*args, **kwargs)
    except _CapturedCallError as c:
        return c.call
    finally:
        _STATE["capture"] = False
    # The language model returned without reaching the decoder: it ran some
    # other forward (and may have written a cache). Never continue from here.
    raise RuntimeError("fused prefill: capture ran a forward outside the decoder")


def _eligible(gen: Any, lm: Any) -> bool:
    gb, pb = gen._generation_batch, gen._prompt_batch
    if pb is None or getattr(pb, "draft_model", None) is not None:
        return False
    if (
        getattr(gb, "is_speculative", False)
        or getattr(gb, "_next_tokens", None) is None
    ):
        return False
    if len(gb) >= gen.completion_batch_size:
        return False  # upstream returns before the prefill step
    if any(
        getattr(p, "requires_immediate_decode_yield", False)
        for procs in getattr(gb, "logits_processors", []) or []
        for p in procs or []
    ):
        return False  # upstream yields right after the decode step
    if (
        gb.greedy_sampling
        and not gb.compute_logprobs
        and gb.top_logprobs_k == 0
        and callable(getattr(lm, "fused_greedy_decode", None))
    ):
        return False  # the decode step would not call the decoder
    if callable(getattr(lm, "_batch_invariant_decode", None)):
        return False
    return supports(lm)


def plan(gen: Any, budget: int, full: int) -> FusedStep | None:
    """Arm a fused step for the next ``gen.next()`` when it will run a decode
    step and a prefill step; always sets the prefill chunk size."""
    decoding = set_chunk(gen, budget, full)
    lm = gen.model
    if not decoding or not _eligible(gen, lm):
        return None
    install()
    gb, pb = gen._generation_batch, gen._prompt_batch
    with mx.stream(gen._stream):
        fwd = {} if gb._rope_deltas is None else {"rope_deltas": gb._rope_deltas}
        dec = _capture(lm, gb._next_tokens[:, None], cache=gb.prompt_cache, **fwd)
        if pb.needs_processing():
            n = prefill_chunk(pb)
            if n <= 0:
                return None
            kw = {**pb._prompt_kwargs_for_step(n), **pb._speculative_prefill.kwargs}
            pre = _capture(
                lm,
                pb._input_ids[:, :n],
                cache=pb.prompt_cache,
                inputs_embeds=pb._inputs_embeds[:, :n],
                n_to_process=n,
                **kw,
            )
        else:  # the last prompt token (PromptProcessingBatch.generate)
            kw = dict(pb._prompt_kwargs)
            kw["logits_to_keep"] = 1 + max(
                (
                    pad
                    for i, pad in enumerate(pb._right_pad_per_row or [])
                    if i not in pb._finished_prompt_logits
                ),
                default=0,
            )
            pre = _capture(
                lm,
                pb._input_ids,
                cache=pb.prompt_cache,
                inputs_embeds=pb._inputs_embeds,
                **kw,
            )
    segments = []
    for call in (dec, pre):
        if (
            call["mask"] is not None
            or call["capture_layer_ids"] is not None
            or call["hidden_sink"] is not None
            or call["cache"] is None
        ):
            return None
        segments.append(
            _Segment(
                cache=call["cache"],
                inputs=call["inputs"],
                inputs_embeds=call["inputs_embeds"],
                position_ids=call["position_ids"],
            )
        )
    step = FusedStep(dec["model"], segments)
    _STATE["step"] = step
    _STATE["steps"] += 1
    _STATE["tokens"] += int(segments[1].inputs.shape[-1])
    if not _STATE["logged"]:
        _STATE["logged"] = True
        logger.info(
            "Fused prefill engaged: %d decode rows + %d prefill tokens per step",
            len(gb),
            int(segments[1].inputs.shape[-1]),
        )
    return step


def stats() -> dict:
    """Fused steps planned so far and the prefill tokens they carried."""
    return {"steps": _STATE["steps"], "tokens": _STATE["tokens"]}


def finish(step: FusedStep | None) -> None:
    """Disarm after ``gen.next()``; a computed step that was not fully consumed
    advanced a cache upstream did not account for, so it must not go unnoticed."""
    _STATE["step"] = None
    if step is not None and step.computed and not step.done():
        raise RuntimeError("fused prefill: a planned model call did not happen")


# ── the fused forward ────────────────────────────────────────────────────────


class _Memo:
    """Stands in for a projection: returns the precomputed rows for the
    segment inputs it was given, computes anything else itself."""

    def __init__(self, inner: Any, table: dict):
        self.inner = inner
        self.table = table

    def __call__(self, x):
        hit = self.table.get(id(x))
        if hit is not None and hit[0] is x:
            return hit[1]
        return self.inner(x)


class _Defer:
    """Stands in for the output projection: records the segment's mixer output
    and returns a placeholder; the real projection runs once over all
    segments afterwards."""

    def __init__(self, width: int):
        self.width = width
        self.last: tuple | None = None

    def __call__(self, x):
        placeholder = mx.zeros((*x.shape[:-1], self.width), dtype=x.dtype)
        self.last = (x, placeholder)
        return placeholder


def _split(x: mx.array, segments: list[_Segment]) -> list[mx.array]:
    """Packed ``[1, T, W]`` -> each segment's ``[rows, length, W]``."""
    out, start = [], 0
    for s in segments:
        n = s.rows * s.length
        out.append(x[:, start : start + n].reshape(s.rows, s.length, x.shape[-1]))
        start += n
    return out


def _pack(parts: list[mx.array]) -> mx.array:
    return mx.concatenate([p.reshape(1, -1, p.shape[-1]) for p in parts], axis=1)


def _mixer(mixer, name, layer_idx, linear, xn, parts, segments, width) -> mx.array:
    """Run one layer's sequence mixer per segment with shared projections;
    returns the packed mixer output ``[1, T, width]``."""
    saved = {}
    try:
        for proj in _IN[name]:
            inner = mixer[proj]
            full = inner(xn)
            table = {
                id(p): (p, sl)
                for p, sl in zip(parts, _split(full, segments), strict=True)
            }
            saved[proj] = inner
            mixer[proj] = _Memo(inner, table)
        out_name = _OUT[name]
        out_proj = mixer[out_name]
        saved[out_name] = out_proj
        defer = _Defer(width)
        mixer[out_name] = defer
        results: list = []
        for s, p in zip(segments, parts, strict=True):
            defer.last = None
            cache = s.view[layer_idx]
            if linear:
                o = mixer(p, s.ssm_mask, cache)
            else:
                o = mixer(
                    p,
                    mask=s.fa_mask,
                    cache=cache,
                    position_ids=s.position_ids,
                    position_embeddings=s.position_embeddings,
                )
            if defer.last is not None and o is defer.last[1]:
                results.append(("defer", defer.last[0]))
            else:  # the mixer projected its output itself
                results.append(("done", o))
    finally:
        for k, v in saved.items():
            mixer[k] = v
    deferred = [x for kind, x in results if kind == "defer"]
    projected = iter(
        _split(
            out_proj(_pack(deferred)),
            [
                s
                for (kind, _), s in zip(results, segments, strict=True)
                if kind == "defer"
            ],
        )
        if deferred
        else []
    )
    outs = [next(projected) if kind == "defer" else x for kind, x in results]
    return _pack(outs)


def forward(model: Any, segments: list[_Segment]) -> list[mx.array]:
    """``Qwen3_5Model.__call__`` for several independent segments at once;
    returns each segment's final hidden states ``[rows, length, D]``."""
    from mlx_vlm.models.qwen3_5 import language as q35

    layers = model.layers
    hs, merges = [], []
    for s in segments:
        h = (
            s.inputs_embeds
            if s.inputs_embeds is not None
            else model.embed_tokens(s.inputs)
        )
        s.rows, s.length = int(h.shape[0]), int(h.shape[1])
        if s.rows > 1 and s.length > 1:
            # upstream splits left-padded multi-row prefill per row; the runner
            # prefills one request at a time, so this never reaches here
            raise RuntimeError("fused prefill: multi-row prefill segment")
        cache = s.cache
        view = cache
        fa = cache[model.fa_idx]
        if s.rows == 1 and fa is not None and q35._is_single_row_batch_cache(fa):
            # upstream's single-row shortcut: run on the row's own caches and
            # merge back afterwards
            view = [
                None
                if c is None
                else (
                    q35._extract_row_cache(c, 0)
                    if q35._is_single_row_batch_cache(c)
                    else c
                )
                for c in cache
            ]
            merges.append((cache, view))
        s.view = view
        s.fa_mask = q35._create_qwen3_5_attention_mask(h, view[model.fa_idx])
        s.ssm_mask = q35._create_qwen3_5_ssm_mask(h, view[model.ssm_idx])
        pads = (
            getattr(view[model.fa_idx], "_qwen3_5_decode_left_padding", None)
            if isinstance(s.fa_mask, str) and s.fa_mask == "left_padded_decode"
            else None
        )
        q35._set_qwen3_5_decode_left_padding(view, layers, pads)
        s.position_embeddings = None
        if s.position_ids is not None:
            for layer in layers:
                if not layer.is_linear:
                    if not layer.self_attn.rotary_emb.fused_apply:
                        s.position_embeddings = layer.self_attn.rotary_emb(
                            h, s.position_ids
                        )
                    break
        hs.append(h)

    x = _pack(hs)
    width = x.shape[-1]
    for i, layer in enumerate(layers):
        xn = layer.input_layernorm(x)
        parts = _split(xn, segments)
        name = "linear_attn" if layer.is_linear else "self_attn"
        r = _mixer(
            getattr(layer, name), name, i, layer.is_linear, xn, parts, segments, width
        )
        h = x + r
        x = h + layer.mlp(layer.post_attention_layernorm(h))
    outs = _split(model.norm(x), segments)

    for cache, view in merges:
        for i, entry in enumerate(view):
            if cache[i] is None or entry is None:
                continue
            if hasattr(cache[i].__class__, "merge"):
                cache[i] = cache[i].__class__.merge([entry])
    return outs


__all__ = ["FusedStep", "finish", "forward", "install", "plan", "stats", "supports"]
