# Upstream (inspired): Blaizzy/mlx-vlm (MIT) mlx_vlm/generate/ar.py @ 92b31ad308b651a7c1e22d0c58bcdc35c3807750
"""Generic model layer of the round driver: any mlx-vlm language model, on the model's own forward.

The Qwen3.5 layer (``forward.py`` / ``batch.py``) and the Flash-Next layer (``flashnext.py``) pack rows
into their own kernels. This layer has none: prefill runs the model's upstream call per row on the row's
own cache, exactly as the stock mlx-vlm generator prefills a single request (``PromptProcessingBatch``
chunks, then the last prompt token on its own with ``logits_to_keep=1``); decode rows are batched through
the upstream batched forward (``BatchKVCache`` / left padding), the same call and arithmetic as the stock
``GenerationBatch``. A family's outputs are what the stock runner produced for it (no identity change).

What the layer returns to the driver are logits, not hidden states: the driver's ``_draw`` reads them
through ``RoundDriver._logits``, which is the identity here. A prompt span yields ``[1, V]`` only when
it ends the prompt (the first token's distribution), nothing otherwise; a decode step yields ``[N, V]``,
one row per decoding window position. No model drafter: ``RoundDriver.can_draft`` is False (the
model-free n-gram / suffix drafter, ``ngram.py``, is a per-request opt-in on a lone row). The user
option ``YUNSHU_JUMP_FORWARD`` (default off) batches singleton grammar continuations; T>1 on
upstream kernels may round differently from T=1. Ordinary rows still decode one token a round.

Per-row model state: mRoPE models keep ``_rope_deltas`` / ``_position_ids`` on the language model
between calls (upstream resets it per batch). Rows interleave here, so each call swaps the row's saved
values in and out around the upstream call.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx

# attributes an mRoPE language model carries from one call to the next (mrope.clear_rope_state)
STATE_ATTRS = ("_position_ids", "_rope_deltas", "_batch_rope_deltas")


def new_cache(lm: Any) -> list:
    """The row's prompt cache, built as the stock generator builds a single row's."""
    from mlx_vlm.models import cache as vlm_cache

    return cast(list[Any], vlm_cache.make_prompt_cache(lm))


class PromptFeatures:
    """A prompt's model inputs: the embeddings of every prompt token (``get_input_embeddings`` of the
    runner, text prompts too) and the model's other prompt kwargs (mRoPE positions, deepstack features,
    per-layer inputs, masks, ...). Sequence-aligned kwargs are cut with the span, as the stock prompt
    batch cuts them (``_slice_sequence_aligned_prompt_kwarg``); everything else is passed whole."""

    def __init__(self, features: dict, length: int):
        from mlx_vlm.generate import ar

        private = set(getattr(ar, "APC_PRIVATE_PROMPT_KEYS", ()))
        kwargs = {
            k: v
            for k, v in features.items()
            if v is not None and k not in private and not k.startswith("_apc")
        }
        embeds = kwargs.pop("inputs_embeds", None)
        if embeds is None:
            raise ValueError(
                "generic round driver row: prompt features lack inputs_embeds"
            )
        if embeds.shape[1] != length:
            raise ValueError(
                f"generic round driver row: {embeds.shape[1]} embedding rows for a {length}-token prompt"
            )
        self.length = length
        self.embeds = embeds  # [1, L, D]
        self._slice = ar._slice_sequence_aligned_prompt_kwarg
        self.aware = [
            k
            for k, v in kwargs.items()
            if isinstance(v, mx.array)
            and ar._prompt_kwarg_batch_size(k, v) == 1
            and ar._is_sequence_aligned_prompt_kwarg(k, v, length)
        ]
        self.kwargs = kwargs

    def span(self, start: int, end: int) -> dict:
        """The upstream call's keyword arguments for prompt positions ``[start, end)``."""
        out = dict(self.kwargs)
        for k in self.aware:
            out[k] = self._slice(k, out[k], start=start, stop=end)
        return out

    @property
    def rope_deltas(self) -> mx.array | None:
        """Explicit decode offsets carried by the prompt (media prompts); None: read from the model."""
        v = self.kwargs.get("rope_deltas")
        return v if isinstance(v, mx.array) else None


class RowState:
    """What of the language model's own state one row owns, saved between calls."""

    def __init__(self) -> None:
        self.saved: dict[str, Any] = {}
        self.rope_deltas: mx.array | None = None  # the decode kwarg (after the prompt)


def _call(lm: Any, row: Any, *args, **kwargs):
    """The upstream call with this row's model state swapped in and saved back after."""
    state: RowState = row.state
    for name in STATE_ATTRS:
        if hasattr(lm, name):
            setattr(lm, name, state.saved.get(name))
    try:
        return lm(*args, **kwargs)
    finally:
        for name in STATE_ATTRS:
            if hasattr(lm, name):
                state.saved[name] = getattr(lm, name)


def _logits_of(out: Any) -> mx.array:
    return cast(mx.array, out.logits if hasattr(out, "logits") else out)


def prefill(lm: Any, row: Any, start: int, end: int) -> mx.array | None:
    """Prompt positions ``[start, end)`` of ``row`` through the model; ``[1, V]`` logits of the last
    prompt position when the span ends the prompt, else None.

    The span ending the prompt runs as the stock generator runs it: everything but the last token as a
    chunk, then the last token alone asking for one row of logits (``logits_to_keep``). The row's decode
    offsets are captured with the stock helper after that call."""
    feats: PromptFeatures = row.features
    if start > 0 and not row.state.saved and hasattr(lm, "_rope_deltas"):
        # A fresh APC row has K/V but no model-local mRoPE state. Upstream
        # treats a missing delta as a new prompt and numbers only the suffix
        # from zero. Text has delta zero; explicit media deltas stay per row.
        delta = feats.rope_deltas
        if delta is not None:
            row.state.saved["_rope_deltas"] = delta
        elif not getattr(row.req, "media", False):
            row.state.saved["_rope_deltas"] = mx.zeros((1, 1), dtype=mx.int32)
    n = len(row.req.ids)
    final = end == n
    stop = end - 1 if final else end
    if stop > start:
        out = _call(
            lm,
            row,
            mx.array(row.req.ids[start:stop], dtype=mx.int32)[None],
            cache=row.cache,
            inputs_embeds=feats.embeds[:, start:stop],
            n_to_process=stop - start,
            **feats.span(start, stop),
        )
        if getattr(row.req, "prompt_logprobs", None) is not None:
            from .prompt_logprobs import score_span

            score_span(row, start, _logits_of(out)[0])
        del out  # the chunk's logits are never read; only the cache it advanced matters
    if not final:
        return None
    from mlx_vlm.generate import ar

    out = _call(
        lm,
        row,
        mx.array(row.req.ids[n - 1 : n], dtype=mx.int32)[None],
        cache=row.cache,
        inputs_embeds=feats.embeds[:, n - 1 : n],
        logits_to_keep=1,
        **feats.span(n - 1, n),
    )
    # the decode offsets: the prompt's own, else what the call just recorded on the model
    row.state.rope_deltas = (
        ar.PromptProcessingBatch._capture_rope_deltas_from_prompt_kwargs(
            feats.kwargs, lm, 1
        )
    )
    return _logits_of(out)[:, -1, :]


def cache_arrays(cache: list) -> list:
    """The buffers a row's cache keeps between steps: KV buffers (the raw backing, not the sliced
    ``state`` views, which an evaluation would copy), recurrent / conv state, and the sub-caches of a
    ``CacheList``."""
    out: list = []
    for c in cache:
        if getattr(c, "keys", None) is not None:
            out += [c.keys, c.values]
        elif getattr(c, "caches", None) is not None:
            out += cache_arrays(list(c.caches))
        else:
            out += [a for a in getattr(c, "cache", ()) if isinstance(a, mx.array)]
    return out


def _trimmable(cache: Any, extra: int) -> bool:
    """``cache`` can drop its last ``extra`` positions after they are appended: ``is_trimmable`` now
    and, for a rotating window, still after ``extra`` more positions (it stops trimming once wrapped)."""
    subs = getattr(cache, "caches", None)
    if subs is not None:
        return all(_trimmable(c, extra) for c in subs)
    if not cache.is_trimmable():
        return False
    size = getattr(cache, "max_size", None)
    if size is None:
        return True
    off = getattr(cache, "offset", 0)
    off = int(off.max().item()) if hasattr(off, "max") else int(off)
    return off + extra <= int(size)


class GenericBatch:
    """The decoding rows of a generic target. Each row decodes on its own cache, one upstream call per
    row per step: exactly the call a lone request makes, so a row's output does not depend on which
    other rows share the round (solo == concurrent, greedy and seeded).

    A batched forward (left-padded ``BatchKVCache`` rows through ONE upstream call, the stock
    generator's way) runs every projection at M=rows and attention over padding, whose kernels round
    differently from the M=1 decode GEMV; the identity of a row beside others was lost to that
    (measured on Gemma-4: first difference at the per-layer-input projection, then every layer).
    Decode is weight-bandwidth bound, so the per-row calls cost about what the batch's matmuls did
    in launches, not in bytes.  Same interface as ``DecodeBatch`` / ``FlashNextBatch`` (``join`` /
    ``leave`` / ``arrays`` / ``forward`` / ``commit``). The window of a row starts with its pending
    token, optionally followed by grammar-forced tokens (no drafter).

    ``row.cache`` is always the row's own cache; ``restore`` has nothing to cut out."""

    def __init__(self, language_model: Any, kv_format: str = "bf16"):
        if kv_format != "bf16":
            raise ValueError(
                f"YUNSHU_KV_PRECISION={kv_format} is not available for this model on the round driver"
            )
        self.lm = language_model
        self.kv_format = kv_format
        self.rows: list = []
        self._lens: list[int] = []
        # a hidden-state drafter (Gemma-4 assistant) reads the last layer's output of a lone row's window
        self.want_hidden = False
        self.hidden: mx.array | None = (
            None  # (1, T, D) normed hidden of the last forward
        )

    def join(self, rows: list) -> None:
        for row in rows:
            row.slot = id(row)
            self.rows.append(row)

    def leave(self, gone: list) -> None:
        ids = {id(r) for r in gone}
        for r in gone:
            r.slot = None
        self.rows = [r for r in self.rows if id(r) not in ids]

    def cut_out(self, row: Any) -> None:
        """Rows keep their own caches; nothing to give back."""

    @property
    def cache(self) -> list | None:
        """Every decoding row's cache (None with no rows)."""
        if not self.rows:
            return None
        return [c for r in self.rows for c in r.cache]

    def arrays(self) -> list:
        return [a for r in self.rows for a in cache_arrays(r.cache)]

    def forward(self, lens: list[int], tokens: mx.array) -> mx.array:
        """One upstream call per row over its decode window; logits in row/window order."""
        if len(lens) != len(self.rows) or any(int(n) < 1 for n in lens):
            raise ValueError("generic round driver needs a nonempty window per row")
        self._lens = list(lens)
        self.hidden = None
        hidden = (
            self.want_hidden
            and len(lens) == 1
            and hasattr(self.lm, "speculative_draft_hidden")
        )
        outs, at = [], 0
        for row, n in zip(self.rows, lens, strict=True):
            kwargs = {}
            if row.state.rope_deltas is not None:
                kwargs["rope_deltas"] = row.state.rope_deltas
            if hidden:
                out = self.lm(
                    tokens[None, at : at + n],
                    cache=row.cache,
                    return_hidden=True,
                    **kwargs,
                )
                sink = getattr(out, "hidden_states", None)
                if sink:
                    self.hidden = self.lm.speculative_draft_hidden(sink[-1])
            else:
                out = self.lm(tokens[None, at : at + n], cache=row.cache, **kwargs)
            logits = _logits_of(out)
            outs.append(logits[:, -1, :] if n == 1 else logits[0, :n])
            at += n
        return outs[0] if len(outs) == 1 else mx.concatenate(outs)

    def can_trim(self, extra: int = 0) -> bool:
        """Every cache of the decode batch can cut ``extra`` more window positions after appending them
        (KV caches; a rotating cache only while its window has not wrapped; not recurrent state)."""
        return bool(self.rows) and all(
            _trimmable(c, extra) for r in self.rows for c in r.cache
        )

    def commit(self, used: list[int]) -> None:
        for r, u, n in zip(self.rows, used, self._lens, strict=True):
            if u < n:
                # a processor can reject a forced token; keep only the verified prefix
                for c in r.cache:
                    if not c.is_trimmable():
                        raise RuntimeError(
                            "generic window cache cannot trim a rejected prefix"
                        )
                    c.trim(n - u)
            r.n += int(u)


__all__ = [
    "GenericBatch",
    "PromptFeatures",
    "RowState",
    "cache_arrays",
    "new_cache",
    "prefill",
]
