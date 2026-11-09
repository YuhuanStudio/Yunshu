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
one row per decoding window position. No drafter: ``RoundDriver.can_draft`` is False. The user
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


class GenericBatch:
    """The decoding rows of a generic target, batched as the stock generator batches them.

    Prefill is per row (own cache, ``prefill`` above). Once a row's first token is out it joins here: a
    lone row keeps decoding on its own cache (upstream's single-row path); a second row turns the caches
    into batch caches (``BatchKVCache`` and the family's own batch classes, left padded, merged with
    upstream's ``_extend_cache``) and every step is ONE upstream forward over all rows, the call and
    arithmetic ``GenerationBatch._step`` makes. Same interface as ``DecodeBatch`` / ``FlashNextBatch``
    (``join`` / ``leave`` / ``arrays`` / ``forward`` / ``commit``). The window of a row starts with its
    pending token, optionally followed by grammar-forced tokens (no drafter).

    While a row sits in a batch cache its ``row.cache`` is None (the batch owns the K/V); ``restore``
    cuts the row's own cache back out (``extract``) when its reply cache is to be stored."""

    def __init__(self, language_model: Any, kv_format: str = "bf16"):
        if kv_format != "bf16":
            raise ValueError(
                f"YUNSHU_KV_PRECISION={kv_format} is not available for this model on the round driver"
            )
        self.lm = language_model
        self.kv_format = kv_format
        self.rows: list = []
        self.cache: list | None = (
            None  # the decoding rows' cache (batch dimension = row order)
        )

    def join(self, rows: list) -> None:
        from mlx_vlm.generate import ar

        for row in rows:
            merged = self.cache is not None
            self.cache = ar._extend_cache(self.cache, row.cache)
            row.slot = id(row)
            self.rows.append(row)
            if merged:
                # the batch cache owns every row's K/V now (a lone row's own cache was copied in)
                for r in self.rows:
                    r.cache = None

    def leave(self, gone: list) -> None:
        ids = {id(r) for r in gone}
        for r in gone:
            r.slot = None
        keep = [i for i, r in enumerate(self.rows) if id(r) not in ids]
        if not keep:
            self.cache = None
        elif len(keep) < len(self.rows):
            keep_arr = mx.array(keep, mx.int32)
            assert self.cache is not None
            for c in self.cache:
                c.filter(keep_arr)
        self.rows = [self.rows[i] for i in keep]

    def restore(self, row: Any) -> None:
        """Give ``row`` its own cache back (cut out of the batch cache) before it leaves."""
        if row.cache is not None or self.cache is None:
            return
        row.cache = [c.extract(self.rows.index(row)) for c in self.cache]

    def arrays(self) -> list:
        return [] if self.cache is None else cache_arrays(self.cache)

    def forward(self, lens: list[int], tokens: mx.array) -> mx.array:
        """One upstream call over decode windows; unpadded logits in row/window order.

        The upstream prepare/finalize protocol rolls trailing padding into each cache's
        left padding and adjusts its position offset. The T=1 call stays exactly the old call.
        """
        if len(lens) != len(self.rows) or any(int(n) < 1 for n in lens):
            raise ValueError("generic round driver needs a nonempty window per row")
        self._lens = list(lens)
        width = max(lens)
        pads = [width - n for n in lens]
        if width > 1:
            parts, at = [], 0
            for n, pad in zip(lens, pads, strict=True):
                parts.append(mx.pad(tokens[at : at + n], (0, pad)))
                at += n
            inputs = mx.stack(parts)
            if any(pads):
                for c in self.cache or ():
                    c.prepare(right_padding=pads, lengths=lens)
        else:
            inputs = tokens[:, None]
        deltas = [r.state.rope_deltas for r in self.rows]
        kwargs = {}
        if any(d is not None for d in deltas):
            if any(d is None for d in deltas):
                raise RuntimeError(
                    "generic round driver: rows with and without mRoPE offsets cannot share a batch"
                )
            kwargs["rope_deltas"] = (
                deltas[0] if len(deltas) == 1 else mx.concatenate(deltas, axis=0)
            )
        out = self.lm(inputs, cache=self.cache, **kwargs)
        if width == 1:
            return _logits_of(out)[:, -1, :]
        if any(pads):
            for c in self.cache or ():
                c.finalize()
        logits = _logits_of(out)
        return mx.concatenate([logits[b, :n] for b, n in enumerate(lens)])

    def commit(self, used: list[int]) -> None:
        if any(u != n for u, n in zip(used, self._lens, strict=True)):
            # A processor can reject a forced token; keep only the verified prefix.
            # Extracting removes each row's padding before trimming and re-merging.
            from mlx_vlm.generate import ar

            assert self.cache is not None
            merged = None
            for b, (r, u, n) in enumerate(
                zip(self.rows, used, self._lens, strict=True)
            ):
                cache = (
                    r.cache
                    if r.cache is not None
                    else [c.extract(b) for c in self.cache]
                )
                if n > u:
                    for c in cache:
                        if not c.is_trimmable():
                            raise RuntimeError(
                                "generic forced window cache cannot trim a rejected prefix"
                            )
                        c.trim(n - u)
                merged = ar._extend_cache(merged, cache)
            self.cache = merged
            for r in self.rows:
                r.cache = self.cache if len(self.rows) == 1 else None
        for r, u in zip(self.rows, used, strict=True):
            r.n += int(u)


__all__ = [
    "GenericBatch",
    "PromptFeatures",
    "RowState",
    "cache_arrays",
    "new_cache",
    "prefill",
]
