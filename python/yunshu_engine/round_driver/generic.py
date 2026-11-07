"""Generic model layer of the round driver: any mlx-vlm language model, on the model's own forward.

The Qwen3.5 layer (``forward.py`` / ``batch.py``) and the Flash-Next layer (``flashnext.py``) pack rows
into their own kernels. This layer packs nothing: every row runs the model's upstream call on the row's
own cache, one row per call, exactly as the stock mlx-vlm generator runs a single request
(``PromptProcessingBatch`` chunks, then the last prompt token on its own with ``logits_to_keep=1``;
decode one token a call). A row's arithmetic is therefore the stock arithmetic whatever else is
running, so a request decodes the same tokens alone and beside others by construction.

What the layer returns to the driver are logits, not hidden states: the driver's ``_draw`` reads them
through ``RoundDriver._logits``, which is the identity here. A prompt span yields ``[1, V]`` only when
it ends the prompt (the first token's distribution), nothing otherwise; a decode step yields ``[N, V]``,
one row per decoding row. No drafter: ``RoundDriver.can_draft`` is False, so every row decodes one
token a round through its own processors, guide and sampler.

Per-row model state: mRoPE models keep ``_rope_deltas`` / ``_position_ids`` on the language model
between calls (upstream resets it per batch). Rows interleave here, so each call swaps the row's saved
values in and out around the upstream call.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

# attributes an mRoPE language model carries from one call to the next (mrope.clear_rope_state)
STATE_ATTRS = ("_position_ids", "_rope_deltas", "_batch_rope_deltas")


def new_cache(lm: Any) -> list:
    """The row's prompt cache, built as the stock generator builds a single row's."""
    from mlx_vlm.models import cache as vlm_cache

    return vlm_cache.make_prompt_cache(lm)


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
    return out.logits if hasattr(out, "logits") else out


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
    """The decoding rows of a generic target: one upstream call per row and step, each on its own cache.

    Same interface as ``DecodeBatch`` / ``FlashNextBatch`` (``join`` / ``leave`` / ``arrays`` /
    ``forward`` / ``commit``). The window of every row is its one pending token (no drafts)."""

    def __init__(self, language_model: Any, kv_format: str = "bf16"):
        if kv_format != "bf16":
            raise ValueError(
                f"YUNSHU_KV_PRECISION={kv_format} is not available for this model on the round driver"
            )
        self.lm = language_model
        self.kv_format = kv_format
        self.rows: list = []

    def join(self, rows: list) -> None:
        for row in rows:
            row.slot = id(row)  # the caches stay on the row
            self.rows.append(row)

    def leave(self, gone: list) -> None:
        ids = {id(r) for r in gone}
        for r in gone:
            r.slot = None
        self.rows = [r for r in self.rows if id(r) not in ids]

    def arrays(self) -> list:
        out: list = []
        for r in self.rows:
            out += cache_arrays(r.cache)
        return out

    def forward(self, lens: list[int], tokens: mx.array) -> mx.array:
        """Every row's pending token through the model; returns logits ``[N, V]`` in row order."""
        if len(lens) != len(self.rows) or any(int(n) != 1 for n in lens):
            raise ValueError("generic round driver decodes one token per row and step")
        parts = []
        for i, row in enumerate(self.rows):
            kwargs = {}
            if row.state.rope_deltas is not None:
                kwargs["rope_deltas"] = row.state.rope_deltas
            out = _call(
                self.lm, row, tokens[i : i + 1][None], cache=row.cache, **kwargs
            )
            parts.append(_logits_of(out)[:, -1, :])
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)

    def commit(self, used: list[int]) -> None:
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
