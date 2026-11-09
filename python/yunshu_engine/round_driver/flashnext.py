# Upstream (inspired): Blaizzy/mlx-vlm (MIT) models/qwen4_exp/language.py, speculative/drafters/mtp_base.py, speculative/cache_state.py @ 92b31ad3
"""Qwen3.8-Flash-Next (Qwen4: HC + QSA + GDN + MoE + PLE) rows on the round driver.

The driver keeps everything that is not the model: span planning, APC, replay, sampling,
drafting policy, cancel, events. This module is the model side for the Flash-Next layout, behind the
same interface the Qwen3.5 side has (``forward.lane_forward`` / ``batch.DecodeBatch`` /
``mtp.MTPHead``):

- ``prefill``: one prompt span through the model's own forward on the row's caches; returns the
  pre-mixer hyper-connection (HC) state ``[T, hc * D]``. Logits and the MTP head both read that state.
- ``logits``: HC state -> final mixer -> LM head (the speculative readout, row-invariant).
- ``FlashNextBatch``: the decoding rows. Every row keeps its own caches (a ``QSAKVCache`` per attention
  layer carrying the QSA indexer keys and block summaries, an ``ArraysCache`` per GDN layer carrying the
  recurrent state, the conv window and, on PLE layers, the n-gram token history and short-conv window), so
  per-row state is isolated by construction. A step packs the rows' windows (1 or 2 positions) into one
  forward whose weight-bearing ops (fused HC kernels, router + routed/shared experts) run once over all
  rows, while PLE, GDN and QSA run per row on the row's cache; every row's result is the bits of the same
  window verified alone (``speculative_verify_hidden``), so solo == concurrent and spec on == off.
  Rows commit through upstream's speculative cache transactions.
- ``FlashNextHead``: the checkpoint's native MTP head (HC state in, HC state out) per row on a
  per-row ``QSAKVCache``; positions are the head cache offsets.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import mlx.core as mx

from ..qwen4_family import is_qwen4_type, qwen4_first_module
from .verify import DRIVER_MAX_ROWS

logger = logging.getLogger(__name__)

MAX_CHAIN = 2  # positions per row in one packed step: the pending token and one draft
PACK_ROWS = 16  # rows of one fused-kernel pack (kernels/tf_flash_next MAX_ROWS)


def media_of(row: Any) -> tuple[int, bool]:
    """``(rope_delta, multimodal)`` of a driver row: (0, False) for a text prompt."""
    req = row.req
    return int(req.rope_delta), req.position_ids is not None


def decode_caches(driver: Any, row: Any) -> list | None:
    """A finished MTP-head row's target caches followed by its head cache cut to the kept positions: the
    layout of the drafting APC salt, so the next turn can restore the reply. None for other rows."""
    head = getattr(driver, "head", None)
    caches = getattr(head, "caches", {}).get(id(row)) if head is not None else None
    if caches is None:
        return None
    if int(caches[0].offset) < row.n:
        return None  # the head did not absorb every kept position (it lags the target)
    # the head absorbed the last window before its length was known: positions past the kept ones go
    assert head is not None
    head._sync(caches, row.n)
    return list(row.cache) + list(caches)


def is_flash_next(language_model: Any) -> bool:
    """The driver's target is a Qwen4 (Flash-Next) language model."""
    args = getattr(language_model, "args", None)
    return is_qwen4_type(getattr(args, "model_type", None))


def cache_arrays(cache: list) -> list:
    """Every array a row's Flash-Next cache keeps between steps (KV, QSA indexer, GDN / PLE state)."""
    out: list = []
    for c in cache:
        if getattr(c, "keys", None) is not None:
            out += [c.keys, c.values]
            for name in ("index_keys", "index_position_ids", "index_block_keys"):
                a = getattr(c, name, None)
                if isinstance(a, mx.array):
                    out.append(a)
        else:
            out += [a for a in getattr(c, "cache", ()) if isinstance(a, mx.array)]
    return out


# ── prompt spans ─────────────────────────────────────────────────────────
def prefill(
    language_model: Any, seg: Any, *, capture_ids=(), captured: list | None = None
) -> mx.array:
    """One prompt span on the row's caches: the HC state of every span position ``[T, hc * D]``.

    The model's own call: a 1-token span takes the batch-invariant decode forward, longer spans the
    prefill forward (the sparse QSA prefill kernels, chunked GDN), exactly as the speculative lane
    and the batch generator prefilled it. ``skip_logits``: only the last prompt row's logits are read."""
    inputs = seg.tokens[None].astype(mx.int32)
    media = {}
    if seg.embeds is not None:
        media["inputs_embeds"] = seg.embeds[None]
    if seg.position_ids is not None:
        media["position_ids"] = seg.position_ids[:, None, :]
    if media:
        return _prefill_media(
            language_model, inputs, seg.cache, media, capture_ids, captured
        )
    if captured is not None and capture_ids:
        # DFlash2 features: the stream mean of the completed layers (``qwen4_dflash``)
        from .. import qwen4_dflash

        ids = list(capture_ids)
        if inputs.shape[1] == 1:
            out, hidden = qwen4_dflash.capture_forward(
                language_model, inputs, seg.cache, ids, skip_logits=True
            )
            captured.extend(x[0] for x in out.hidden_states)
            return cast(mx.array, hidden[0])
        layers = language_model.model.layers
        sink: list = []
        token = qwen4_dflash._PREFILL_CAPTURE.set(
            (
                {id(layers[i]) for i in ids},
                sink,
                language_model.args.hc_count,
                language_model.args.hidden_size,
            )
        )
        try:
            out = language_model(
                inputs, cache=seg.cache, return_hidden=True, skip_logits=True
            )
        finally:
            qwen4_dflash._PREFILL_CAPTURE.reset(token)
        if len(sink) != len(set(ids)):
            raise ValueError("Qwen4 DFlash capture layer is outside the target")
        captured.extend(x[0] for x in sink)
        return cast(mx.array, out.hidden_states[0][0])
    out = language_model(inputs, cache=seg.cache, return_hidden=True, skip_logits=True)
    return cast(mx.array, out.hidden_states[0][0])


def _prefill_media(
    lm: Any, inputs, cache, media: dict, capture_ids, captured
) -> mx.array:
    """A media prompt's span (prompt embeddings, mRoPE positions) on the row's caches. The model's inner
    call, not ``lm(...)``: that one reads the process-global rope state of the last media prompt, while
    these positions belong to this row."""
    ids = list(capture_ids) if captured is not None else []
    one = inputs.shape[1] == 1
    if one and ids:
        from .. import qwen4_dflash

        out, hidden = qwen4_dflash.capture_forward(
            lm, inputs, cache, ids, skip_logits=True, **media
        )
        captured.extend(x[0] for x in out.hidden_states)
        return cast(mx.array, hidden[0])
    if one:
        out = lm._batch_invariant_decode(inputs, cache=cache, skip_logits=True, **media)
        return cast(mx.array, out.hidden_states[0][0])
    sink: list = []
    token = None
    if ids:
        from .. import qwen4_dflash

        feats: list = []
        layers = lm.model.layers
        token = qwen4_dflash._PREFILL_CAPTURE.set(
            ({id(layers[i]) for i in ids}, feats, lm.args.hc_count, lm.args.hidden_size)
        )
    try:
        lm.model(inputs, cache=cache, capture_layer_ids=[], hidden_sink=sink, **media)
    finally:
        if token is not None:
            from .. import qwen4_dflash

            qwen4_dflash._PREFILL_CAPTURE.reset(token)
    if ids:
        if len(feats) != len(set(ids)):
            raise ValueError("Qwen4 DFlash capture layer is outside the target")
        captured.extend(x[0] for x in feats)
    return cast(mx.array, sink[0][0])


def logits(language_model: Any, hidden: mx.array) -> mx.array:
    """Logits ``[R, V]`` of HC states ``[R, hc * D]``: the final mixer, then the row-invariant head."""
    return cast(
        mx.array, language_model.speculative_logits_from_hidden(hidden[None])[0]
    )


# ── packed decode / verify window ────────────────────────────────────────
def _compute_dtype(lm: Any) -> Any:
    """The dtype the embedding hands the layers (a quantized table dequantizes to its scales' dtype)."""
    table = lm.model.embed_tokens
    scales = getattr(table, "scales", None)
    return scales.dtype if scales is not None else table.weight.dtype


def _fused_ready(lm: Any, dtype: Any) -> bool:
    from ..kernels import tf_hc_forward

    return bool(
        tf_hc_forward._STATE["installed"]
        and dtype == mx.bfloat16
        and mx.default_device() == mx.gpu
        and tf_hc_forward.eligible(lm.model)
    )


def _prepare(lm: Any, tokens: list, caches: list, media: list | None = None):
    """Per row: the position ids and a speculative cache transaction (``_speculative_verify``'s prologue).

    ``media[b] = (rope_delta, multimodal)`` of a media row: positions continue after the prompt's mRoPE
    offset, and a multimodal row's are the three-axis form the model's own verifier uses there."""
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    model = lm.model
    positions, transactions = [], []
    try:
        for b, (t, cache) in enumerate(zip(tokens, caches, strict=True)):
            # queue any lazy prefix-cache writes before the verifier appends to the same buffers
            mx.async_eval([entry.state for entry in cache])
            offset = int(cache[model.fa_idx].offset)
            delta, multimodal = media[b] if media else (0, False)
            pos = (offset + int(delta) + mx.arange(t.shape[1], dtype=mx.int64))[None]
            if multimodal:
                pos = mx.broadcast_to(pos[None], (3, 1, t.shape[1]))
            positions.append(pos)
            transactions.append(start_speculative_cache(cache, t.shape[1]))
    except BaseException:
        for transaction in transactions:
            transaction.abort()
        raise
    return positions, transactions


def fused_chain_forward(
    lm: Any, tokens: list, caches: list, capture_ids=(), media: list | None = None
):
    """The layer loop of ``tf_hc_forward._tf_model`` over several rows' windows.

    HC norm / project and the router + experts run once over all packed rows (the kernels' rows are
    independent, bit-equal at any row count); the sequence mixers and PLE run per row on that row's
    cache with the row's own mask and positions. A one-row call is ``_tf_model`` itself, so a row's
    window gives the bits of ``speculative_verify_hidden`` alone."""
    from ..kernels import tf_hc_forward as tfh
    from ..kernels.tf_flash_next import hc as tf_hc

    ql = qwen4_first_module("language")
    assert ql is not None
    fwd = ql._QWEN4_BATCH_INVARIANT_FORWARD
    model = lm.model
    streams = model.args.hc_count
    widths = [int(t.shape[1]) for t in tokens]
    spans, at = [], 0
    for w in widths:
        spans.append((at, at + w))
        at += w
    total = at
    positions, transactions = _prepare(lm, tokens, caches, media)
    try:
        ids = mx.concatenate(tokens, axis=1)
        hidden = mx.tile(model.embed_tokens(ids), (1, 1, streams))
        width = hidden.shape[-1]
        dims = width // streams
        fa_masks, ssm_masks = [], []
        for (s, e), cache in zip(spans, caches, strict=True):
            fa_masks.append(
                ql._create_qwen3_5_attention_mask(hidden[:, s:e], cache[model.fa_idx])
            )
            ssm_masks.append(
                ql._create_qwen3_5_ssm_mask(hidden[:, s:e], cache[model.ssm_idx])
            )

        def norm(h, pending=None):
            if pending is None:
                return tf_hc.hc_norm(h, streams=streams, write_back="none")
            kind, branch, inject = pending
            return tf_hc.hc_norm(
                h, streams=streams, write_back=kind, branch=branch, inject=inject
            )

        def project(hn, ssp, module):
            plan = tfh._plan(module)
            return tf_hc.hc_project(
                hn,
                ssp,
                plan["down"],
                plan["up"],
                plan["scale"],
                eps=plan["eps"],
                streams=plan["streams"],
                low=plan["low"],
            )

        h = hidden.reshape(total, width)
        pending = None
        capture = set(capture_ids)
        features: list = []
        for index, layer in enumerate(model.layers):
            hn, ssp = norm(h, pending)
            if "ple" in layer:
                stream = hn.reshape(1, total, width)
                masks = ssm_masks if layer.is_linear else fa_masks
                add = mx.concatenate(
                    [
                        fwd._ple(
                            layer.ple,
                            stream[:, s:e],
                            tokens[b],
                            caches[b][index],
                            masks[b],
                        )
                        for b, (s, e) in enumerate(spans)
                    ],
                    axis=1,
                )
                hn = (stream + add).reshape(total, width)
                _, ssp = norm(hn)
            mixed, inject = project(hn, ssp, layer.attn_hyper_connection)
            mixed = mixed.reshape(1, total, dims)
            parts = []
            for b, (s, e) in enumerate(spans):
                if layer.is_linear:
                    parts.append(
                        fwd._gated_delta(
                            layer.linear_attn,
                            mixed[:, s:e],
                            ssm_masks[b],
                            caches[b][index],
                        )
                    )
                else:
                    parts.append(
                        fwd._qsa_attention(
                            layer.self_attn,
                            mixed[:, s:e],
                            caches[b][index],
                            positions[b],
                            fa_masks[b],
                        )
                    )
            branch = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
            hn2, ssp2 = norm(hn, ("plain", (branch.reshape(total, dims),), inject))
            mixed, inject2 = project(hn2, ssp2, layer.mlp_hyper_connection)
            fused = tfh._moe(layer.mlp, mixed)
            if fused is None:
                branch2 = fwd._feed_forward(layer.mlp, mixed.reshape(1, total, dims))
                fused = ("plain", (branch2.reshape(total, dims),))
            h = hn2
            pending = (fused[0], fused[1], inject2)
            if index in capture:
                committed, _ = norm(h, pending)
                features.append(
                    committed.reshape(1, total, streams, dims).mean(axis=-2)
                )
            mx.async_eval(h, *fused[1])
        hn, _ = norm(h, pending)
        hn = hn.reshape(1, total, width)
        out = [
            (hn[:, s:e], transaction)
            for (s, e), transaction in zip(spans, transactions, strict=True)
        ]
        if not capture_ids:
            return out
        return [
            (hid, [f[:, s:e] for f in features], transaction)
            for (hid, transaction), (s, e) in zip(out, spans, strict=True)
        ]
    except BaseException:
        for transaction in transactions:
            transaction.abort()
        raise


def _row_verify(lm: Any, t, cache, medium, capture_ids=()):
    """One row's window with explicit positions (the model's ``_speculative_verify`` with the row's own
    mRoPE offset instead of the process-global one). Returns (hidden, features or [], transaction)."""
    positions, (transaction,) = _prepare(lm, [t], [cache], [medium])
    try:
        if capture_ids:
            from .. import qwen4_dflash

            out, hidden = qwen4_dflash.capture_forward(
                lm,
                t,
                cache,
                list(capture_ids),
                position_ids=positions[0],
                skip_logits=True,
            )
            return hidden, list(out.hidden_states), transaction
        ql = qwen4_first_module("language")
        assert ql is not None
        out = ql._QWEN4_BATCH_INVARIANT_FORWARD(
            lm, t, cache=cache, position_ids=positions[0], skip_logits=True
        )
        return out.hidden_states[-1], [], transaction
    except BaseException:
        transaction.abort()
        raise


def chain_forward_capture(
    lm: Any, tokens: list, caches: list, capture_ids, media: list | None = None
):
    """Windows of any width (a DFlash chain: the pending token and its drafts) of independent rows,
    capturing the DFlash layer features. Returns ``[(HC state [1, w, hc * D], [feature [1, w, D] per
    layer], transaction)]``. The fused loop packs the rows (at most ``PACK_ROWS`` positions in all);
    otherwise every row runs the upstream capture verifier on its own cache, which is the same
    arithmetic as ``_tf_model``'s fallback loop."""
    if not tokens or len(tokens) != len(caches):
        raise ValueError("chain forward needs matching nonempty tokens and caches")
    if len({id(c) for c in caches}) != len(caches):
        raise ValueError("chain forward needs independent caches")
    if _fused_ready(lm, _compute_dtype(lm)):
        if sum(int(t.shape[1]) for t in tokens) > PACK_ROWS:
            raise ValueError("fused pack is limited to PACK_ROWS positions")
        return fused_chain_forward(lm, tokens, caches, capture_ids, media)
    out = []
    try:
        for b, (t, c) in enumerate(zip(tokens, caches, strict=True)):
            if media and media[b] != (0, False):
                hidden, features, transaction = _row_verify(
                    lm, t, c, media[b], capture_ids
                )
            else:
                features, hidden, transaction = lm.speculative_verify_dflash_hidden(
                    t, c, list(capture_ids)
                )
            out.append((hidden, features, transaction))
    except BaseException:
        for got in out:
            got[2].abort()
        raise
    return out


def chain_forward(lm: Any, tokens: list, caches: list, media: list | None = None):
    """Windows ``tokens`` (each ``[1, w]``, ``1 <= w <= PACK_ROWS``) of independent rows through the target.

    Returns ``[(HC state [1, w, hc * D], transaction)]`` in row order; the caller commits each
    transaction with the kept length or aborts it. The fused-kernel loop when the pack's HC weights
    and the device allow it (the arithmetic of ``speculative_verify_hidden``), else the upstream
    layer loop with shared dense projections (``verify.native_chain_forward``); a row's bits are the
    same whichever rows share the call."""
    from .verify import native_chain_forward

    if not tokens or len(tokens) != len(caches):
        raise ValueError("chain forward needs matching nonempty tokens and caches")
    if len({id(c) for c in caches}) != len(caches):
        raise ValueError("chain forward needs independent caches")
    if any(
        t.ndim != 2 or t.shape[0] != 1 or not 1 <= t.shape[1] <= PACK_ROWS
        for t in tokens
    ):
        raise ValueError("chain forward takes 1..PACK_ROWS positions per row")
    if sum(int(t.shape[1]) for t in tokens) > DRIVER_MAX_ROWS:
        raise ValueError("chain forward exceeds the shared node budget")
    if _fused_ready(lm, _compute_dtype(lm)):
        return fused_chain_forward(lm, tokens, caches, (), media)
    if media and any(m != (0, False) for m in media):
        rows = []
        try:
            for b, (t, c) in enumerate(zip(tokens, caches, strict=True)):
                hidden, _, transaction = _row_verify(lm, t, c, media[b])
                rows.append((hidden, transaction))
        except BaseException:
            for _, transaction in rows:
                transaction.abort()
            raise
        return rows
    return native_chain_forward(lm, tokens, caches)


class FlashNextBatch:
    """The decoding rows of one Flash-Next target: every row on its own caches."""

    def __init__(self, language_model: Any, kv_format: str = "bf16"):
        if kv_format != "bf16":
            raise ValueError(
                f"YUNSHU_KV_PRECISION={kv_format} is not available for Flash-Next on the round driver"
            )
        self.lm = language_model
        self.kv_format = kv_format
        self.rows: list = []
        self._last: list | None = None

    def join(self, rows: list) -> None:
        for row in rows:
            row.slot = id(
                row
            )  # the caches stay on the row (``cache`` is its own state)
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

    def packs(self, lens: list[int]) -> list[list[int]]:
        """Row indices per fused pack (at most ``PACK_ROWS`` window positions each)."""
        out: list[list[int]] = [[]]
        used = 0
        for b, w in enumerate(lens):
            if used + w > PACK_ROWS and out[-1]:
                out.append([])
                used = 0
            out[-1].append(b)
            used += w
        return out

    def forward(self, lens: list[int], tokens: mx.array) -> mx.array:
        """Every row's window (``lens[b]`` tokens back to back in ``tokens`` [N] int32, which may still be
        a lazy graph) through the target; returns the HC states ``[N, hc * D]`` in the same order. KV, QSA
        indexer, PLE history and GDN state advance optimistically; ``commit`` keeps ``used[b]`` positions."""
        lens = [int(n) for n in lens]
        if len(lens) != len(self.rows) or max(lens) > PACK_ROWS:
            raise ValueError("Flash-Next decode window is at most PACK_ROWS positions")
        starts, at = [], 0
        for n in lens:
            starts.append(at)
            at += n
        results: list = [None] * len(lens)
        try:
            for pack in self.packs(lens):
                out = chain_forward(
                    self.lm,
                    [tokens[starts[b] : starts[b] + lens[b]][None] for b in pack],
                    [self.rows[b].cache for b in pack],
                    [media_of(self.rows[b]) for b in pack],
                )
                for b, got in zip(pack, out, strict=True):
                    results[b] = got
        except BaseException:
            for got in results:
                if got is not None:
                    got[1].abort()
            raise
        self._last = [got[1] for got in results]
        parts = [got[0][0] for got in results]
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)

    def commit(self, used: list[int]) -> None:
        """Rows keep their first ``used[b]`` window positions (``1 <= used <= len(window)``)."""
        last, self._last = self._last, None
        assert last is not None
        for r, u, transaction in zip(self.rows, used, last, strict=True):
            transaction.commit(int(u))
            r.n += int(u)


# ── the native MTP head ──────────────────────────────────────────────────
class _HeadArrays:
    """``head.slots.arrays()`` for the driver's evaluation lists."""

    def __init__(self, head: FlashNextHead):
        self.head = head

    def arrays(self) -> list:
        out: list = []
        for caches in self.head.caches.values():
            out += cache_arrays(caches)
        return out


class FlashNextHead:
    """The checkpoint's native MTP head (``Qwen4ExpMTPDraftModel``) as a per-row drafter.

    Same lifecycle as ``mtp.MTPHead``: every draftable row absorbs each committed position with the
    target's own HC state, so its head cache is the same whether or not it drafted. The head reads
    ``(embed(token[p + 1]), hc_state[p])`` and returns the next HC state; the draft is the greedy token of
    the mixed state through the target's head. Drafts only decide how much a step can commit."""

    def __init__(self, drafter: Any, language_model: Any, logits_fn):
        self.drafter = drafter
        self.lm = language_model
        self.embed = language_model.model.embed_tokens
        self.logits = logits_fn
        self.vocab = None  # no reduced draft vocabulary on this head
        self.caches: dict[
            int, list
        ] = {}  # id(row) -> its head cache (rows in the decode batch)
        self.rows: list = []
        self.slots = _HeadArrays(self)

    def make_cache(self) -> list:
        return cast(list, self.drafter.make_cache())

    def _mix_logits(self, mixed: mx.array) -> mx.array:
        """Logits of the head's mixed output ``[R, D]``."""
        return cast(mx.array, self.lm.speculative_logits_from_hidden(mixed[None])[0])

    def readout(self, hidden: mx.array, single: bool = False) -> mx.array:
        """Greedy token ids [R] (int32) of head HC states ``hidden`` [R, hc * D]."""
        del single
        mixed = self.drafter.hyper_connection_mixer(hidden[None])[0]
        return mx.argmax(self._mix_logits(mixed), axis=-1).astype(mx.int32)

    def _step(
        self,
        cache: list,
        tokens: mx.array,
        hidden: mx.array,
        delta: int = 0,
        positions: mx.array | None = None,
    ):
        """One row's ``tokens`` [1, T] with HC states ``hidden`` [1, T, hc * D] on ``cache``, positions
        continuing the head cache (plus a media row's rope delta, or the prompt chunk's own temporal mRoPE
        positions ``[1, T]``). Returns (mixed [1, T, D], next HC state [1, T, hc * D])."""
        d = self.drafter
        d._next_position = int(cache[0].offset) + int(delta)
        embedded = self.embed(tokens.astype(mx.int32))
        if positions is None:
            return d._forward_hidden(
                embedded, hidden.astype(embedded.dtype), tokens, cache
            )
        d._position_ids = lambda length=1: positions  # type: ignore[method-assign]
        try:
            return d._forward_hidden(
                embedded, hidden.astype(embedded.dtype), tokens, cache
            )
        finally:
            del d._position_ids

    @staticmethod
    def _sync(cache: list, keep: int) -> None:
        """Cut the head cache back to ``keep`` positions (window / chain entries past what landed)."""
        excess = int(cache[0].offset) - keep
        if excess > 0:
            for c in cache:
                c.trim(excess)

    # prompt spans ------------------------------------------------------
    def absorb_prompt(self, rows: list[tuple]) -> list[mx.array]:
        """``rows``: (row state with ``mtp_cache``, next tokens, HC states [n, hc * D]); returns each
        row's head HC state at its last absorbed position ``[hc * D]``."""
        outs = []
        for row, nxt, hid, *rest in rows:
            pos = rest[0][0][None].astype(mx.int32) if rest else None
            _, state = self._step(
                row.mtp_cache, mx.array([nxt], dtype=mx.int32), hid[None], positions=pos
            )
            outs.append(state[0, -1])
        return outs

    # decoding rows -----------------------------------------------------
    def join(self, rows: list) -> None:
        for row in rows:
            row.hslot = id(row)
            row.hn = int(row.mtp_cache[0].offset)
            self.caches[id(row)] = row.mtp_cache
            row.mtp_cache = None
        self.rows.extend(rows)

    def leave(self, gone: list) -> None:
        ids = {id(r) for r in gone}
        for r in gone:
            self.caches.pop(id(r), None)
            r.hslot = None
        self.rows = [r for r in self.rows if id(r) not in ids]

    def absorb_window(
        self, rows: list, tokens: mx.array, hidden: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Early absorb: the head over every position of the rows' verify windows (``tokens`` [B, T] are
        the target's next tokens, ``hidden`` [B, T, hc * D] its HC states). Keys go to ``hn ..
        hn + T - 1``; once ``used`` is known the caller advances ``hn`` and the rest is overwritten by
        the next window. Returns the head HC states [B, T, hc * D] and the greedy token of each [B, T]."""
        states, greedy = [], []
        for b, row in enumerate(rows):
            cache = self.caches[id(row)]
            self._sync(cache, row.hn)
            mixed, state = self._step(
                cache, tokens[b : b + 1], hidden[b : b + 1], row.req.rope_delta
            )
            states.append(state[0])
            greedy.append(
                mx.argmax(self._mix_logits(mixed[0]), axis=-1).astype(mx.int32)
            )
        return mx.stack(states), mx.stack(greedy)

    def draft(
        self,
        rows: list,
        heads: mx.array,
        depths: list[int],
        first: mx.array | None = None,
    ) -> list[mx.array]:
        """``heads`` [B, hc * D]: each row's head HC state at its last absorbed position (``first`` [B]
        int32: the greedy token already read out of it). Returns each row's greedy drafts as lazy int32
        arrays; chain entries are written past ``hn`` and cut by the next absorb."""
        empty = mx.zeros((0,), dtype=mx.int32)
        live = [b for b, d in enumerate(depths) if d > 0]
        if not live:
            return [empty for _ in rows]
        sel = mx.array(live, dtype=mx.int32)
        hid = heads[sel]
        tok = self.readout(hid) if first is None else first[sel]
        steps = [tok]
        for j in range(1, max(depths[b] for b in live)):
            states, toks = [], []
            for i, b in enumerate(live):
                cache = self.caches[id(rows[b])]
                self._sync(cache, rows[b].hn + j - 1)
                mixed, state = self._step(
                    cache,
                    tok[i : i + 1][None],
                    hid[i : i + 1][None],
                    rows[b].req.rope_delta,
                )
                states.append(state[0, 0])
                toks.append(
                    mx.argmax(self._mix_logits(mixed[0]), axis=-1).astype(mx.int32)
                )
            hid = mx.stack(states)
            tok = mx.concatenate(toks)
            steps.append(tok)
        table = mx.stack(steps, axis=1)
        out = [empty for _ in rows]
        for i, b in enumerate(live):
            out[b] = table[i, : depths[b]]
        return out


__all__ = [
    "FlashNextBatch",
    "FlashNextHead",
    "cache_arrays",
    "chain_forward",
    "is_flash_next",
    "logits",
    "prefill",
]
