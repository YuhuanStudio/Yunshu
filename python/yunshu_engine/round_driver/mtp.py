"""The checkpoint's MTP head as a per-row drafter for the round driver.

Qwen3.5's MTP head is one decoder layer that, at target position ``p``, reads
``(embed(token[p + 1]), target_hidden[p])`` and predicts ``token[p + 2]``. It
has its own KV over every earlier position, so a row can only draft if that KV
covers its whole history. Here every draftable row *absorbs* each committed
position right after its step — prompt chunks as they prefill, then each
step's kept window positions — always with the target's own hidden states.
The head's cache is therefore the same whether or not (and how deep) the row
drafted, and any row can start drafting at any step.

Drafting chains ``depth`` tokens: the first from the head's output at the last
absorbed position, each next one by feeding the previous draft with the head's
own hidden (upstream's draft loop). Those chain entries are temporary: the next
``absorb`` trims them before appending the committed positions. Rows are packed
into one head forward per chain depth (projections shared, attention per row).
Drafts only decide how much a step can commit, never what: the target verifies.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from .batch import KVPlan, Slots, attend


class MTPHead:
    """Upstream ``Qwen3_5MTPDraftModel`` weights, driven per row."""

    def __init__(self, drafter: Any, language_model: Any, logits_fn):
        self.drafter = drafter
        self.lm = language_model
        self.embed = language_model.model.embed_tokens
        self.logits = logits_fn
        self.slots = Slots(len(drafter.layers))
        self.rows: list = []  # rows in the slot buffers

    def make_cache(self) -> list:
        from mlx_vlm.models.cache import KVCache

        return [KVCache() for _ in self.drafter.layers]

    # ── one packed head forward over rows ────────────────────────────────
    def _forward(
        self, caches: list[list], tokens: list[mx.array], hidden: list[mx.array]
    ) -> list[mx.array]:
        """Each row's ``tokens`` [n] with target-side ``hidden`` [n, D] through
        the head; returns each row's head output [n, D] (after its norm)."""
        d = self.drafter
        lengths = [int(t.shape[0]) for t in tokens]
        emb = self.embed(mx.concatenate(tokens).astype(mx.int32))
        hid = mx.concatenate(hidden, axis=0).astype(emb.dtype)
        h = mx.concatenate(
            [d.pre_fc_norm_embedding(emb), d.pre_fc_norm_hidden(hid)], axis=-1
        )
        x = d.fc(h)[None]
        for li, layer in enumerate(d.layers):
            xn = layer.input_layernorm(x)
            at = layer.self_attn
            q, k, v = at.q_proj(xn), at.k_proj(xn), at.v_proj(xn)
            parts, s = [], 0
            for cache, n in zip(caches, lengths, strict=True):
                queries, keys, values, gate, _ = at._prepare_projected_qkv(
                    q[:, s : s + n],
                    k[:, s : s + n],
                    v[:, s : s + n],
                    cache[li],
                    None,
                    None,
                    None,
                )
                out = mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=at.scale,
                    mask="causal" if n > 1 else None,
                )
                parts.append(
                    out.transpose(0, 2, 1, 3).reshape(1, n, -1) * mx.sigmoid(gate)
                )
                s += n
            r = at.o_proj(mx.concatenate(parts, axis=1))
            hh = x + r
            x = hh + layer.mlp(layer.post_attention_layernorm(hh))
        out = d.norm(x)[0]
        res, s = [], 0
        for n in lengths:
            res.append(out[s : s + n])
            s += n
        return res

    # ── prompt chunks: per row, on the row's own KV ─────────────────────
    def absorb_prompt(
        self, rows: list[tuple[Any, list[int], mx.array]]
    ) -> list[mx.array]:
        """``rows``: (row state with ``mtp_cache``, next tokens, target hidden
        [n, D]) of prompt chunks — position ``p`` pairs token ``p + 1`` with
        hidden ``p``. Returns each row's head output at its last absorbed
        position [D]."""
        if not rows:
            return []
        outs = self._forward(
            [r.mtp_cache for r, _, _ in rows],
            [mx.array(t, dtype=mx.int32) for _, t, _ in rows],
            [h for _, _, h in rows],
        )
        return [o[-1] for o in outs]

    # ── decoding rows: shared slot buffers, one launch per layer ─────────
    def join(self, rows: list) -> None:
        """Move rows whose prompt is absorbed (``mtp_cache`` holds ``hn``
        positions) into the slot buffers."""
        slots = self.slots
        for row in rows:
            row.hslot = slots.alloc()
            row.hn = int(row.mtp_cache[0].offset)
        slots.reserve(max(r.hn for r in rows) + 16)
        for row in rows:
            for li, c in enumerate(row.mtp_cache):
                slots.write_prefix(li, row.hslot, c.keys, c.values, row.hn)
            row.mtp_cache = None
        self.rows.extend(rows)

    def leave(self, gone: list) -> None:
        ids = {id(r) for r in gone}
        for r in self.rows:
            if id(r) in ids:
                self.slots.release(r.hslot)
                r.hslot = None
        self.rows = [r for r in self.rows if id(r) not in ids]
        if not self.rows:
            self.slots = Slots(len(self.drafter.layers))

    def _run(self, rows: list, T: int, emb: mx.array, hid: mx.array) -> mx.array:
        """Head forward over the rows' ``T`` padded tokens (``emb`` / ``hid``
        [B * T, D]); keys go to positions ``hn .. hn + T - 1``. [B, T, D]."""
        d = self.drafter
        self.slots.reserve(max(r.hn for r in rows) + T + 1)
        plan = KVPlan.make([r.hslot for r in rows], [r.hn for r in rows], T)
        h = mx.concatenate(
            [d.pre_fc_norm_embedding(emb), d.pre_fc_norm_hidden(hid.astype(emb.dtype))],
            axis=-1,
        )
        x = d.fc(h).reshape(len(rows), T, -1)
        for li, layer in enumerate(d.layers):
            xn = layer.input_layernorm(x)
            hh = x + attend(layer.self_attn, xn, self.slots, li, plan)
            x = hh + layer.mlp(layer.post_attention_layernorm(hh))
        return d.norm(x)

    def absorb(
        self,
        rows: list,
        tokens: list[list[int]],
        hidden: mx.array,
        where: list[list[int]],
    ) -> mx.array:
        """Decoding rows absorb the positions they kept: ``tokens[b]`` (next
        tokens) pair with ``hidden[where[b]]`` (the target's hidden states).
        Returns each row's head output at its last absorbed position [B, D]."""
        T = max(len(t) for t in tokens)
        tok = np.array([t + [t[-1]] * (T - len(t)) for t in tokens], dtype=np.int32)
        idx = np.array([w + [w[-1]] * (T - len(w)) for w in where], dtype=np.int32)
        out = self._run(
            rows,
            T,
            self.embed(mx.array(tok.reshape(-1))),
            hidden[mx.array(idx.reshape(-1))],
        )
        last = out[mx.arange(len(rows)), mx.array([len(t) - 1 for t in tokens])]
        for r, t in zip(rows, tokens, strict=True):
            r.hn += len(t)
        return last

    def draft(self, rows: list, heads: mx.array, depths: list[int]) -> list[list[int]]:
        """``heads`` [B, D]: each row's head output at its last absorbed
        position. Returns each row's ``depth`` greedy drafts (chain entries are
        written past ``hn`` and overwritten by the next absorb)."""
        live = [b for b, d in enumerate(depths) if d > 0]
        if not live:
            return [[] for _ in rows]
        sel = mx.array(live, dtype=mx.int32)
        lrows = [rows[b] for b in live]
        hid = heads[sel]
        tok = mx.argmax(self.logits(hid), axis=-1).astype(mx.int32)
        steps = [tok]
        for j in range(1, max(depths[b] for b in live)):
            saved = [r.hn for r in lrows]
            for r in lrows:
                r.hn += j - 1
            hid = self._run(lrows, 1, self.embed(tok), hid)[:, 0]
            for r, n in zip(lrows, saved, strict=True):
                r.hn = n
            tok = mx.argmax(self.logits(hid), axis=-1).astype(mx.int32)
            steps.append(tok)
        mx.eval(*steps)
        table = [s.tolist() for s in steps]
        out: list[list[int]] = [[] for _ in rows]
        for i, b in enumerate(live):
            out[b] = [table[j][i] for j in range(depths[b])]
        return out


__all__ = ["MTPHead"]
