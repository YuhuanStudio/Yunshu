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


class MTPHead:
    """Upstream ``Qwen3_5MTPDraftModel`` weights, driven per row."""

    def __init__(self, drafter: Any, language_model: Any, logits_fn):
        self.drafter = drafter
        self.lm = language_model
        self.embed = language_model.model.embed_tokens
        self.logits = logits_fn

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

    # ── row bookkeeping ──────────────────────────────────────────────────
    @staticmethod
    def _trim(cache: list, n: int) -> None:
        if n > 0:
            for c in cache:
                c.trim(n)

    def absorb(self, rows: list[tuple[Any, list[int], mx.array]]) -> list[mx.array]:
        """``rows``: (row state with ``mtp_cache`` / ``mtp_temp``, next tokens,
        target hidden [n, D]) — position ``p`` pairs token ``p + 1`` with hidden
        ``p``. Drops each row's temporary chain entries first; returns each
        row's head output at its last absorbed position [D]."""
        if not rows:
            return []
        for row, _, _ in rows:
            self._trim(row.mtp_cache, row.mtp_temp)
            row.mtp_temp = 0
        outs = self._forward(
            [r.mtp_cache for r, _, _ in rows],
            [mx.array(t, dtype=mx.int32) for _, t, _ in rows],
            [h for _, _, h in rows],
        )
        return [o[-1] for o in outs]

    def draft(self, rows: list[tuple[Any, mx.array, int]]) -> list[list[int]]:
        """``rows``: (row state, head output at the last absorbed position [D],
        depth). Returns each row's ``depth`` greedy drafts."""
        live = [(r, h, d) for r, h, d in rows if d > 0]
        drafts: dict[int, list] = {id(r): [] for r, _, _ in rows}
        if not live:
            return [[] for _ in rows]
        hidden = mx.stack([h for _, h, _ in live])
        tok = mx.argmax(self.logits(hidden), axis=-1).astype(mx.int32)
        steps = [tok]
        depth = max(d for _, _, d in live)
        active = list(range(len(live)))
        prev_h = hidden
        for step in range(1, depth):
            active = [i for i in active if live[i][2] > step]
            if not active:
                break
            idx = mx.array(active, dtype=mx.int32)
            outs = self._forward(
                [live[i][0].mtp_cache for i in active],
                [steps[-1][idx][j : j + 1] for j in range(len(active))],
                [prev_h[idx][j : j + 1] for j in range(len(active))],
            )
            for i in active:
                live[i][0].mtp_temp += 1
            h_new = mx.concatenate(outs, axis=0)
            t_new = mx.argmax(self.logits(h_new), axis=-1).astype(mx.int32)
            full_t = mx.zeros_like(steps[-1])
            full_h = mx.zeros_like(prev_h)
            full_t[idx] = t_new
            full_h[idx] = h_new
            steps.append(full_t)
            prev_h = full_h
        mx.eval(*steps)
        table = [s.tolist() for s in steps]
        for i, (r, _, d) in enumerate(live):
            drafts[id(r)] = [table[j][i] for j in range(d)]
        return [drafts[id(r)] for r, _, _ in rows]


__all__ = ["MTPHead"]
