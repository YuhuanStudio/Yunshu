"""Reduce prompt readouts immediately; retain only scalar scores and top-k entries."""

from __future__ import annotations

from typing import cast

import mlx.core as mx

from ..blocks.head import top_logprobs_rows


def score_span(row, start: int, logits: mx.array) -> None:
    n = min(logits.shape[0], len(row.req.ids) - start - 1)
    if n <= 0:
        return
    logits = logits[:n].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    targets = row.req.ids[start + 1 : start + n + 1]
    values = mx.take_along_axis(lp, mx.array(targets)[:, None], axis=-1)
    k = int(row.req.prompt_logprobs or 0)
    ids, tops = top_logprobs_rows(lp, k)
    vals = cast(list[float], values[:, 0].tolist())
    for j, token in enumerate(targets):
        entry = {"token_id": token, "logprob": vals[j]}
        if k > 0:
            entry["top_logprobs"] = [
                {"token_id": t, "logprob": v}
                for t, v in zip(ids[j], tops[j], strict=True)
            ]
        row.prompt_scores.append(entry)
