"""Validate the ported Gemma4MTPMaskedEmbedder against REAL drafter weights.

Loads the real centroids / token_ordering / tied lm_head from the
gemma-4-E4B-it-assistant-bf16 checkpoint (159MB, no target model — no OOM
risk) and checks that the MLX sparse-masking math is correct on real data:

  1. Sparse logits for the selected vocab IDs exactly equal the dense logits
     (hidden @ lm_head.T) at those same IDs.
  2. get_top_tokens equals the argmax of the dense logits *restricted to the
     selected set* (the approximation the algorithm is designed to make).
  3. Selected token IDs are a valid subset of the vocabulary.

Run: PYTHONPATH=python uv run python scripts/validate_gemma4_masked_embedder.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx

from yunshu_engine.gemma4_assistant import Gemma4MTPMaskedEmbedder

MODEL = Path("/Volumes/P5Plus/models/gemma-4-E4B-it-assistant-bf16")


def main() -> int:
    if not MODEL.exists():
        print(f"SKIP: model drive not mounted ({MODEL})")
        return 0

    cfg = json.loads((MODEL / "config.json").read_text())
    tcfg = cfg["text_config"]
    hidden = tcfg["hidden_size"]
    vocab = tcfg["vocab_size"]
    num_centroids = cfg["num_centroids"]
    top_k = cfg["centroid_intermediate_top_k"]
    print(f"config: hidden={hidden} vocab={vocab} centroids={num_centroids} top_k={top_k}")

    weights = mx.load(str(MODEL / "model.safetensors"))
    centroids_w = weights["masked_embedding.centroids.weight"].astype(mx.float32)
    token_ordering = weights["masked_embedding.token_ordering"]
    lm_head = weights["model.embed_tokens.weight"].astype(mx.float32)  # tied
    print(
        f"loaded: centroids={centroids_w.shape} "
        f"token_ordering={token_ordering.shape} lm_head={lm_head.shape}"
    )

    emb = Gemma4MTPMaskedEmbedder(hidden, vocab, num_centroids, top_k)
    emb.centroids.weight = centroids_w
    emb.token_ordering = token_ordering.astype(mx.int32)
    mx.eval(emb.centroids.weight, emb.token_ordering)

    # Synthetic hidden states (scale matched to real embeddings).
    mx.random.seed(0)
    t = 4
    h = mx.random.normal((t, hidden)) * float(mx.std(lm_head).item())

    sparse_logits, indices = emb._select_and_score(h, lm_head)
    mx.eval(sparse_logits, indices)
    assert sparse_logits.shape == (t, emb.num_selected), sparse_logits.shape
    print(f"sparse: scored {emb.num_selected}/{vocab} tokens per position "
          f"({vocab / emb.num_selected:.0f}x reduction)")

    # Dense reference over the full real vocabulary.
    dense = h @ lm_head.T  # (t, vocab)
    mx.eval(dense)

    max_abs_err = 0.0
    for ti in range(t):
        idx_row = indices[ti]
        sparse_row = sparse_logits[ti]
        dense_sel = mx.take(dense[ti], idx_row)
        err = float(mx.max(mx.abs(sparse_row - dense_sel)).item())
        max_abs_err = max(max_abs_err, err)
        # token IDs valid
        assert int(mx.min(idx_row).item()) >= 0
        assert int(mx.max(idx_row).item()) < vocab
    print(f"[1] sparse logits match dense at selected IDs: max_abs_err={max_abs_err:.2e}")
    assert max_abs_err < 1e-2, f"sparse/dense mismatch {max_abs_err}"

    # get_top_tokens == argmax over the selected set.
    top = emb.get_top_tokens(h, lm_head)
    mx.eval(top)
    for ti in range(t):
        idx_row = indices[ti]
        dense_sel = mx.take(dense[ti], idx_row)
        expected = int(idx_row[int(mx.argmax(dense_sel).item())].item())
        assert int(top[ti].item()) == expected, (ti, int(top[ti].item()), expected)
    print(f"[2] get_top_tokens == argmax over selected set: OK (tokens={top.tolist()})")

    # How often is the sparse top-token also the GLOBAL dense argmax?
    global_argmax = mx.argmax(dense, axis=-1)
    hits = int(mx.sum(top == global_argmax).item())
    print(f"[3] sparse top == global dense argmax: {hits}/{t} "
          f"(approximation; misses are expected when the true argmax's centroid "
          f"isn't in the top-{top_k})")

    print("\nPASS — Gemma4MTPMaskedEmbedder validated against REAL drafter weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
