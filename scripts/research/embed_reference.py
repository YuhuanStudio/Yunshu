"""Reference text embeddings for last-token embedders (Qwen3-Embedding), computed straight with mlx-lm
following the official recipe: the tokenizer's own post-processor appends <|endoftext|>, the
vector is the final-norm hidden state of the LAST token, L2-normalised; queries carry
``Instruct: {task}\\nQuery:{text}``. One text per forward (identical to the official left-padded
batch with an attention mask). Used by the served embed check as an independent oracle.
The pure helpers are unit-tested on CPU; `reference_embed` needs MLX and runs inside gpuq."""

from __future__ import annotations

import math

DEFAULT_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def format_query(task: str, query: str) -> str:
    """Qwen3-Embedding query format (the model card's get_detailed_instruct)."""
    return f"Instruct: {task}\nQuery:{query}"


def l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def reference_embed(
    model_dir: str, texts: list[str], pooling: str = "last", add_eos: bool = True
) -> list[list[float]]:
    import mlx.core as mx
    from pathlib import Path

    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model, load_tokenizer

    mp = Path(model_dir)
    model = load_model(mp, strict=False)
    model = model[0] if isinstance(model, tuple) else model
    w = {}
    import glob

    for f in sorted(glob.glob(str(mp / "*.safetensors"))):
        w.update(mx.load(f))
    names = {k for k, _ in tree_flatten(model.parameters())}
    # the official checkpoint names the bare backbone: add the wrapper's "model." prefix
    w = {(k if k in names else "model." + k): v for k, v in w.items()}
    missing = names - set(w)
    if missing:
        raise RuntimeError(f"reference load: {len(missing)} parameters without tensors")
    model.load_weights(list(w.items()), strict=False)
    mx.eval(model.parameters())
    tok = load_tokenizer(mp)
    hf = getattr(tok, "_tokenizer", tok)
    eos = hf.convert_tokens_to_ids("<|endoftext|>")
    out = []
    for t in texts:
        ids = list(hf(t, add_special_tokens=False)["input_ids"])
        if add_eos:
            ids.append(eos)
        h = model.model(mx.array([ids]))
        v = h[0, -1, :] if pooling == "last" else mx.mean(h[0], axis=0)
        out.append(l2(v.astype(mx.float32).tolist()))
    return out
