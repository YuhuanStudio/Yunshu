"""Where does ragged per-row KV cost time next to the stock padded cache?

In-process on one upstream BatchGenerator (small model, e.g. Qwen3.5-0.8B), for
each shape (B=1 short / long, B=8 uniform short, B=8 one long + short rows):

- ``step_ms``: steady plain decode step, stock padded vs ragged bf16 / int8
  (``ragged_kv.enable``: the batch forms through upstream's merge, as served);
- ``convert_ms``: a whole stock batch of the shape to ragged (all attention
  layers, synthetic keys);
- ``first_join_ms``: the lone decoding row (``KVCache``) meeting a 512-token
  row, through upstream ``_extend_cache``;
- ``filter_ms``: dropping one short row;
- ``join_ms``: a 512-token row joining the decoding ragged batch in its place.

    python scripts/research/bench_ragged_overheads.py <model_dir> \
        --output runs/ragged-overheads.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_context_batch import CORPUS, make_prompt  # noqa: E402
from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.ar import BatchGenerator  # noqa: E402

from yunshu_engine.mrope import clear_rope_state  # noqa: E402

SHAPES = {
    "b1_short": [512],
    "b1_8k": [8192],
    "b8_short": [512] * 8,
    "b8_mixed": [8192] + [512] * 7,
}


def _sync_caches(caches):
    mx.eval([s for c in caches for s in getattr(c, "state", []) or [] if s is not None])
    mx.synchronize()


def build(model, processor, tok, lengths, steps):
    corpus = CORPUS.read_text()
    gen = BatchGenerator(
        model.language_model,
        processor,
        max_tokens=steps + 64,
        greedy_sampling=True,
        compute_logprobs=False,
        prefill_batch_size=1,
    )
    clear_rope_state(model)
    for n in lengths:
        x = tok.encode(make_prompt(tok, corpus, n), add_special_tokens=False)
        kw = model.get_input_embeddings(mx.array(x)[None], None, mask=None).to_dict()
        gen.insert([x], max_tokens=steps + 64, prompt_kwargs=[kw])
    while not (
        gen._generation_batch is not None
        and len(gen._generation_batch) == len(lengths)
        and gen._prompt_batch is None
        and not gen._unprocessed_sequences
    ):
        gen.next()
    return gen


def time_steps(gen, steps):
    for _ in range(3):
        gen.next()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        gen.next()
    mx.synchronize()
    return (time.perf_counter() - t0) / steps * 1000


def _stock_rows(lengths, H, D, lone=False):
    """Synthetic stock caches of one attention layer: a left-padded
    BatchKVCache of ``lengths`` or (``lone``) a one-row KVCache."""
    from mlx_vlm.models.cache import BatchKVCache, KVCache

    L = max(lengths)
    c = KVCache() if lone else BatchKVCache([L - n for n in lengths])
    k = mx.random.normal((len(lengths), H, L, D)).astype(mx.bfloat16)
    c.update_and_fetch(k, k)
    return c


def _timed(fn, caches):
    _sync_caches(caches)
    t0 = time.perf_counter()
    out = fn()
    _sync_caches(out if isinstance(out, list) else caches)
    return round((time.perf_counter() - t0) * 1000, 2)


def run_shape(model, processor, tok, name, lengths, ragged, steps):
    from mlx_vlm.generate import ar

    from yunshu_engine.kernels.ragged_kv import RaggedKVCache, enable

    enable(ragged)
    gen = build(model, processor, tok, lengths, steps)
    batch = gen._generation_batch
    row = {"shape": name, "ragged": ragged}
    row["caches"] = sorted({type(c).__name__ for c in batch.prompt_cache})
    row["step_ms"] = round(time_steps(gen, steps), 2)
    rag = [c for c in batch.prompt_cache if isinstance(c, RaggedKVCache)]
    if ragged and rag:
        H, D = rag[0].keys.shape[1], rag[0].keys.shape[3]
        # convert: a whole stock batch of this shape (every attention layer)
        stock = [_stock_rows(lengths, H, D) for _ in rag]
        row["convert_ms"] = _timed(
            lambda: [RaggedKVCache.from_cache(c, ragged) for c in stock], stock
        )
        del stock
        # first join: the lone decoding row (KVCache) meets a 512-token row
        a = [_stock_rows([lengths[0]], H, D, lone=True) for _ in rag]
        b = [_stock_rows([512], H, D) for _ in rag]
        row["first_join_ms"] = _timed(lambda: ar._extend_cache(a, b), a + b)
        del a, b
        # a short row finishes, then a 512-token row joins in its place
        keep = mx.array(list(range(len(rag[0].lengths) - 1)), dtype=mx.int32)
        row["filter_ms"] = _timed(lambda: [c.filter(keep) for c in rag] and rag, rag)
        extras = [_stock_rows([512], H, D) for _ in rag]
        row["join_ms"] = _timed(
            lambda: [c.extend(e) for c, e in zip(rag, extras, strict=True)] and rag,
            extras + rag,
        )
    gen.close()
    enable(None)
    mx.clear_cache()
    return row


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_dir")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--modes", default="off,bf16,int8")
    ap.add_argument("--label", default="")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    from yunshu_engine.kernels.omlx import apply

    apply()
    model, processor = load(a.model_dir)
    from yunshu_engine.kernels.ragged_kv import install

    install()
    tok = processor.tokenizer
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a") as f:
        for name in a.shapes.split(","):
            for mode in a.modes.split(","):
                ragged = None if mode == "off" else mode
                row = run_shape(
                    model, processor, tok, name, SHAPES[name], ragged, a.steps
                )
                row = {"label": a.label, **row}
                f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
