"""Where does a mixed-length decode step spend its time? (in-process, one batch)

Builds one upstream BatchGenerator batch like a long-reasoning concurrency
load: one row with a long context and several short rows, prefilled one at a
time, then times plain decode steps. Runs the same batch with the stock padded
BatchKVCache and with Yunshu's ragged per-row KV (bf16 and int8), and optionally
times the model's 16 attention layers alone (--attn-only) by timing a decode
step with attention replaced by a no-op of the same output shape, so the
difference is the attention share.

    python scripts/research/bench_decode_step_mixed.py <model_dir> \
        --long 16000 --short 500 --rows 8 --steps 64 --output runs/step-mixed.jsonl
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


def run(model, processor, tok, a, ragged: str | None, attn_noop: bool):
    corpus = CORPUS.read_text()
    lengths = [a.long] + [a.short] * (a.rows - 1)
    ids = [
        tok.encode(make_prompt(tok, corpus, n), add_special_tokens=False)
        for n in lengths
    ]
    gen = BatchGenerator(
        model.language_model,
        processor,
        max_tokens=a.steps + 8,
        greedy_sampling=True,
        compute_logprobs=False,
        prefill_batch_size=1,
    )
    clear_rope_state(model)
    for x in ids:
        kw = model.get_input_embeddings(mx.array(x)[None], None, mask=None).to_dict()
        gen.insert([x], max_tokens=a.steps + 8, prompt_kwargs=[kw])
    # Prefill everything and reach steady decode with all rows.
    while not (
        gen._generation_batch is not None
        and len(gen._generation_batch) == a.rows
        and gen._prompt_batch is None
        and not gen._unprocessed_sequences
    ):
        gen.next()
    if ragged:
        from yunshu_engine.kernels.ragged_kv import convert_batch

        convert_batch(gen._generation_batch.prompt_cache, ragged)
    restore = None
    if attn_noop:
        from mlx_vlm.models.qwen3_5 import language as q35

        cls = q35.Qwen3_5Attention
        orig = cls.__call__

        def noop(self, x, *args, **kwargs):
            return mx.zeros_like(x)

        cls.__call__ = noop
        restore = (cls, orig)
    for _ in range(4):
        gen.next()
    mx.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(a.steps):
        _, resps = gen.next()
        n += len(resps)
    mx.synchronize()
    dt = time.perf_counter() - t0
    if restore:
        restore[0].__call__ = restore[1]
    gen.close()
    mx.clear_cache()
    return {
        "ragged": ragged,
        "attn_noop": attn_noop,
        "step_ms": round(dt / a.steps * 1000, 2),
        "aggregate_tps": round(n / dt, 1),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_dir")
    ap.add_argument("--long", type=int, default=16000)
    ap.add_argument("--short", type=int, default=500)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    from yunshu_engine.kernels.omlx import apply, is_nax_available, pack_projections

    apply()
    model, processor = load(a.model_dir)
    if is_nax_available():
        pack_projections(model)
    from yunshu_engine.kernels.ragged_kv import install

    install()
    tok = processor.tokenizer
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a") as f:
        meta = {
            "kind": "meta",
            "long": a.long,
            "short": a.short,
            "rows": a.rows,
            "steps": a.steps,
        }
        f.write(json.dumps(meta) + "\n")
        runs = [(None, False), ("bf16", False), ("int8", False), (None, True)]
        for ragged, noop in runs:
            row = {"kind": "run", **run(model, processor, tok, a, ragged, noop)}
            f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
