"""Cold decode sweep: upstream BatchGenerator MTP draft_block_size vs AR on Qwen3.8.

Same in-memory drafter as probe_apc_mtp_batchgen.py; no APC so every request is
cold. Reports decode tok/s, first-token latency and token parity against AR.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/research/sweep_mtp_depth.py MODEL_DIR [sizes...]
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.ar import BatchGenerator  # noqa: E402

from yunshu_engine.mlxvlm_mtp import _load_drafter_in_memory  # noqa: E402
from yunshu_engine.mrope import clear_rope_state  # noqa: E402

model_dir = sys.argv[1]
sizes = [int(x) for x in sys.argv[2:]] or [3, 4, 5, 6]
model, processor = load(model_dir)
tok = processor.tokenizer
drafter = _load_drafter_in_memory(model_dir)
lm = model.language_model

tasks = [
    (
        "code",
        "Write a Python LRU cache class with get, put, delete and resize, with docstrings and type hints. Output code only.",
        384,
    ),
    (
        "prose",
        "Explain in detail how a refrigerator works, covering the refrigerant cycle, compressor, condenser and evaporator.",
        384,
    ),
    (
        "json_like",
        "List ten European capitals with their countries and approximate populations as a markdown table.",
        256,
    ),
]


def run(name, prompt, max_tokens, block):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tok.encode(text, add_special_tokens=False)
    gen = BatchGenerator(
        lm,
        processor,
        max_tokens=max_tokens,
        draft_model=drafter if block else None,
        draft_kind="mtp" if block else None,
        draft_block_size=block or None,
        greedy_sampling=True,
        compute_logprobs=False,
    )
    clear_rope_state(model)
    kw = model.get_input_embeddings(mx.array(ids)[None], None, mask=None).to_dict()
    t0 = time.perf_counter()
    uid = gen.insert([ids], max_tokens=max_tokens, prompt_kwargs=[kw])[0]
    out, first = [], None
    try:
        while True:
            _, resps = gen.next()
            done = False
            for r in resps:
                if r.uid == uid:
                    first = first or (time.perf_counter() - t0)
                    out.append(int(r.token))
                    done = done or r.finish_reason is not None
            if done:
                break
    finally:
        gen.close()
    wall = time.perf_counter() - t0
    return {
        "task": name,
        "block": block,
        "n": len(out),
        "first_s": round(first, 3),
        "decode_tps": round((len(out) - 1) / (wall - first), 2),
        "tokens": out,
    }


ref = {}
for name, prompt, mt in tasks:
    run(name, prompt, 16, 0)  # warm kernels for this shape
    for block in [0] + sizes:
        r = run(name, prompt, mt, block)
        if block == 0:
            ref[name] = r["tokens"]
        r["parity"] = r["tokens"] == ref[name]
        r.pop("tokens")
        print(json.dumps(r), flush=True)
    mx.clear_cache()
