"""Cold prefill TTFT vs prefill chunk size on upstream BatchGenerator (Qwen3.8).

No APC, no draft: every request is a cold prefill of a unique prompt. Reports
first-token latency, prefill tokens/s and MLX peak memory per (length, step).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/research/bench_prefill_step.py MODEL_DIR
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.ar import BatchGenerator  # noqa: E402

from yunshu_engine.mrope import clear_rope_state  # noqa: E402

model_dir = sys.argv[1]
lengths = [2048, 8192, 32768]
steps = [512, 1024, 2048, 4096, 8192]
model, processor = load(model_dir)
tok = processor.tokenizer
lm = model.language_model
line = "Section {i}: routine maintenance log for building {b}, wiring and plumbing checked.\n"


def prompt_ids(n_tokens, salt):
    body, i = "", 0
    ids = []
    while len(ids) < n_tokens:
        body += "".join(
            line.format(i=i + j, b=salt * 7 + (i + j) % 31) for j in range(200)
        )
        i += 200
        msgs = [
            {
                "role": "user",
                "content": body
                + "\nWhat is the last section number? Reply with the number only.",
            }
        ]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok.encode(text, add_special_tokens=False)
    return ids


def ttft(ids, step):
    gen = BatchGenerator(
        lm,
        processor,
        max_tokens=1,
        greedy_sampling=True,
        compute_logprobs=False,
        prefill_step_size=step,
    )
    clear_rope_state(model)
    kw = model.get_input_embeddings(mx.array(ids)[None], None, mask=None).to_dict()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    uid = gen.insert([ids], max_tokens=1, prompt_kwargs=[kw])[0]
    try:
        while True:
            _, resps = gen.next()
            if any(r.uid == uid for r in resps):
                break
    finally:
        gen.close()
    return time.perf_counter() - t0, mx.get_peak_memory() / 2**30


ttft(prompt_ids(512, 99), 2048)  # warm kernels
salt = 0
for n in lengths:
    for step in steps:
        salt += 1
        ids = prompt_ids(n, salt)
        dt, peak = ttft(ids, step)
        print(
            json.dumps(
                {
                    "tokens": len(ids),
                    "prefill_step": step,
                    "ttft_s": round(dt, 3),
                    "prefill_tps": round(len(ids) / dt, 1),
                    "peak_gib": round(peak, 2),
                }
            ),
            flush=True,
        )
        mx.clear_cache()
