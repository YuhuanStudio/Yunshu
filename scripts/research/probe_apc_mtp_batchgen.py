"""Engine-level probe: can upstream mlx-vlm BatchGenerator combine APC and an MTP draft?

Loads the Qwen3.8 MTP checkpoint once (target + in-memory MTP head, no drafter
copy on disk) and runs the same request sequence through four configurations:
AR, APC only, MTP only, APC+MTP. Records token ids (parity vs AR), prompt/cached
tokens, first-token latency and decode rate. Prints JSON lines.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/research/probe_apc_mtp_batchgen.py MODEL_DIR > out.jsonl
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from mlx_vlm import load  # noqa: E402
from mlx_vlm.apc import APCManager, semantic_extra_hash  # noqa: E402
from mlx_vlm.generate.ar import BatchGenerator  # noqa: E402

from yunshu_engine.mlxvlm_mtp import _load_drafter_in_memory  # noqa: E402
from yunshu_engine.mrope import clear_rope_state  # noqa: E402

model_dir = sys.argv[1]
model, processor = load(model_dir)
tok = processor.tokenizer
drafter = _load_drafter_in_memory(model_dir)
lm = model.language_model
sem = semantic_extra_hash(
    image_hash=0, media={"audio": None, "video": None}, model=lm, processor=processor
)


def prompt_ids(messages):
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok.encode(text, add_special_tokens=False)


doc = "".join(
    f"Section {i}: The municipal archive logs routine maintenance for building {100 + i % 37}.\n"
    for i in range(260)
)
ask = (
    "\n\nQuestion: what is the access code in the final line? Reply with the code only."
)
code_q = "Write a Python LRU cache class with get, put and delete, with docstrings. Output code only."
seq = [
    (
        "doc_cold",
        [
            {
                "role": "user",
                "content": doc + "Final line: the access code is ALPHA." + ask,
            }
        ],
        16,
    ),
    (
        "doc_repeat",
        [
            {
                "role": "user",
                "content": doc + "Final line: the access code is ALPHA." + ask,
            }
        ],
        16,
    ),
    (
        "doc_edited",
        [
            {
                "role": "user",
                "content": doc + "Final line: the access code is COBALT." + ask,
            }
        ],
        16,
    ),
    ("code_long", [{"role": "user", "content": code_q}], 256),
    (
        "doc_followup",
        [
            {
                "role": "user",
                "content": doc + "Final line: the access code is COBALT." + ask,
            },
            {"role": "assistant", "content": "COBALT"},
            {
                "role": "user",
                "content": "Write two sentences about why archives keep maintenance logs.",
            },
        ],
        96,
    ),
]


def run(config):
    apc = (
        APCManager(
            num_blocks=512, block_size=16, disk=None, overrides={"memory_max_gb": 6.0}
        )
        if "apc" in config
        else None
    )
    rows = []
    for name, msgs, max_tokens in seq:
        ids = prompt_ids(msgs)
        before = apc.stats.matched_tokens if apc else 0
        gen = BatchGenerator(
            lm,
            processor,
            max_tokens=max_tokens,
            apc_manager=apc,
            draft_model=drafter if "mtp" in config else None,
            draft_kind="mtp" if "mtp" in config else None,
            greedy_sampling=True,
            compute_logprobs=False,
            prefill_step_size=2048,
        )
        t0 = time.perf_counter()
        clear_rope_state(model)
        kw = model.get_input_embeddings(mx.array(ids)[None], None, mask=None).to_dict()
        kw["_apc_semantic_hash"] = sem
        uid = gen.insert([ids], max_tokens=max_tokens, prompt_kwargs=[kw])[0]
        out, first = [], None
        try:
            while True:
                _, resps = gen.next()
                done = False
                for r in resps:
                    if r.uid != uid:
                        continue
                    if first is None:
                        first = time.perf_counter() - t0
                    out.append(int(r.token))
                    if r.finish_reason is not None:
                        done = True
                if done:
                    break
        finally:
            gen.close()
        wall = time.perf_counter() - t0
        rows.append(
            {
                "config": config,
                "case": name,
                "prompt_tokens": len(ids),
                "cached": (apc.stats.matched_tokens - before) if apc else 0,
                "first_token_s": round(first, 3),
                "wall_s": round(wall, 3),
                "n": len(out),
                "decode_tps": round((len(out) - 1) / (wall - first), 2)
                if len(out) > 8
                else None,
                "text": tok.decode(out)[:120],
                "tokens": out,
            }
        )
        print(
            json.dumps(
                {k: v for k, v in rows[-1].items() if k != "tokens"}, ensure_ascii=False
            ),
            flush=True,
        )
    mx.clear_cache()
    return rows


results = {c: run(c) for c in ("ar", "apc", "mtp", "apc+mtp")}
for c in ("apc", "mtp", "apc+mtp"):
    parity = [
        a["tokens"] == b["tokens"]
        for a, b in zip(results["ar"], results[c], strict=True)
    ]
    print(
        json.dumps(
            {
                "parity_vs_ar": c,
                "per_case": dict(zip([s[0] for s in seq], parity, strict=True)),
            }
        ),
        flush=True,
    )
