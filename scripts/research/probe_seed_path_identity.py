"""Same seed, same tokens: speculative lane vs shared batch (alone and mixed), plus decode speed.

For each (prompt, sampling config, seed):
  L  solo, drafting allowed  -> the single-row speculative lane (KeyedSampler)
  S  solo, drafting off      -> the shared batch (RowSampler)
  M  drafting off, with three other sampled requests decoding at the same time
All three must emit the same tokens. The speed rows time S solo and a 4-wide shared batch.

    PYTHONPATH=python probe_seed_path_identity.py --model $M --tokens 160 --out result.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

PROMPTS = {
    "code": "Write a Python function that parses ISO-8601 durations, with tests.",
    "prose": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "cn": "請用繁體中文解釋為什麼天空是藍色的，並舉出三個生活中的例子。",
}
CONFIGS = [
    dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0),
    dict(temperature=1.0, top_p=0.9, top_k=0, min_p=0.05),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--speed-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from yunshu_engine import vlm_batch_runner as vbr
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(a.model)
    asyncio.new_event_loop().run_until_complete(engine.start())
    runner = engine._batch_runner
    tok = getattr(runner.processor, "tokenizer", runner.processor)
    out = open(a.out, "a")  # noqa: SIM115

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row)[:300], flush=True)

    def ids_of(text):
        msgs = [{"role": "user", "content": text}]
        s = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return list(tok.encode(s))

    def consume(ids, cfg, seed, draft, n, sink, stats):
        def go():
            for t in runner.iter_tokens(
                ids,
                max_tokens=n,
                seed=seed,
                prompt_kwargs=None,
                allow_draft=draft,
                stats=stats,
                **cfg,
            ):
                sink.append(t)

        th = threading.Thread(target=go)
        th.start()
        return th

    def digest(t):
        return hashlib.sha256(json.dumps(t).encode()).hexdigest()[:16]

    ok = True
    for pname, text in {} if a.speed_only else PROMPTS.items():
        ids = ids_of(text)
        for ci, cfg in enumerate(CONFIGS):
            for seed in a.seeds:
                res = {}
                for mode in ("L", "S", "M"):
                    toks, st = [], vbr.RunStats()
                    ths = [consume(ids, cfg, seed, mode == "L", a.tokens, toks, st)]
                    if mode == "M":
                        for k in range(3):
                            ths.append(
                                consume(
                                    ids_of(text + f" (variant {k})"),
                                    cfg,
                                    100 + k,
                                    False,
                                    a.tokens,
                                    [],
                                    vbr.RunStats(),
                                )
                            )
                    for th in ths:
                        th.join()
                    res[mode] = (toks, st)
                same = res["L"][0] == res["S"][0] == res["M"][0]
                ok &= same

                def first_diff(m, res=res):
                    pairs = zip(res["S"][0], res[m][0], strict=False)
                    return next((i for i, (x, y) in enumerate(pairs) if x != y), None)

                emit(
                    dict(
                        kind="identity",
                        prompt=pname,
                        cfg=ci,
                        seed=seed,
                        identical=same,
                        n=len(res["S"][0]),
                        digest={m: digest(res[m][0]) for m in res},
                        used_draft={m: res[m][1].used_draft for m in res},
                        first_diff={m: first_diff(m) for m in ("L", "M")},
                    )
                )
    # speed: S solo and a 4-wide shared batch (sampled, seeded)
    ids = ids_of(PROMPTS["prose"])
    cfg = CONFIGS[0]
    for rep in range(3):
        t0 = time.perf_counter()
        toks = []
        consume(ids, cfg, 5, False, 256, toks, vbr.RunStats()).join()
        emit(
            dict(
                kind="speed",
                mode="solo",
                rep=rep,
                tokens=len(toks),
                s=time.perf_counter() - t0,
            )
        )
        sinks = [[] for _ in range(4)]
        t0 = time.perf_counter()
        ths = [
            consume(
                ids_of(PROMPTS["prose"] + f" ({k})"),
                cfg,
                5 + k,
                False,
                256,
                sinks[k],
                vbr.RunStats(),
            )
            for k in range(4)
        ]
        for th in ths:
            th.join()
        emit(
            dict(
                kind="speed",
                mode="b4",
                rep=rep,
                tokens=sum(map(len, sinks)),
                s=time.perf_counter() - t0,
            )
        )
    emit(dict(kind="summary", all_identical=ok))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
