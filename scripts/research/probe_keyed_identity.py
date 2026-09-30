"""Keyed sampled speculative lane vs keyed serial sampling, token for token.

For each (prompt, sampling config, seed) it generates N tokens with
  B  the speculative lane (position-keyed sampler, drafts verified),
  C  keyed serial: the same KeyedSampler on full-vocabulary logits from one-token decode steps
     (the non-speculative batch with RowSampler replaced by the keyed sampler),
  A  the stock non-speculative sampled path (RowSampler / mx.random.categorical), for a speed and
     sanity reference only (a different random stream, so no token comparison).
Lossless means B == C. It also checks, at the first positions of each prompt, that the keyed
draws over many seeds follow softmax(filtered logits / T) (chi-square against the histogram).

    probe_keyed_identity.py --model $M --tokens 160 --out result.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
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
    "json": 'Return a JSON object describing three fruits with fields "name", "color" and "calories".',
}
CONFIGS = [
    dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0),
    dict(temperature=0.7, top_p=0.8, top_k=0, min_p=0.05),
    dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0),
    dict(temperature=1.0, top_p=0.9, top_k=40, min_p=0.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    ap.add_argument("--configs", type=int, nargs="+", default=list(range(len(CONFIGS))))
    ap.add_argument("--hist", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import mlx.core as mx
    import numpy as np

    from yunshu_engine import keyed_sampling, vlm_batch_runner
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(a.model)
    asyncio.new_event_loop().run_until_complete(engine.start())
    runner = engine._batch_runner
    tok = getattr(runner.processor, "tokenizer", runner.processor)
    out = open(a.out, "a")  # noqa: SIM115

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row)[:400], flush=True)

    def run(ids, pkw, salt, cfg, seed, draft, n):
        stats = vlm_batch_runner.RunStats()
        toks: list[int] = []

        def consume():
            for t in runner.iter_tokens(
                ids,
                max_tokens=n,
                seed=seed,
                prompt_kwargs=pkw,
                apc_semantic_hash=salt,
                allow_draft=draft,
                stats=stats,
                **cfg,
            ):
                toks.append(t)

        t0 = time.perf_counter()
        th = threading.Thread(target=consume)
        th.start()
        th.join()
        return toks, stats, time.perf_counter() - t0

    from yunshu_engine import mtp_lane

    lane = {"calls": 0, "keyed": 0, "rounds": 0, "accepted": 0}
    orig_rounds = mtp_lane.rounds

    reject = [False]

    def counting_rounds(*args, **kw):
        if reject[0]:
            dm = args[1]
            real = getattr(dm, "_real_greedy", None) or dm._greedy_token
            dm._real_greedy = real
            # drafts are token 0 everywhere: the lane verifies and draws serially
            dm._greedy_token = lambda h: mx.zeros(h.shape[:2], dtype=mx.int32)
        elif getattr(args[1], "_real_greedy", None):
            args[1]._greedy_token = args[1]._real_greedy
        lane["calls"] += 1
        lane["keyed"] += kw.get("keyed") is not None
        for toks, meta in orig_rounds(*args, **kw):
            if meta and meta.get("round_pos") == 0:
                lane["rounds"] += 1
            if meta and meta.get("round_pos"):
                lane["accepted"] += (
                    1  # a token past the first of its round = an accepted draft
                )
            yield toks, meta

    mtp_lane.rounds = counting_rounds

    RS = vlm_batch_runner.RowSampler
    orig_call, orig_target = RS.__call__, RS.sample_target
    captured: list = []

    seen = [0]

    def keyed_serial(cfg, seed):
        seen[0] = 0
        params = vlm_batch_runner.RowParams(
            cfg["temperature"], cfg["top_p"], cfg["top_k"], cfg["min_p"], seed
        )
        ks = keyed_sampling.KeyedSampler(params, seed)

        def call(self, logprobs):
            seen[0] += 1
            if seen[0] % 20 == 5 and len(captured) < 6:
                captured.append((ks._next, logprobs.astype(mx.float32)))
            return ks(logprobs)

        def target(self, logprobs, row_ids=None, positions=None):
            seen[0] += 1
            if seen[0] % 20 == 5 and len(captured) < 6:
                captured.append((ks._next, logprobs.astype(mx.float32)))
            return ks.sample_target(logprobs, row_ids, positions)

        RS.__call__, RS.sample_target = call, target

    def restore():
        RS.__call__, RS.sample_target = orig_call, orig_target

    bad = 0
    for pname in a.prompts:
        msgs = [{"role": "user", "content": PROMPTS[pname]}]
        ids, pkw, salt = engine._executor.submit(
            engine._runner_input, msgs, [], [], False, {}
        ).result()
        for ci in a.configs:
            cfg = CONFIGS[ci]
            for seed in a.seeds:
                captured.clear()
                keyed_serial(cfg, seed)
                c, cs, ct = run(ids, pkw, salt, cfg, seed, False, a.tokens)
                restore()
                reject[0] = True
                for k in lane:
                    lane[k] = 0
                c2, _, ct2 = run(ids, pkw, salt, cfg, seed, True, a.tokens)
                lane_c2 = dict(lane)
                reject[0] = False
                # one lane request at a time; the lane engages when draft is allowed
                for k in lane:
                    lane[k] = 0
                b, bs, bt = run(ids, pkw, salt, cfg, seed, True, a.tokens)
                lane_b = dict(lane)
                al, als, at = run(ids, pkw, salt, cfg, seed, False, a.tokens)
                first = next(
                    (i for i, (x, y) in enumerate(zip(b, c, strict=False)) if x != y),
                    None if len(b) == len(c) else min(len(b), len(c)),
                )
                first2 = next(
                    (i for i, (x, y) in enumerate(zip(b, c2, strict=False)) if x != y),
                    None if len(b) == len(c2) else min(len(b), len(c2)),
                )
                row = {
                    "B_eq_C2": first2 is None,
                    "first_diff_C2": first2,
                    "lane_C2": lane_c2,
                    "tps_C2": round(len(c2) / ct2, 1),
                    "C_eq_C2": c == c2,
                    "prompt": pname,
                    "cfg": ci,
                    "seed": seed,
                    "n": len(c),
                    "B_eq_C": first is None,
                    "first_diff": first,
                    "used_draft": bs.used_draft,
                    "lane": lane_b,
                    "tps_B": round(len(b) / bt, 1),
                    "tps_C": round(len(c) / ct, 1),
                    "tps_A": round(len(al) / at, 1),
                }
                if first is not None:
                    bad += first2 is not None
                    lo = max(0, first - 3)
                    row["ctx_B"] = tok.decode(b[lo : first + 4])
                    row["ctx_C"] = tok.decode(c[lo : first + 4])
                    row["tok_B"], row["tok_C"] = (
                        b[first : first + 1],
                        c[first : first + 1],
                    )
                emit(row)
            # distribution at the first captured positions (real full-vocab logits)
            if captured:

                def hist_job(captured=tuple(captured), cfg=cfg, pname=pname, ci=ci):
                    params = vlm_batch_runner.RowParams(
                        cfg["temperature"], cfg["top_p"], cfg["top_k"], cfg["min_p"]
                    )
                    for pi, lp in captured[:4]:
                        lp = lp.reshape(1, -1)
                        filt = keyed_sampling.filter_logprobs(lp, params)
                        p = (
                            np.array(mx.softmax(filt, axis=-1))
                            .reshape(-1)
                            .astype(np.float64)
                        )
                        counts = np.zeros_like(p)
                        for s in range(a.hist):
                            ks = keyed_sampling.KeyedSampler(params, 10_000 + s)
                            counts[int(ks.sample_positions(lp, [pi])[0])] += 1
                        support = p > 0
                        outside = int(counts[~support].sum())
                        exp = p * a.hist
                        m = exp >= 5
                        chi = float((((counts[m] - exp[m]) ** 2) / exp[m]).sum())
                        dof = int(m.sum()) - 1
                        tv = float(0.5 * np.abs(counts / a.hist - p).sum())
                        emit(
                            {
                                "prompt": pname,
                                "cfg": ci,
                                "hist_pos": pi,
                                "outside_support": outside,
                                "chi2": round(chi, 1),
                                "dof": dof,
                                "tv": round(tv, 4),
                                "support": int(support.sum()),
                            }
                        )

                engine._executor.submit(hist_job).result()
    emit({"summary": "done", "mismatches": bad})
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
