"""Timeline of one MTP cycle on the served speculative lane (Qwen3.5 family).

Same stack as ``sweep_mtp_depth.py --yunshu-kernels=exact --invariant
--invariant-packed --ragged-lane``. Upstream's ``_mtp_rounds_batch`` is wrapped
phase by phase (draft chain, verify forward, acceptance walk, drafter absorb,
commit); each phase reports the CPU time to build / launch it. Two modes:

  default   no extra synchronisation. ``walk`` is the wait for the GPU (the one
            sync of the cycle); the other phases are CPU graph-build time. If
            build phases are large next to the GPU time, the GPU idles.
  --barrier ``mx.synchronize()`` after every phase, so each phase's time is its
            own GPU time (plus launch): the additive split.

    python scripts/research/profile_mtp_cycle.py MODEL_DIR --context 1024 \
        --task code --tokens 160 [--barrier] --output runs/mtp-cycle.jsonl
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FILLER = "".join(
    f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n" for i in range(40000)
)
TASKS = {
    "code": "Write a Python LRU cache class with get, put, delete and resize, "
    "with docstrings and type hints. Output code only.",
    "prose": "Explain in detail how a refrigerator works, covering the refrigerant "
    "cycle, compressor, condenser and evaporator.",
    "json_like": "List ten European capitals with their countries and approximate "
    "populations as a markdown table.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--context", type=int, nargs="+", default=[1024])
    ap.add_argument("--task", default="code")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument(
        "--modes",
        default="plain,barrier",
        help="plain and / or barrier runs per context",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--packed-geometry", default=None, choices=("target", "few"))
    a = ap.parse_args()
    a.barrier = False

    import mlx_vlm.speculative.mtp as mtp
    from mlx_vlm import load
    from mlx_vlm.generate.ar import BatchGenerator

    from yunshu_engine.kernels import omlx, ragged_kv
    from yunshu_engine.kernels.omlx import qwen35_packed_linear

    if a.packed_geometry:
        qwen35_packed_linear.GEOMETRY = a.packed_geometry
    from yunshu_engine.kernels.batch_invariant import install as install_invariant
    from yunshu_engine.kernels.batch_invariant import set_active
    from yunshu_engine.mlxvlm_mtp import _load_drafter_in_memory
    from yunshu_engine.mrope import clear_rope_state

    omlx.apply(row_exact=False)
    model, processor = load(a.model_dir)
    lm = model.language_model
    install_invariant(lm, model=model, packed=omlx.is_nax_available())
    set_active(True)
    ragged_kv.install()
    ragged_kv.set_dense_lane(True)
    drafter = _load_drafter_in_memory(a.model_dir)
    tok = processor.tokenizer
    filler_ids = tok.encode(FILLER, add_special_tokens=False)

    acc = defaultdict(float)
    cnt = defaultdict(int)
    state = {"cycle_t0": None, "cycles": 0, "accepted": 0, "on": False}

    def timed(name, fn, sync_after=True):
        def inner(*args, **kw):
            if not state["on"]:
                return fn(*args, **kw)
            t = time.perf_counter()
            out = fn(*args, **kw)
            if a.barrier and sync_after:
                mx.synchronize()
            acc[name] += time.perf_counter() - t
            cnt[name] += 1
            return out

        return inner

    from mlx_vlm.speculative import common as spec_common

    rng = spec_common._SpeculativeSamplerRNG
    mtp._mtp_verify_target = timed("verify_build", mtp._mtp_verify_target, False)
    mtp._speculative_walk_batch = timed(
        "walk(host read)", mtp._speculative_walk_batch, False
    )
    rng.draft_tokens = timed("draft_chain(submit)", rng.draft_tokens)
    rng.target_eval = timed("verify(submit)", rng.target_eval)
    rng.draft_call = timed("absorb(submit)", rng.draft_call)
    mtp._MTPVerifyResult.commit = timed("commit(build)", mtp._MTPVerifyResult.commit)
    orig_record = mtp._record_speculative_round

    def record(model_, accepted, drafted):
        if state["on"]:
            state["cycles"] += 1
            state["accepted"] += float(accepted)
        return orig_record(model_, accepted, drafted)

    mtp._record_speculative_round = record

    from bench_context_batch import CORPUS, make_prompt

    def run(ids, block, max_tokens):
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
        uid = gen.insert([ids], max_tokens=max_tokens, prompt_kwargs=[kw])[0]
        out, stamps = [], []
        try:
            while True:
                _, resps = gen.next()
                done = False
                for r in resps:
                    if r.uid == uid:
                        out.append(int(r.token))
                        stamps.append(time.perf_counter())
                        if len(out) == 20:  # warmed: measure from here
                            state["on"] = True
                            acc.clear()
                            cnt.clear()
                            state["cycles"] = 0
                            state["accepted"] = 0
                            state["t0"] = time.perf_counter()
                            state["n0"] = len(out)
                        done = done or r.finish_reason is not None
                if done:
                    break
        finally:
            gen.close()
            state["on"] = False
        return out, stamps

    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ctx in a.context:
        if a.task == "corpus":
            prompt = make_prompt(tok, CORPUS.read_text(), ctx)
        else:
            prompt = tok.decode(filler_ids[:ctx]) + "\n\n" + TASKS[a.task]
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok.encode(text, add_special_tokens=False)
        run(ids, a.block, 32)  # kernels warm for this shape
        for mode in a.modes.split(","):
            a.barrier = mode == "barrier"
            out, stamps = run(ids, a.block, a.tokens)
            n = state["cycles"]
            wall = stamps[-1] - state["t0"]
            toks = len(out) - state["n0"]
            row = {
                "context": len(ids),
                "task": a.task,
                "barrier": a.barrier,
                "tokens_per_cycle": round(toks / max(n, 1), 3),
                "cycle_ms": round(wall / max(n, 1) * 1e3, 2),
                "tok_s": round(toks / wall, 2),
                "phase_ms": {k: round(v / max(n, 1) * 1e3, 2) for k, v in acc.items()},
            }
            row["other_ms"] = round(row["cycle_ms"] - sum(row["phase_ms"].values()), 2)
            with a.output.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            mx.clear_cache()


if __name__ == "__main__":
    main()
