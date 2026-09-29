"""Where does a single-request MTP cycle spend its time at long context?

Serving stack of the Qwen3.5-family speculative lane (exact oMLX kernels,
batch-invariant packed projections, ragged lane attention, in-memory MTP head),
one request, greedy. The MTP rounds are timed by wrapping upstream's target
verify forward and the drafter's ``draft_block`` with ``mx.synchronize`` on both
sides, so each cycle splits into verify / draft / rest (acceptance walk,
commits, Python). A plain (no draft) run of the same prompt gives the T=1 step.
Adding syncs costs a little overlap; the split, not the absolute tok/s, is the
point (compare tok/s against sweep_mtp_depth.py).

    python scripts/research/profile_long_decode.py MODEL_DIR --context 1024 131072 \
        --tokens 192 --output runs/profile-long.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

FILLER = "".join(
    f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n" for i in range(40000)
)
TASKS = {
    "code": "Write a Python LRU cache class with get, put, delete and resize, "
    "with docstrings and type hints. Output code only.",
    "prose": "Explain in detail how a refrigerator works, covering the refrigerant "
    "cycle, compressor, condenser and evaporator.",
}


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.verify_s = 0.0
        self.draft_s = 0.0
        self.cycles = 0
        self.accepted = 0
        self.drafted = 0
        self.depth_hits: list[int] = []
        self.depth_tries: list[int] = []

    def wrap(self, fn, attr):
        def inner(*a, **k):
            mx.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            if isinstance(out, mx.array):
                mx.eval(out)
            mx.synchronize()
            setattr(self, attr, getattr(self, attr) + time.perf_counter() - t)
            return out

        return inner

    def record(self, accepted, drafted):
        self.cycles += 1
        a = int(accepted if not isinstance(accepted, mx.array) else accepted.item())
        d = int(drafted)
        self.accepted += a
        self.drafted += d
        while len(self.depth_tries) < d:
            self.depth_tries.append(0)
            self.depth_hits.append(0)
        for i in range(d):
            self.depth_tries[i] += 1
            if i < a:
                self.depth_hits[i] += 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_dir")
    ap.add_argument("--context", type=int, nargs="+", default=[1024, 131072])
    ap.add_argument("--tasks", default="code,prose")
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--kv", choices=("bf16", "int8"), default="bf16")
    ap.add_argument("--no-plain", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    import mlx_vlm.speculative.mtp as mtp
    from mlx_vlm import load
    from mlx_vlm.generate.ar import BatchGenerator

    from yunshu_engine.kernels import omlx, ragged_kv
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
    if a.kv == "int8":
        ragged_kv.set_lane_format("int8")
    drafter = _load_drafter_in_memory(a.model_dir)
    tok = processor.tokenizer

    timer = Timer()
    mtp._mtp_verify_target = timer.wrap(mtp._mtp_verify_target, "verify_s")
    drafter.draft_block = timer.wrap(drafter.draft_block, "draft_s")
    orig_record = mtp._record_speculative_round

    def record(model_, accepted, drafted):
        timer.record(accepted, drafted)
        return orig_record(model_, accepted, drafted)

    mtp._record_speculative_round = record
    filler_ids = tok.encode(FILLER, add_special_tokens=False)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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
        timer.reset()
        t0 = time.perf_counter()
        uid = gen.insert([ids], max_tokens=max_tokens, prompt_kwargs=[kw])[0]
        out, first, stamps = [], None, []
        try:
            while True:
                _, resps = gen.next()
                done = False
                for r in resps:
                    if r.uid == uid:
                        if first is None:
                            first = time.perf_counter()
                            timer.reset()
                        out.append(int(r.token))
                        stamps.append(time.perf_counter())
                        done = done or r.finish_reason is not None
                if done:
                    break
        finally:
            gen.close()
        end = time.perf_counter()
        # stamps[i]: arrival of token i. Tokens of one MTP cycle arrive together,
        # so the first gap after token 0 holds any one-off work done after the
        # first token (e.g. the drafter's pass over the prompt).
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        first_gap = max(gaps[:8]) if gaps else 0.0
        return out, first - t0, end - first, first_gap, stamps

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a") as f:
        for ctx in a.context:
            for task in a.tasks.split(","):
                if task == "corpus":
                    # the server speed sweep's prompt (bench_context_batch.py)
                    prompt = make_prompt(tok, CORPUS.read_text(), ctx)
                else:
                    prompt = tok.decode(filler_ids[:ctx]) + "\n\n" + TASKS[task]
                text = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                ids = tok.encode(text, add_special_tokens=False)
                row = {"context": len(ids), "task": task, "kv": a.kv}
                if not a.no_plain:
                    plain, _, dec, _, _ = run(ids, 0, 48)
                    row["plain_step_ms"] = round(dec / (len(plain) - 1) * 1000, 2)
                    ref = plain
                out, ttft, dec, first_gap, stamps = run(ids, a.block, a.tokens)
                # steady state: from the 16th token on (one-off work excluded)
                k = min(16, len(stamps) - 2)
                steady = (len(stamps) - 1 - k) / (stamps[-1] - stamps[k])
                c = max(timer.cycles, 1)
                row.update(
                    {
                        "ttft_s": round(ttft, 2),
                        "tokens": len(out),
                        "mtp_tps": round((len(out) - 1) / dec, 2),
                        "steady_tps": round(steady, 2),
                        "first_gap_ms": round(first_gap * 1000, 1),
                        "cycles": timer.cycles,
                        "tokens_per_cycle": round(
                            (timer.accepted + timer.cycles) / c, 3
                        ),
                        "cycle_ms": round(dec / c * 1000, 2),
                        "verify_ms": round(timer.verify_s / c * 1000, 2),
                        "draft_ms": round(timer.draft_s / c * 1000, 2),
                        "rest_ms": round(
                            (dec - timer.verify_s - timer.draft_s) / c * 1000, 2
                        ),
                        "depth_acceptance": [
                            round(h / t, 3) if t else None
                            for h, t in zip(
                                timer.depth_hits, timer.depth_tries, strict=False
                            )
                        ],
                    }
                )
                if not a.no_plain:
                    n = min(len(ref), len(out))
                    row["parity_prefix"] = out[:n] == ref[:n]
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
                mx.clear_cache()


if __name__ == "__main__":
    main()
