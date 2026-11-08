"""Cost of each keyed-draw stage at the sampled-spec shapes (top_p 0.95, top_k 20, T 1)."""

import json
import time


def per_call_ms(fn, reps=20):
    import mlx.core as mx

    mx.eval(fn())
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    return round((time.perf_counter() - t) / reps * 1e3, 3)


def main():
    import mlx.core as mx

    from yunshu_engine import keyed_sampling as ks
    from yunshu_engine.constrained_spec import normalize_rows
    from yunshu_engine.vlm_batch_runner import RowParams

    params = RowParams(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)
    vocab = 248320
    for n in (15, 31):
        logits = (mx.random.normal((1, n, vocab)) * 3).astype(mx.bfloat16)
        mx.eval(logits)
        lp = normalize_rows(logits)
        mx.eval(lp)
        out = dict(
            n=n,
            normalize=per_call_ms(lambda: normalize_rows(logits)),
            top_p=per_call_ms(lambda: ks.top_p_filter(lp, 0.95)),
            top_k=per_call_ms(lambda: ks.top_k_filter(lp, 20)),
            filter_all=per_call_ms(lambda: ks.filter_logprobs(lp, params)),
        )
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
