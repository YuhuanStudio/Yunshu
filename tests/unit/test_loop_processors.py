"""The engine loop's penalties / logit_bias / suppress / min_tokens use the shared dense
processors (text_logit_processors) and its logprobs use the shared top-k helper: same bits
as the per-token-id loops and the full-vocabulary sort they replaced."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.request import SamplingParams
from yunshu_engine.scheduler import Scheduler

V = 97
EOS = [1, 2]
PROMPT = [11, 12, 13, 14, 15]


class _Tok:
    eos_token_ids = set(EOS)


class _Stub:
    running: list = []
    tokenizer = _Tok()


# ── the loop's old processors, verbatim (scheduler._make_sampler before this change) ──


def old_chain(sp, prompt):
    from mlx_lm.sample_utils import make_logits_processors

    procs = make_logits_processors(
        repetition_penalty=sp.repetition_penalty
        if sp.repetition_penalty != 1.0
        else None,
        logit_bias=getattr(sp, "logit_bias", None),
    )
    fp, pp = sp.frequency_penalty, sp.presence_penalty
    if fp != 0.0 or pp != 0.0:
        st = {"counts": {}, "last_len": -1}
        n_prompt = len(prompt)

        def _freq_pres_penalty(tokens, logits, fp=fp, pp=pp, n_prompt=n_prompt, _st=st):
            counts = _st["counts"]
            last_len = int(_st["last_len"])
            cur_len = len(tokens)
            if cur_len <= n_prompt:
                _st["last_len"] = cur_len
                return logits
            if cur_len < last_len or last_len < n_prompt:
                counts = {}
                for t in tokens[n_prompt:]:
                    counts[int(t)] = counts.get(int(t), 0) + 1
                _st["counts"] = counts
            else:
                start = max(last_len, n_prompt)
                for t in tokens[start:]:
                    counts[int(t)] = counts.get(int(t), 0) + 1
            _st["last_len"] = cur_len
            for tid, cnt in counts.items():
                if fp != 0.0:
                    logits[..., tid] = logits[..., tid] - fp * cnt
                if pp != 0.0 and cnt > 0:
                    logits[..., tid] = logits[..., tid] - pp
            return logits

        procs.append(_freq_pres_penalty)

    def _ban(logits, ids):
        vocab = logits.shape[-1]
        for tid in ids:
            if 0 <= tid < vocab:
                logits[..., tid] = -float("inf")
        return logits

    if sp.suppress_tokens:
        ids = [int(t) for t in sp.suppress_tokens]
        procs.append(lambda _t, lg, ids=ids: _ban(lg, ids))
    if sp.ignore_eos:
        procs.append(lambda _t, lg: _ban(lg, list(EOS)))
    if sp.min_tokens > 0:

        def _min(toks, lg, floor=sp.min_tokens, nprompt=len(prompt)):
            return _ban(lg, list(EOS)) if (len(toks) - nprompt) < floor else lg

        procs.append(_min)
    return procs


def _bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32).tolist()


PARAMS = [
    dict(frequency_penalty=0.7, presence_penalty=0.4),
    dict(frequency_penalty=-1.3),
    dict(presence_penalty=1.9),
    dict(logit_bias={3: 2.5, 9: -4.0, 50: 0.1}),
    dict(repetition_penalty=1.3, logit_bias={5: 1.5}, frequency_penalty=0.5),
    dict(suppress_tokens=[4, 8, 8], ignore_eos=True),
    dict(min_tokens=4),
    dict(
        frequency_penalty=0.3,
        presence_penalty=0.2,
        logit_bias={7: 3.0},
        suppress_tokens=[20],
        min_tokens=3,
        repetition_penalty=1.1,
    ),
]


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("kw", PARAMS)
def test_loop_processors_match_the_old_per_id_loops(dtype, kw):
    sp = SamplingParams(temperature=0.0, **kw)
    sampler = Scheduler._make_sampler(_Stub(), sp, list(PROMPT))
    new = sampler._logits_processors
    old = old_chain(sp, PROMPT)
    rng = np.random.default_rng(1)
    toks = list(PROMPT)
    for _ in range(40):
        base = mx.array(rng.standard_normal((1, V)).astype(np.float32) * 3).astype(
            dtype
        )
        a, b = mx.array(base), mx.array(base)
        for p in old:
            a = p(toks, a)
        for p in new:
            b = p(toks, b)
        assert _bits(a) == _bits(b)
        toks.append(int(rng.choice([3, 3, 9, 20, int(rng.integers(0, V))])))


def test_each_row_keeps_its_own_counts():
    sp = SamplingParams(temperature=0.0, frequency_penalty=1.0)
    s1 = Scheduler._make_sampler(_Stub(), sp, list(PROMPT))
    s2 = Scheduler._make_sampler(_Stub(), sp, list(PROMPT))
    zero = mx.zeros((1, V))
    s1._tokens.extend([30, 30, 30])
    s2._tokens.extend([31])
    o1 = s1._logits_processors[0](s1._tokens, zero)
    o2 = s2._logits_processors[0](s2._tokens, zero)
    assert o1[0, 30].item() == -3.0 and o1[0, 31].item() == 0.0
    assert o2[0, 31].item() == -1.0 and o2[0, 30].item() == 0.0


def test_the_sampler_end_to_end_picks_the_same_tokens():
    """Greedy through _LogitsProcessorSampler: old chain and new chain walk one stream."""
    sp = SamplingParams(
        temperature=0.0,
        frequency_penalty=0.9,
        presence_penalty=0.3,
        logit_bias={6: 1.0},
        suppress_tokens=[10],
        min_tokens=3,
    )
    sampler = Scheduler._make_sampler(_Stub(), sp, list(PROMPT))
    old = old_chain(sp, PROMPT)
    rng = np.random.default_rng(9)
    toks = list(PROMPT)
    for _ in range(30):
        lg = mx.array(rng.standard_normal((1, V)).astype(np.float32) * 2)
        want = mx.array(lg)
        for p in old:
            want = p(toks, want)
        got = int(sampler(lg).item())
        assert got == int(mx.argmax(want, axis=-1).item())
        toks.append(got)


# ── logprobs ──


def old_entry(log_probs, token, top_k):
    """The full-vocabulary sort the text path replaced (stable argsort of -lp)."""
    lp = log_probs.astype(mx.float32)
    lp = mx.where(mx.isnan(lp), mx.array(-100.0, dtype=lp.dtype), lp)
    e = {"token_id": int(token), "logprob": float(lp[token])}
    if top_k and top_k > 0:
        idx = mx.argsort(-lp)[:top_k]
        e["top_logprobs"] = [
            {"token_id": int(idx[j]), "logprob": float(lp[idx[j]])}
            for j in range(min(top_k, lp.shape[-1]))
        ]
    return e


def _lp_rows():
    rng = np.random.default_rng(4)
    rows = [rng.standard_normal(V).astype(np.float32)]
    t = rng.standard_normal(V).astype(np.float32)
    t[[5, 40, 6, 70]] = 9.0  # ties at the top
    rows.append(t)
    t2 = np.full(V, -np.inf, dtype=np.float32)
    t2[[3, 60]] = -1.0  # -inf tail, fewer finite than k
    rows.append(t2)
    t3 = rng.standard_normal(V).astype(np.float32)
    t3[[2, 8]] = np.nan
    rows.append(t3)
    rows.append(np.zeros(V, dtype=np.float32))  # everything tied
    return rows


def _norm(entry):
    out = [entry["token_id"], np.float32(entry["logprob"]).tobytes()]
    for d in entry.get("top_logprobs", []):
        out.append((d["token_id"], np.float32(d["logprob"]).tobytes()))
    return out


@pytest.mark.parametrize("k", [None, 0, 1, 5, 20, V + 5])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_loop_logprob_entry_matches_the_full_sort(k, dtype):
    from yunshu_engine.text_logprobs import logprob_entry

    for row in _lp_rows():
        lp = mx.array(row).astype(dtype)
        assert _norm(old_entry(lp, 6, k)) == _norm(logprob_entry(lp, 6, k))
