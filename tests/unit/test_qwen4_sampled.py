"""Qwen4 native MTP loop: sampled requests keep speculation, losslessly, with exact reports."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.constrained_spec import SpecRequest, normalize_rows
from yunshu_engine.keyed_sampling import KeyedSampler
from yunshu_engine.qwen4_sampled import NativeKeyedSampler, install_logit_stash
from yunshu_engine.vlm_batch_runner import _DONE, RowParams, VLMBatchRunner

VOCAB = 40


def params(**kw):
    base = dict(temperature=0.9, top_p=0.95, top_k=0, min_p=0.0)
    base.update(kw)
    return RowParams(**base)


def table():
    return np.random.default_rng(3).normal(size=(VOCAB, VOCAB)).astype(np.float32) * 2.0


def serial(tbl, sampler, first_logits, n):
    """Plain decode: one keyed draw per position from one-row logprobs."""
    out, rows = [], []
    logits = mx.array(first_logits)[None]
    for pos in range(n):
        row = normalize_rows(logits[None])
        tok = int(sampler.sample_positions(row, [pos])[0].item())
        out.append(tok)
        rows.append(row)
        logits = mx.array(tbl[tok])[None]
    return out, rows


def speculate(tbl, sampler, first_logits, n, block=3, lm=None, draft_scale=1.0):
    """The upstream native loop's calls on a toy Markov target and a perfect drafter."""
    lm = lm or SimpleNamespace(_yunshu_logit_stash=None)
    sampler.lm = lm
    row = normalize_rows(mx.array(first_logits)[None, None])
    first = int(sampler.sample_target(row, row_ids=[0], positions=[0])[0].item())
    out, emitted, bonus = [first], 1, first
    drafted_total = accepted_total = 0
    while emitted < n:
        drafts, prev = [], bonus
        for _ in range(block - 1):
            tok = int(
                sampler(mx.array(tbl[prev] * draft_scale)[None, None])
                .reshape(-1)[0]
                .item()
            )
            drafts.append(tok)
            prev = tok
        window = [bonus, *drafts]
        logits = mx.array(tbl[window])[None]  # [1, T, V]
        lm._yunshu_logit_stash = logits
        rows = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target = sampler.sample_target(
            rows[0], row_ids=[0] * block, positions=[emitted + i for i in range(block)]
        ).tolist()
        accepted = 0
        while accepted < len(drafts) and drafts[accepted] == target[accepted]:
            accepted += 1
        out.extend(target[: accepted + 1])
        emitted += accepted + 1
        bonus = target[accepted]
        drafted_total += len(drafts)
        accepted_total += accepted
    return out[:n], drafted_total, accepted_total


def test_drafts_use_the_noise_of_their_own_position():
    tbl, first = table(), table()[0]
    ref, _ = serial(tbl, KeyedSampler(params(), 11), first, 200)
    got, drafted, accepted = speculate(
        tbl, NativeKeyedSampler(params(), 11), first, 200, draft_scale=1.0
    )
    assert got == ref  # lossless: the committed stream is the serial keyed stream
    assert accepted == drafted  # equal distributions with coupled noise always agree
    # A drafter that is only close: a rejection moves the plain sampler's serial cursor off the
    # positions the next drafts occupy, so its drafts lose the coupling.
    close = dict(draft_scale=0.8)
    got, drafted, accepted = speculate(
        tbl, NativeKeyedSampler(params(), 11), first, 200, **close
    )
    plain_got, plain_drafted, plain_accepted = speculate(
        tbl, KeyedSampler(params(), 11), first, 200, **close
    )
    assert got == plain_got == ref
    assert accepted / drafted > 0.5
    assert accepted / drafted > plain_accepted / plain_drafted + 0.1


@pytest.mark.parametrize("top_k", [0, 8])
def test_stream_does_not_depend_on_block(top_k):
    tbl, first = table(), table()[5]
    ref, _ = serial(tbl, KeyedSampler(params(top_k=top_k), 99), first, 48)
    for block in (2, 3, 5):
        got, _, _ = speculate(
            tbl, NativeKeyedSampler(params(top_k=top_k), 99), first, 48, block=block
        )
        assert got == ref


def test_reports_are_the_serial_float32_logprobs():
    tbl, first = table(), table()[1]
    request = SpecRequest(logprobs=True, top_logprobs=0)
    sampler = NativeKeyedSampler(params(), 5, request=request)
    got, _, _ = speculate(tbl, sampler, first, 40)
    ref, rows = serial(tbl, KeyedSampler(params(), 5), first, 40)
    assert got == ref
    # speculation computed a few rows past the end; the reported ones are those of the stream
    for index in range(1, 38):
        lp, top = sampler.report(index, got[index])
        want = float(rows[index].astype(mx.float32)[0, got[index]].item())
        assert lp == want and top == []


def test_top_logprobs_follow_the_request_order():
    tbl, first = table(), table()[2]
    request = SpecRequest(logprobs=True, top_logprobs=3)
    sampler = NativeKeyedSampler(params(), 8, request=request)
    got, _, _ = speculate(tbl, sampler, first, 12)
    _, rows = serial(tbl, KeyedSampler(params(), 8), first, 12)
    lp, top = sampler.report(4, got[4])
    want = request.reports(rows[4], [got[4]])[0]
    assert (got[4], lp, top) == want
    assert len(top) == 3


def test_take_reads_native_reports_after_the_first_token():
    tbl, first = table(), table()[2]
    request = SpecRequest(logprobs=True)
    sampler = NativeKeyedSampler(params(), 8, request=request)
    request.native = sampler
    got, _, _ = speculate(tbl, sampler, first, 10)
    row = normalize_rows(mx.array(first)[None, None])
    request.first_probs = row  # captured from the prefill sampler
    lps = [request.take(t)[0] for t in got[:8]]
    _, rows = serial(tbl, KeyedSampler(params(), 8), first, 8)
    want = [float(r[0, t].item()) for r, t in zip(rows, got, strict=False)]
    assert lps == want


def test_logit_stash_widens_for_logprob_requests_and_remembers_the_window():
    from yunshu_engine import constrained_spec

    lm = SimpleNamespace(speculative_logits_from_hidden=lambda h: h.astype(mx.bfloat16))
    assert install_logit_stash(lm) and install_logit_stash(lm)
    hidden = mx.ones((1, 2, 4))
    constrained_spec.set_request(None)
    assert lm.speculative_logits_from_hidden(hidden).dtype == mx.bfloat16
    constrained_spec.set_request(SpecRequest(logprobs=True))
    try:
        out = lm.speculative_logits_from_hidden(hidden)
        assert out.dtype == mx.float32 and lm._yunshu_logit_stash is out
    finally:
        constrained_spec.set_request(None)


def _runner(monkeypatch, native, kind="mtp"):
    drafter = SimpleNamespace(supports_greedy_draft_argmax=False, native_keyed=native)
    runner = VLMBatchRunner(
        SimpleNamespace(language_model=SimpleNamespace()),
        None,
        drafter=drafter,
        draft_kind=kind,
    )
    captured = []

    def submit(job):
        captured.append(job)
        job.out.put(_DONE)

    monkeypatch.setattr(runner, "_submit", submit)
    return runner, captured


def test_sampled_logprob_request_keeps_native_speculation(monkeypatch):
    runner, captured = _runner(monkeypatch, True)
    list(
        runner.iter_tokens([1, 2], max_tokens=3, temperature=0.8, seed=4, logprobs=True)
    )
    job = captured[0]
    assert job.use_draft and isinstance(job.keyed, NativeKeyedSampler)


def test_sampled_request_without_logprobs_uses_coupled_drafts(monkeypatch):
    runner, captured = _runner(monkeypatch, True)
    list(runner.iter_tokens([1, 2], max_tokens=3, temperature=0.8, seed=4))
    assert isinstance(captured[0].keyed, NativeKeyedSampler)


def test_other_paths_keep_their_gates(monkeypatch):
    runner, captured = _runner(monkeypatch, True)
    list(runner.iter_tokens([1, 2], max_tokens=3, logprobs=True))  # greedy
    assert not captured[0].use_draft
    runner, captured = _runner(monkeypatch, False)  # not the native loop
    list(
        runner.iter_tokens([1, 2], max_tokens=3, temperature=0.8, seed=4, logprobs=True)
    )
    assert not captured[0].use_draft


def test_greedy_admission_never_runs_the_sampled_setup(monkeypatch):
    import yunshu_engine.qwen4_sampled as sampled

    calls = []
    monkeypatch.setattr(sampled, "install_logit_stash", lambda lm: calls.append(lm))
    runner, captured = _runner(monkeypatch, True)
    list(runner.iter_tokens([1, 2], max_tokens=3))
    list(runner.iter_tokens([1, 2], max_tokens=3, logprobs=True))
    assert calls == [] and captured[0].keyed is None
    list(runner.iter_tokens([1, 2], max_tokens=3, temperature=0.8, seed=4))
    assert len(calls) == 1


def test_loading_the_lane_does_not_wrap_the_target_logits():
    import inspect

    from yunshu_engine import qwen4_mtp

    assert "install_logit_stash" not in inspect.getsource(qwen4_mtp)
