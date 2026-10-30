"""int8 KV identity at serving shapes: tokens AND logprobs, contexts past the 512-key chunk grid.

The 27B gates compare a digest of tokens plus logprobs, so these tests do too (a token-only
comparison at 40-key contexts hides summation-order differences).
"""

import pytest

from .test_round_driver_dflash import _draft, target  # noqa: F401  (fixtures)

mx = pytest.importorskip("mlx.core")

LENS = (700, 1300, 2100)


@pytest.fixture(autouse=True)
def one_atom_prefill_segments(monkeypatch):
    """The tiny model's prefill is not split-invariant (the real kernels are): a segment that merges
    atoms only when no row decodes would make solo and beside-others prefills differ in bf16 too.
    One atom per segment pins prefill, so these tests isolate the decode / verify KV path."""
    from yunshu_engine.round_driver.driver import RoundDriver

    monkeypatch.setattr(
        RoundDriver,
        "_run_end",
        lambda self, row, start, budget: self._span_end(row, start),
    )


def _prompt(n, seed):
    return [(seed * 7 + 11 * i + (i // 5)) % 500 for i in range(n)]


def _run(lm, prompts, drafter, mode, tokens=24, apc=None, stagger=False, fmt="int8"):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    kw = (
        dict(dflash_drafter=_draft(lm) if mode == "dflash" else None)
        if mode in ("dflash", "plain")
        else {}
    )
    d = RoundDriver(
        lm,
        stop_tokens=set(),
        chunk=64,
        kv_format=fmt,
        apc=apc,
        plain_verify=mode == "plain",
        drafter=drafter if mode == "mtp" else None,
        **kw,
    )
    out = {i: [] for i in range(len(prompts))}
    todo = list(enumerate(prompts))
    steps = 0
    while d.busy() or todo:
        if todo and (not stagger or steps % 4 == 0):
            i, p = todo.pop(0)
            d.add(Request(p, tokens, handle=i, logprobs=True, top_logprobs=2))
        for e in d.step():
            out[e.handle].append((e.token, e.logprob))
        steps += 1
        assert steps < 2000
    return out, d


@pytest.mark.parametrize("fmt", ["bf16", "int8"])
@pytest.mark.parametrize("mode", ["dflash", "plain"])
def test_long_rows_solo_equals_beside_others_with_logprobs(target, mode, fmt):  # noqa: F811
    prompts = [_prompt(n, k) for k, n in enumerate(LENS)]
    solo = [_run(target, [p], None, mode, fmt=fmt)[0][0] for p in prompts]
    for stagger in (False, True):
        beside, _ = _run(target, prompts, None, mode, stagger=stagger, fmt=fmt)
        for i in range(3):
            assert beside[i] == solo[i], f"{mode} {fmt} stagger={stagger} row {i}"


def test_int8_long_dflash_spec_on_equals_spec_off_with_logprobs(target):  # noqa: F811
    prompts = [_prompt(n, k) for k, n in enumerate(LENS)]
    on, _ = _run(target, prompts[:1], None, "dflash")
    off, _ = _run(target, prompts[:1], None, "plain")
    assert on == off


@pytest.mark.parametrize("fmt", ["bf16", "int8"])
def test_long_apc_hit_equals_miss_with_logprobs(target, fmt):  # noqa: F811
    from yunshu_engine.apc_manager import YunshuAPCManager
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    def manager():
        return YunshuAPCManager(
            num_blocks=256,
            block_size=16,
            overrides={
                "memory_max_gb": 1,
                "checkpoint_entries": 4,
                "checkpoint_interval_tokens": 64,
            },
        )

    prompt = _prompt(1500, 3)
    cold, _ = _run(target, [prompt], None, "dflash", fmt=fmt)
    apc = manager()
    first, d = _run(target, [prompt], None, "dflash", apc=apc, fmt=fmt)
    d.flush_checkpoints()
    d2 = RoundDriver(
        target,
        dflash_drafter=_draft(target),
        stop_tokens=set(),
        chunk=64,
        kv_format=fmt,
        apc=apc,
    )
    hit = d2.add(Request(prompt, 24, handle=0, logprobs=True, top_logprobs=2))
    out = []
    while d2.busy():
        out += [(e.token, e.logprob) for e in d2.step()]
    assert hit > 0
    assert first[0] == cold[0]
    assert out == cold[0]
