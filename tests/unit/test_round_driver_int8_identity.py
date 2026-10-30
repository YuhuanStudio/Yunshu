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


def test_split_invariant_prefill_does_not_depend_on_the_kv_precision():
    """Prefill and the prefix cache are bf16 for every decode precision, so int8 rows need the same
    split-invariant prefill: with it off, a prefix-cache hit restores at a non-grid length and a row
    prefilled beside others splits its prompt differently from a solo one (27B 32K int8: first-token
    logprob 0.078 off, hit != miss and solo != concurrent)."""
    import inspect

    from yunshu_engine import vlm_engine

    src = inspect.getsource(vlm_engine)
    start = src.index("split_ready = bool(")
    end = src.index("gdn_prefill.configure_split(split_ready)")
    assert "YUNSHU_KV_PRECISION" not in src[start:end]


@pytest.mark.parametrize("fmt", ["bf16", "int8"])
def test_long_mtp_rows_spec_on_equals_spec_off_and_solo_equals_beside(fmt):
    """MTP rows (DecodeBatch) and the plain-verify batch read int8 KV through one arithmetic: spec on
    == spec off at several 512-key chunks, solo == beside other rows."""
    from yunshu_engine.kernels.ragged_attention import tile_ready

    from .test_round_driver import _tiny

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs M5-class tensor ops")
    lm, drafter = _tiny()
    prompts = [_prompt(n, k) for k, n in enumerate(LENS)]

    def toks(out):
        return {i: [t for t, _ in v] for i, v in out.items()}

    # the tiny MTP head's logprobs are not meaningful (nan), so tokens are compared; the arithmetic
    # itself is pinned bit for bit by test_int8_key_parallel_invariance (int8 tile == bf16 tile)
    off, _ = _run(lm, prompts[:1], None, "mtp", fmt=fmt)
    on, d = _run(lm, prompts[:1], drafter, "mtp", fmt=fmt)
    assert d.drafted > 0
    assert toks(on) == toks(off)
    beside, _ = _run(lm, prompts, drafter, "mtp", fmt=fmt)
    assert toks(beside)[0] == toks(on)[0]
