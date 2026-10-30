"""Round driver with int8 KV (row 17): every decode row's target KV is int8 codes + fp16 scales per
32-dim group, quantized once at join and on every write; verify windows read the same codes."""

import pytest

from .test_round_driver import PROMPTS, N, _tiny

mx = pytest.importorskip("mlx.core")


@pytest.fixture(scope="module")
def tiny():
    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs M5-class tensor ops")
    return _tiny()


def _tokens(lm, drafter, prompt, kv_format, others=(), steps_hook=None):
    from yunshu_engine.round_driver import driver as drv

    d = drv.RoundDriver(lm, drafter=drafter, stop_tokens=set(), kv_format=kv_format)
    d.add(drv.Request(prompt, N, handle="row"))
    for i, p in enumerate(others):
        d.add(drv.Request(p, N, handle=f"o{i}"))
    toks: dict = {}
    while d.busy():
        for e in d.step():
            toks.setdefault(e.handle, []).append(e.token)
        if steps_hook:
            steps_hook(d)
    return toks["row"], d


def test_int8_slots_hold_codes_and_scales_and_use_half_the_bytes(tiny):
    lm, _ = tiny
    seen = {}

    def hook(d):
        s = d.batch.slots
        if (
            s.k[3] is not None and "k" not in seen
        ):  # first full-attention layer of the tiny model
            seen["k"], seen["ks"] = s.k[3].dtype, s.ks[3].dtype
            seen["bytes"] = sum(a.nbytes for a in s.arrays())
            seen["shape"] = (s.k[3].shape, s.ks[3].shape)

    _tokens(lm, None, PROMPTS[1], "int8", steps_hook=hook)
    assert seen["k"] == mx.int8 and seen["ks"] == mx.float16
    (S, H, cap, D), (S2, H2, cap2, G) = seen["shape"]
    assert (S, H, cap) == (S2, H2, cap2) and G == D // 32

    seen16 = {}

    def hook16(d):
        s = d.batch.slots
        if s.k[3] is not None and "bytes" not in seen16:
            seen16["bytes"] = sum(a.nbytes for a in s.arrays())

    _tokens(lm, None, PROMPTS[1], "bf16", steps_hook=hook16)
    assert seen["bytes"] < 0.6 * seen16["bytes"]  # ~0.53x (codes + scales vs bf16)


def test_int8_solo_equals_int8_beside_other_rows(tiny):
    lm, _ = tiny
    solo, _ = _tokens(lm, None, PROMPTS[1], "int8")
    beside, _ = _tokens(lm, None, PROMPTS[1], "int8", others=PROMPTS[2:3] + PROMPTS[:1])
    assert beside == solo


def test_int8_spec_on_equals_spec_off_for_mtp_rows(tiny):
    lm, drafter = tiny
    plain, _ = _tokens(lm, None, PROMPTS[1], "int8")
    drafted, d = _tokens(lm, drafter, PROMPTS[1], "int8")
    assert drafted == plain
    assert d.drafted > 0  # the head really drafted against the int8 KV
    beside, _ = _tokens(lm, drafter, PROMPTS[1], "int8", others=PROMPTS[2:3])
    assert beside == plain


def test_int8_attention_is_close_to_bf16(tiny):
    """The quantization error of the decode path stays small: the final hidden state of the same row
    after the same tokens differs by a few percent (relative L2), not by an order of magnitude."""
    from yunshu_engine.round_driver import driver as drv

    lm, _ = tiny

    def last_hidden(kv_format):
        d = drv.RoundDriver(lm, stop_tokens=set(), kv_format=kv_format)
        d.add(drv.Request(PROMPTS[1], 8, handle="r"))
        while not d._pending_join and d.busy():
            d.step()
        d.step()  # joins the row into the decode batch (and runs a decode step)
        batch = d.batch
        row = batch.rows[0]
        hidden = batch.forward([1], mx.array([row.pending], dtype=mx.int32))
        return hidden.astype(mx.float32)

    a, b = last_hidden("bf16"), last_hidden("int8")
    rel = float(mx.linalg.norm(a - b) / mx.linalg.norm(a))
    assert rel < 0.15, rel
