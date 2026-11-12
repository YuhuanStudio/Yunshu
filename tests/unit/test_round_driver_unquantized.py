"""Packed Qwen3.5 layer on an UNQUANTIZED checkpoint (plain bf16 Linear, dense LM head): a greedy row decodes the
same tokens alone, beside other rows, joining mid-flight, with MTP drafts or without. The verifier runs plain
Linear one token at a time and the LM head one row at a time, so the window width never changes a row's bits.
Needs the tile kernels (Metal), like the quantized fixture of test_round_driver.py."""

import pytest

mx = pytest.importorskip("mlx.core")

PROMPTS = [
    list(range(3, 8)),
    [(7 * i + 11) % 500 for i in range(300)],
    [(13 * i + 5) % 500 for i in range(40)],
]
N = 20


@pytest.fixture(autouse=True)
def one_atom_prefill_segments(monkeypatch):
    """The tiny model's prefill is not split-invariant (the 27B's NAX / BQ32 prefill is): one atom per
    segment pins prefill so the test isolates the decode / verify forwards."""
    from yunshu_engine.round_driver.layer import PackedLayer

    monkeypatch.setattr(
        PackedLayer,
        "span_end",
        lambda self, row, start, budget, *, atom, scoring=False: self.cut(
            row, start, atom
        ),
    )


@pytest.fixture(scope="module")
def tiny():
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5.language import LanguageModel
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
        Qwen3_5MTPDraftModel,
    )
    from scripts.research.phaseb_verify import tiny_config

    from yunshu_engine.kernels import (
        gdn_prefill,
        lane_linear,
        omlx,
        ragged_attention,
        ragged_kv,
    )
    from yunshu_engine.kernels.batch_invariant import install, set_active

    if not mx.metal.is_available() or not ragged_attention.tile_ready():
        pytest.skip("needs the tile kernels")
    omlx.apply()
    mx.random.seed(23)
    lm = LanguageModel(tiny_config())
    lm.set_dtype(
        mx.bfloat16
    )  # unquantized: no nn.quantize, nothing for the lane to convert
    assert not any(isinstance(m, nn.QuantizedLinear) for _, m in lm.named_modules())
    assert lane_linear.convert(lm)["converted"] == 0
    install(lm, packed=False)
    set_active(True)
    gdn_prefill.install()
    ragged_kv.install()
    ragged_kv.set_dense_lane(False)
    mx.eval(lm.parameters())
    cfg = dict(
        model_type="qwen3_5",
        hidden_size=256,
        intermediate_size=512,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=512,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        head_dim=256,
        tie_word_embeddings=False,
    )
    mx.random.seed(3)
    drafter = Qwen3_5MTPDraftModel(
        Qwen3_5MTPConfig(text_config={**cfg, "mtp_num_hidden_layers": 1})
    )
    drafter.set_dtype(mx.bfloat16)
    mx.eval(drafter.parameters())
    return lm, drafter


def _run(lm, drafter, prompts, stagger=False):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(lm, drafter=drafter, stop_tokens=set())
    out = {i: [] for i in range(len(prompts))}
    todo = list(range(len(prompts)))
    if not stagger:
        for i in todo:
            d.add(Request(prompts[i], N, handle=i))
        todo = []
    steps = 0
    while d.busy() or todo:
        if todo and steps % 2 == 0:
            i = todo.pop(0)
            d.add(Request(prompts[i], N, handle=i))
        for e in d.step():
            out[e.handle].append(e.token)
        steps += 1
    return [out[i] for i in range(len(prompts))]


def test_unquantized_rows_are_the_same_alone_together_and_staggered(tiny):
    lm, _ = tiny
    from yunshu_engine import tree_verify

    assert tree_verify.lane_projections(lm)
    ref = [_run(lm, None, [p])[0] for p in PROMPTS]
    assert all(len(r) == N for r in ref)
    assert _run(lm, None, PROMPTS) == ref
    assert _run(lm, None, PROMPTS, stagger=True) == ref


def test_unquantized_mtp_drafting_equals_plain_decode(tiny):
    lm, drafter = tiny
    ref = [_run(lm, None, [p])[0] for p in PROMPTS]
    assert _run(lm, drafter, PROMPTS) == ref
    assert _run(lm, drafter, PROMPTS, stagger=True) == ref
