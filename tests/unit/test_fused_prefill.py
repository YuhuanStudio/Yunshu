"""Fused chunked prefill (decode rows + a prefill chunk in one forward).

A tiny random float32 Qwen3.5 decoder (GatedDeltaNet + attention layers) on
the CPU: the fused forward must give each segment the hidden states and cache
state it gets from its own forward. On the CPU a float32 matmul row does not
depend on how many rows the call has; on the GPU it does (M5 float32 GEMM vs
GEMV differ by ~1e-3), which would hide logic errors behind rounding.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from yunshu_engine import fused_prefill as fp  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    with mx.stream(mx.cpu):
        yield


def _tiny_lm():
    from mlx_vlm.models.qwen3_5.config import TextConfig

    mx.random.seed(0)
    args = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=96,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=8,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=101,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        head_dim=32,
        rope_parameters={
            "type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    lm = q35.LanguageModel(args)
    lm.set_dtype(mx.float32)
    mx.eval(lm.parameters())
    return lm


def _pos(start: int, length: int, rows: int = 1):
    p = mx.arange(start, start + length)[None, None, :]
    return mx.broadcast_to(p, (3, rows, length))


def _prefilled(lm, tokens, rows=1):
    """Caches holding ``tokens`` (same prompt in every row)."""
    if rows == 1:
        cache = lm.make_cache()
    else:
        from mlx_vlm.generate.ar import _make_cache

        cache = _make_cache(lm, [0] * rows)
    x = mx.array([tokens] * rows)
    lm.model(x, cache=cache, position_ids=_pos(0, len(tokens), rows))
    mx.eval([c.state for c in cache])
    return cache


def _states(cache):
    out = []
    for c in cache:
        st = c.state
        out.extend(st if isinstance(st, (list, tuple)) else [st])
    return [a for a in out if isinstance(a, mx.array)]


@pytest.mark.parametrize("rows", [1, 2])
def test_fused_forward_matches_separate_forwards(rows):
    lm = _tiny_lm()
    prompt_a = [3, 14, 15, 92, 65]
    prompt_c = [27, 18, 28, 18, 28, 45, 90, 45, 23, 53, 60, 28, 74]
    head = prompt_c[:6]
    chunk = prompt_c[6:]

    # reference: each segment through its own forward
    dec_ref = _prefilled(lm, prompt_a, rows)
    pre_ref = _prefilled(lm, head)
    dec_tok = mx.array([[35]] * rows)
    h_dec = lm.model(dec_tok, cache=dec_ref, position_ids=_pos(5, 1, rows))
    h_pre = lm.model(
        mx.array([chunk]), cache=pre_ref, position_ids=_pos(len(head), len(chunk))
    )
    mx.eval(h_dec, h_pre, _states(dec_ref), _states(pre_ref))

    # fused: same starting states, one forward
    dec = _prefilled(lm, prompt_a, rows)
    pre = _prefilled(lm, head)
    segs = [
        fp._Segment(
            cache=dec, inputs=dec_tok, inputs_embeds=None, position_ids=_pos(5, 1, rows)
        ),
        fp._Segment(
            cache=pre,
            inputs=mx.array([chunk]),
            inputs_embeds=None,
            position_ids=_pos(len(head), len(chunk)),
        ),
    ]
    f_dec, f_pre = fp.forward(lm.model, segs)
    mx.eval(f_dec, f_pre, _states(dec), _states(pre))

    assert f_dec.shape == h_dec.shape and f_pre.shape == h_pre.shape
    assert mx.allclose(f_dec, h_dec, atol=1e-5).item()
    assert mx.allclose(f_pre, h_pre, atol=1e-5).item()
    for got, ref in zip(
        _states(dec) + _states(pre), _states(dec_ref) + _states(pre_ref), strict=True
    ):
        assert got.shape == ref.shape
        assert mx.allclose(got, ref, atol=1e-5).item()
    # the model's projection modules are back in place
    for layer in lm.model.layers:
        mixer = layer.linear_attn if layer.is_linear else layer.self_attn
        for name in fp._IN["linear_attn" if layer.is_linear else "self_attn"] + (
            fp._OUT["linear_attn" if layer.is_linear else "self_attn"],
        ):
            assert not isinstance(mixer[name], (fp._Memo, fp._Defer))


def test_fused_forward_then_decode_continues():
    """After a fused step both caches keep decoding correctly."""
    lm = _tiny_lm()
    a, c = [5, 6, 7], [8, 9, 10, 11, 12, 13, 14]
    ref_a, ref_c = _prefilled(lm, a), _prefilled(lm, c[:3])
    lm.model(mx.array([[40]]), cache=ref_a, position_ids=_pos(3, 1))
    lm.model(mx.array([c[3:]]), cache=ref_c, position_ids=_pos(3, 4))
    ra = lm.model(mx.array([[41]]), cache=ref_a, position_ids=_pos(4, 1))
    rc = lm.model(mx.array([[42]]), cache=ref_c, position_ids=_pos(7, 1))

    fa, fc = _prefilled(lm, a), _prefilled(lm, c[:3])
    fp.forward(
        lm.model,
        [
            fp._Segment(fa, mx.array([[40]]), None, _pos(3, 1)),
            fp._Segment(fc, mx.array([c[3:]]), None, _pos(3, 4)),
        ],
    )
    ga = lm.model(mx.array([[41]]), cache=fa, position_ids=_pos(4, 1))
    gc = lm.model(mx.array([[42]]), cache=fc, position_ids=_pos(7, 1))
    assert mx.allclose(ga, ra, atol=1e-5).item()
    assert mx.allclose(gc, rc, atol=1e-5).item()


def test_supports_qwen3_5_only():
    assert fp.supports(_tiny_lm())
    assert not fp.supports(SimpleNamespace(model=object()))


def _prompt_batch(remaining, step, processed=0, right_pad=None, suffix=None, col=None):
    return SimpleNamespace(
        _inputs_embeds=mx.zeros((1, remaining, 4)),
        prefill_step_size=step,
        _right_pad_per_row=right_pad,
        _suffix_lens=suffix or [remaining],
        _processed_prompt_columns=processed,
        _next_apc_checkpoint_column=lambda: col,
    )


def test_prefill_chunk_mirrors_prompt_step():
    assert fp.prefill_chunk(_prompt_batch(1000, 256)) == 256
    # never the last token (generate() takes it)
    assert fp.prefill_chunk(_prompt_batch(100, 256)) == 99
    # stops at an APC checkpoint column
    assert fp.prefill_chunk(_prompt_batch(1000, 256, processed=512, col=600)) == 88
    # right-padded rows finish at a chunk boundary
    pb = _prompt_batch(1000, 256, processed=100, right_pad=[0, 500], suffix=[1000, 250])
    assert fp.prefill_chunk(pb) == 150


def test_set_chunk_uses_budget_only_while_rows_decode():
    pb = SimpleNamespace(prefill_step_size=2048)
    gen = SimpleNamespace(
        _generation_batch=[1, 2], _prompt_batch=pb, prefill_step_size=2048
    )
    assert fp.set_chunk(gen, 128, 2048)
    assert gen.prefill_step_size == 128 and pb.prefill_step_size == 128
    gen._generation_batch = []
    assert not fp.set_chunk(gen, 128, 2048)
    assert gen.prefill_step_size == 2048 and pb.prefill_step_size == 2048


def test_take_checks_the_call_and_finish_catches_a_missing_call():
    lm = _tiny_lm()
    a = _prefilled(lm, [1, 2, 3])
    c = _prefilled(lm, [4, 5])
    step = fp.FusedStep(
        lm.model,
        [
            fp._Segment(a, mx.array([[7]]), None, _pos(3, 1)),
            fp._Segment(c, mx.array([[8, 9]]), None, _pos(2, 2)),
        ],
    )
    assert step.take(lm.model, mx.array([[7]]), [None]) is None  # not planned
    with pytest.raises(RuntimeError):
        step.take(lm.model, mx.array([[7, 7]]), a)  # wrong shape
    assert step.take(lm.model, mx.array([[7]]), a) is not None
    assert not step.done()
    with pytest.raises(RuntimeError):
        fp.finish(step)  # computed, prefill call never came
    assert fp._STATE["step"] is None
    fp.finish(None)
