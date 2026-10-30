"""One arithmetic for every decode row: a one-token DecodeBatch step equals a one-token tree_forward step
(the DFlash batch's plain verify), bit for bit, so spec on == spec off whichever batch a row is in."""

import pytest

from .test_round_driver_dflash import target  # noqa: F401  (fixture)

mx = pytest.importorskip("mlx.core")

PROMPT = [(7 * i + 11) % 500 for i in range(150)]


def _joined(lm, plain_verify):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(lm, plain_verify=plain_verify, stop_tokens=set(), chunk=64)
    d.add(Request(PROMPT, 8, handle=0))
    steps = 0
    while not d.batch.rows:
        d.step()
        steps += 1
        assert steps < 50
    return d


@pytest.mark.parametrize("fmt", ["bf16"])
def test_decode_batch_one_token_step_equals_tree_forward_bit_for_bit(target, fmt):  # noqa: F811
    from yunshu_engine import tree_verify as tv
    from yunshu_engine.round_driver.batch import DecodeBatch

    a = _joined(target, plain_verify=False)
    assert isinstance(a.batch, DecodeBatch)
    b = _joined(target, plain_verify=True)
    assert b.dflash is not None
    row_a, row_b = a.batch.rows[0], b.batch.rows[0]
    assert row_a.pending == row_b.pending and row_a.n == row_b.n
    hidden_a = a.batch.forward([1], mx.array([row_a.pending], dtype=mx.int32))
    res = tv.tree_forward(
        target,
        mx.array([[row_b.pending]], dtype=mx.int32),
        tv.TreeShape([-1]),
        row_b.cache,
    )
    hidden_b = res.hidden[0]
    mx.eval(hidden_a, hidden_b)
    assert mx.array_equal(hidden_a, hidden_b)
    tv.tree_abort(row_b.cache, res)


@pytest.fixture(scope="module")
def wide():
    """A wider tiny Qwen3.5 (1024 hidden, 8 layers, 6 GDN) so that rare bf16 rounding differences between two
    arithmetics show up within a few rounds."""
    from scripts.research import phaseb_verify as pv

    from yunshu_engine.kernels import ragged_kv
    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs M5-class tensor ops")
    from mlx_vlm.models.qwen3_5.config import TextConfig

    def config():
        return TextConfig(
            model_type="qwen3_5",
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
            max_position_embeddings=4096,
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_value_heads=8,
            linear_num_key_heads=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            vocab_size=512,
            tie_word_embeddings=False,
        )

    previous = ragged_kv._STATE["dense_lane"]
    original = pv.tiny_config
    pv.tiny_config = config
    try:
        lm = pv.tiny()[0]
    finally:
        pv.tiny_config = original
    yield lm
    ragged_kv.set_dense_lane(previous)


def test_decode_batch_rounds_equal_tree_forward_rounds_bit_for_bit(wide):
    from yunshu_engine import tree_verify as tv

    a = _joined(wide, plain_verify=False)
    b = _joined(wide, plain_verify=True)
    row_b = b.batch.rows[0]
    for step in range(10):
        token = (7 * step + 3) % 500
        hidden_a = a.batch.forward([1], mx.array([token], dtype=mx.int32))
        res = tv.tree_forward(
            wide, mx.array([[token]], dtype=mx.int32), tv.TreeShape([-1]), row_b.cache
        )
        hidden_b = res.hidden[0]
        mx.eval(hidden_a, hidden_b)
        assert mx.array_equal(hidden_a, hidden_b), f"round {step}"
        a.batch.commit([1])
        tv.tree_commit(wide, row_b.cache, res, [0])
        mx.eval(a.batch.arrays())
