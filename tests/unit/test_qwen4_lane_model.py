"""A tiny Qwen4-Exp target with quantized mixers, served by the lane matmul: speculative output equals plain
decode, rows are invariant to the verify width, and the result stays close to the stock singleton arithmetic."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx.utils import tree_flatten  # noqa: E402

from yunshu_engine import qwen4_mtp  # noqa: E402
from yunshu_engine.kernels import lane_linear, lane_virtual  # noqa: E402
from yunshu_engine.kernels.tensorfold.lane_qmm import ready  # noqa: E402

pytestmark = pytest.mark.skipif(
    not ready(), reason="lane tensor-op arithmetic self-test failed on this GPU"
)

MIXERS = ("in_proj_", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj")


def config(hidden=128):
    return dict(
        model_type="qwen4_exp",
        hidden_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=128,
        max_position_embeddings=4096,
        hc_count=2,
        hc_lowrank=16,
        ple_layer_ids=[],
        eos_token_id=127,
        ple_embed_dim=hidden,
        heads_per_ngram=2,
        indexer_n_heads=2,
        indexer_head_dim=64,
        indexer_budget=8,
        indexer_compress_ratio=4,
        layer_types=["linear_attention", "qwen_sparse_attention"],
    )


def build(bits=8, group=64):
    from mlx_vlm.models.qwen4_exp.config import ModelConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
        Qwen4ExpMTPDraftModel,
    )

    mx.random.seed(23)
    cfg = config()
    model_config = ModelConfig.from_dict(
        {
            "model_type": "qwen4_exp",
            "text_config": cfg,
            "vision_config": {
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": 32,
                "deepstack_visual_indexes": [],
            },
        }
    )
    target = LanguageModel(model_config.text_config, model_config)
    target.set_dtype(mx.bfloat16)
    nn.quantize(
        target,
        group_size=group,
        bits=bits,
        class_predicate=lambda path, m: (
            isinstance(m, nn.Linear) and any(tag in path for tag in MIXERS)
        ),
    )
    ref = Qwen4ExpMTPDraftModel(Qwen4ExpMTPConfig.from_dict({"text_config": cfg}))
    head = qwen4_mtp.load_native_head(
        {"text_config": cfg}, dict(tree_flatten(ref.parameters()))
    )
    mx.eval(target.parameters())
    return target, head


@pytest.fixture(autouse=True)
def reset_lane():
    yield
    lane_linear.set_stock_rows(0)


def plain_decode(target, prompt, steps):
    cache = target.make_cache()
    out = target(prompt, cache=cache)
    token = int(mx.argmax(out.logits[:, -1:], axis=-1).item())
    ids = [token]
    for _ in range(steps - 1):
        out = target._batch_invariant_decode(mx.array([[token]]), cache=cache)
        token = int(mx.argmax(out.logits[:, -1:], axis=-1).item())
        ids.append(token)
    return ids


@pytest.mark.parametrize("bits,group", [(8, 64), (5, 128)])
def test_lane_projections_keep_speculative_output_equal_to_plain_decode(bits, group):
    from mlx_vlm.speculative.utils import run_speculative_rounds

    target, head = build(bits, group)
    prompt = mx.array([[1, 3, 5, 7, 9, 11]], dtype=mx.int32)
    report = qwen4_mtp.lane_projections(target, head=False)
    assert report["lane_linear"] >= 6
    assert any(isinstance(m, lane_linear.LaneLinear) for _, m in target.named_modules())
    plain = plain_decode(target, prompt, 20)
    assert lane_virtual.STATS["launches"] > 0  # the mixers' projections shared launches
    for depth in (2, 3, 4):
        cache = target.make_cache()
        out = target(prompt, cache=cache, return_hidden=True)
        first = mx.argmax(out.logits[:, -1:], axis=-1)
        ids = [
            int(t)
            for t, _ in run_speculative_rounds(
                target,
                head,
                cache,
                prompt,
                first,
                out.logits[:, -1:],
                out,
                draft_kind="mtp",
                max_tokens=20,
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                draft_block_size=depth + 1,
                sampler_is_greedy=True,
            )
        ]
        assert ids == plain, depth


def test_verify_rows_are_invariant_to_the_window_width_and_close_to_stock():
    target, _ = build(8, 64)
    prompt = mx.array([[2, 4, 6, 8, 10, 12, 14]], dtype=mx.int32)
    window = mx.array([[20, 22, 24, 26, 28, 30, 32, 34]], dtype=mx.int32)

    def hidden(width):
        cache = target.make_cache()
        target(prompt, cache=cache)
        h, _, tx = target.speculative_verify_hidden(window[:, :width], cache)
        tx.abort()
        return h

    stock = {w: hidden(w) for w in (1, 4, 8)}
    qwen4_mtp.lane_projections(target, head=False)
    lane = {w: hidden(w) for w in (1, 4, 8)}
    for w in (1, 4):
        assert mx.array_equal(lane[w], lane[8][:, :w]).item(), w
    ref = stock[8].astype(mx.float32)
    err = mx.max(mx.abs(lane[8].astype(mx.float32) - ref)) / mx.max(mx.abs(ref))
    assert float(err.item()) < 0.1
