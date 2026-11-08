"""Flash-Next packed chains preserve HC/QSA/PLE and routed expert arithmetic."""

import copy

import mlx.core as mx
import pytest

from yunshu_engine.round_driver.verify import native_chain_forward


@pytest.mark.parametrize("abort", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("context", [4, 10, 11, 16])
def test_native_chain_hidden_commit_and_abort_match_solo(rows, context, abort):
    from mlx_vlm.models.qwen4_exp.config import ModelConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    from scripts.research.qwen4_mtp_probe import tiny_config

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(17)
        cfg = ModelConfig.from_dict(
            {
                "model_type": "qwen4_exp",
                "text_config": tiny_config(),
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
        import mlx.nn as nn

        lm = LanguageModel(cfg.text_config, cfg)
        lm.set_dtype(mx.bfloat16)
        nn.quantize(
            lm,
            group_size=32,
            bits=4,
            class_predicate=lambda p, m: (
                hasattr(m, "to_quantized") and m.weight.shape[-1] % 32 == 0
            ),
        )
        prefixes = []
        for b in range(rows):
            cache = lm.make_cache()
            lm(mx.array([[1 + (i + b) % 60 for i in range(context + b)]]), cache=cache)
            mx.eval([c.state for c in cache])
            prefixes.append(cache)
        tokens = [mx.array([[8 + b, 9 + b]][:1])[:, : 1 + b % 2] for b in range(rows)]
        left, right = copy.deepcopy(prefixes), copy.deepcopy(prefixes)
        solo = [
            lm.speculative_verify_hidden(t, c)
            for t, c in zip(tokens, left, strict=True)
        ]
        packed = native_chain_forward(lm, tokens, right)
        for b, ((ref, _, transaction), (got, packed_transaction)) in enumerate(
            zip(solo, packed, strict=True)
        ):
            assert mx.array_equal(ref, got).item()
            assert mx.array_equal(
                lm.speculative_logits_from_hidden(ref),
                lm.speculative_logits_from_hidden(got),
            ).item()
            if abort:
                transaction.abort()
                packed_transaction.abort()
            else:
                transaction.commit(1)
                packed_transaction.commit(1)
            # The next token exercises QSA indexer, PLE history/conv and GDN
            # recurrence as well as the visible append-only KV state.
            probe = mx.array([[21 + b]])
            a = lm(probe, cache=left[b]).logits
            z = lm(probe, cache=right[b]).logits
            assert mx.array_equal(a, z).item()
    finally:
        mx.set_default_device(previous)


def test_native_chain_rejects_alias_before_touching_state():
    cache = []
    with pytest.raises(ValueError, match="independent"):
        native_chain_forward(None, [mx.array([[1]]), mx.array([[2]])], [cache, cache])


def test_native_probe_cpu_plan_and_small_array_schema():
    from scripts.research.phaseb_verify import parser, plan, run_native_tiny

    a = parser().parse_args(
        ["--native-tiny", "--context", "11", "--rows", "2", "--output", "unused"]
    )
    assert plan(a) == [dict(rows=2, nodes="1/2 chain", context=11, native=True)]
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rows = []
        run_native_tiny(a, rows.append)
        assert len(rows) == 5
        assert rows[-1]["complete"] and rows[-1]["success"]
        assert all(all(r["checks"].values()) for r in rows[:-1])
    finally:
        mx.set_default_device(previous)
