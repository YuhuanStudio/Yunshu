"""Tiny Qwen4 HC/QSA/GDN identity and rejection rollback; run via yv/gpuq."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def tiny_config():
    return dict(
        model_type="qwen4_exp",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=32,
        moe_intermediate_size=32,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=4096,
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[1],
        eos_token_id=63,
        ngram_vocab_size_base=64,
        split_ngram_parts=8,
        ple_embed_dim=32,
        heads_per_ngram=2,
        indexer_n_heads=2,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        layer_types=["linear_attention", "qwen_sparse_attention"],
    )


def probe():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen4_exp.config import ModelConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    from mlx_vlm.speculative.cache_state import rollback_speculative_cache
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
        Qwen4ExpMTPDraftModel,
    )
    from mlx_vlm.speculative.utils import run_speculative_rounds

    from yunshu_engine.qwen4_mtp import load_native_head

    mx.random.seed(17)
    config = tiny_config()
    model_config = ModelConfig.from_dict(
        {
            "model_type": "qwen4_exp",
            "text_config": config,
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
    ref = Qwen4ExpMTPDraftModel(Qwen4ExpMTPConfig.from_dict({"text_config": config}))
    head = load_native_head(
        {"text_config": config}, dict(tree_flatten(ref.parameters()))
    )
    prompt = mx.array([[1, 3, 5, 7, 9]], dtype=mx.int32)

    def sample(logits):
        return mx.argmax(logits, axis=-1)

    plain_cache = target.make_cache()
    out = target(prompt, cache=plain_cache)
    token = int(sample(out.logits[:, -1:]).item())
    plain = [token]
    for _ in range(23):
        out = target._batch_invariant_decode(mx.array([[token]]), cache=plain_cache)
        token = int(sample(out.logits[:, -1:]).item())
        plain.append(token)
    checks = []
    for depth in (2, 3, 4):
        cache = target.make_cache()
        out = target(prompt, cache=cache, return_hidden=True)
        first = sample(out.logits[:, -1:])
        ids = [
            int(token)
            for token, _ in run_speculative_rounds(
                target,
                head,
                cache,
                prompt,
                first,
                out.logits[:, -1:],
                out,
                draft_kind="mtp",
                max_tokens=24,
                sampler=sample,
                draft_block_size=depth + 1,
                sampler_is_greedy=True,
            )
        ]
        checks.append(
            dict(
                check=f"spec-depth-{depth}",
                passed=ids == plain,
                tokens=len(ids),
                accepted=list(head.accept_lens),
                drafted=list(head.draft_lens),
            )
        )
        if ids != plain:
            return {"passed": False, "checks": checks, "plain": plain, "spec": ids}
    # Every rejection boundary must resume with exactly the serial next token.
    for kept in (1, 2, 3, 4):
        serial_cache, verify_cache = target.make_cache(), target.make_cache()
        target(prompt, cache=serial_cache)
        target(prompt, cache=verify_cache)
        proposed = mx.array([[11, 13, 15, 17]])
        for i in range(kept):
            target._batch_invariant_decode(proposed[:, i : i + 1], cache=serial_cache)
        _, _, transaction = target.speculative_verify_hidden(proposed, verify_cache)
        rollback_speculative_cache(verify_cache, transaction, kept - 1, 4)
        next_input = mx.array([[19]])
        a = target._batch_invariant_decode(next_input, cache=serial_cache).logits
        b = target._batch_invariant_decode(next_input, cache=verify_cache).logits
        bit_equal = bool(mx.array_equal(a, b).item())
        same_token = bool(
            mx.array_equal(mx.argmax(a, axis=-1), mx.argmax(b, axis=-1)).item()
        )
        equal = same_token and bool(mx.allclose(a, b, rtol=1e-6, atol=1e-6).item())
        checks.append(
            dict(
                check=f"rollback-{kept}",
                passed=equal,
                bit_equal=bit_equal,
                max_abs=float(mx.max(mx.abs(a - b)).item()),
                same_token=same_token,
            )
        )
        if not equal:
            return {"passed": False, "checks": checks}
    return {
        "passed": True,
        "checks": checks,
        "device": "cpu" if mx.default_device() == mx.cpu else "m5",
        "engaged": "qwen4-native",
    }


def main(argv=None):
    args = parser().parse_args(argv)
    if args.dry_run:
        result = {"dry_run": True, "passed": True, "config": tiny_config()}
    else:
        result = probe()
    result["complete"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
