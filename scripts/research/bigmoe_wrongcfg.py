"""Tiny DeepSeek-V4 pack whose quantization config is wrong: tensors must decide.

Builds a random quantized tiny V4 checkpoint (2-bit routed experts, 4-bit elsewhere),
then writes it with the declared quantization (a) truthful, (b) top-level 4-bit only,
(c) pipenetwork style top-level bits=8/group 64 plus an ``expert_bits`` field, (d) no
quantization block at all, and loads each through ``VLMEngine._load_vision_model``.
Every variant's logits must equal the in-memory reference exactly.  Runs inside gpuq.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

VARIANTS = {
    "truthful": lambda q: q,
    "top_only_4bit": lambda q: {"bits": 4, "group_size": 32, "mode": "affine"},
    "expert_bits_style": lambda q: {"bits": 8, "group_size": 64, "expert_bits": 2},
    "no_block": lambda q: None,
}


def variant_config(base: dict, name: str, quant: dict) -> dict:
    cfg = {k: v for k, v in base.items() if k != "quantization"}
    declared = VARIANTS[name](quant)
    if declared is not None:
        cfg["quantization"] = declared
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.src)
    from dataclasses import asdict
    from types import SimpleNamespace

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_vlm.models.deepseek_v4.config import ModelConfig
    from mlx_vlm.models.deepseek_v4.deepseek_v4 import Model

    from yunshu_engine.vlm_engine import VLMEngine

    config = ModelConfig(
        vocab_size=128, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, n_routed_experts=8,
        num_experts_per_tok=2, q_lora_rank=32, head_dim=32, qk_rope_head_dim=16,
        o_groups=2, o_lora_rank=32, index_n_heads=4, index_head_dim=16,
        index_topk=8, compress_ratios=[0, 4, 128], num_hash_layers=0, hc_mult=4,
        sliding_window=128, index_block=0,
    )  # fmt: skip
    model = Model(config)
    quant = dict(bits=4, group_size=32, mode="affine")

    def predicate(path, module):
        if not hasattr(module, "to_quantized") or ".ffn.gate" in path:
            return False
        policy = dict(
            bits=2 if ".switch_mlp." in path else 4, group_size=32, mode="affine"
        )
        quant[path.removeprefix("language_model.")] = policy
        return policy

    nn.quantize(model, bits=4, group_size=32, class_predicate=predicate)
    model.eval()
    ids = mx.array([[1, 5, 9, 12, 33, 7]])
    reference = model(ids).logits
    mx.eval(reference)
    weights = {
        k.removeprefix("language_model."): v
        for k, v in tree_flatten(model.parameters())
    }
    rows = []
    ok = True
    for name in VARIANTS:
        with tempfile.TemporaryDirectory(prefix="bigmoe-wrongcfg-") as tmp:
            path = Path(tmp)
            base = {**asdict(config), "bos_token_id": 95, "eos_token_id": 96}
            cfg = variant_config(base, name, quant)
            (path / "config.json").write_text(json.dumps(cfg))
            mx.save_safetensors(str(path / "model.safetensors"), weights)
            try:
                loaded = VLMEngine._load_vision_model(SimpleNamespace(), str(path))
                logits = loaded(ids).logits
                mx.eval(logits)
                equal = bool(mx.array_equal(logits, reference).item())
                row = {"variant": name, "loaded": True, "bit_equal": equal}
            except Exception as exc:  # noqa: BLE001 - recorded, run fails closed
                equal = False
                row = {"variant": name, "loaded": False, "error": repr(exc)[:300]}
        ok = ok and equal
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps({"complete": True, "pass": ok, "rows": rows}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
