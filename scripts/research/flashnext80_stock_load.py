"""Does stock mlx-vlm load + generate coherent text from a pack?  (bypasses Yunshu's loader)

Run through gpuq only:  flashnext80_stock_load.py --model PACK --out F.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quantize-gate", action="store_true")
    a = ap.parse_args(argv)
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(a.model)
    lm = model.language_model
    rows = []
    gate = lm.model.layers[0].mlp.gate
    rows.append(
        {
            "kind": "gate",
            "type": type(gate).__name__,
            "weight": [str(gate.weight.dtype), list(gate.weight.shape)],
            "layer0_types": {
                n: type(m).__name__
                for n, m in lm.model.layers[0].named_modules()
                if n.count(".") <= 2 and "mlp" in n
            },
        }
    )
    cfg = json.loads((Path(a.model) / "config.json").read_text())
    quant = cfg["quantization"]
    default = (quant["bits"], quant["group_size"])
    mismatches, counts = [], {}
    for path, mod in model.named_modules():
        if hasattr(mod, "bits") and hasattr(mod, "group_size"):
            exp = quant.get(path)
            exp = (exp["bits"], exp["group_size"]) if isinstance(exp, dict) else default
            got = (mod.bits, mod.group_size)
            counts[f"{got}"] = counts.get(f"{got}", 0) + 1
            if got != exp:
                mismatches.append([path, got, exp])
    rows.append(
        {
            "kind": "quant_audit",
            "quantized_modules": sum(counts.values()),
            "by_bits_group": counts,
            "mismatches": len(mismatches),
            "first": mismatches[:12],
        }
    )
    prompt = apply_chat_template(
        processor,
        model.config,
        "What is the capital of France? Answer in one word.",
        num_images=0,
        enable_thinking=False,
    )
    res = generate(
        model, processor, prompt, max_tokens=24, temperature=0.0, verbose=False
    )
    text = getattr(res, "text", str(res))
    row = {"kind": "stock_generate", "text": text}
    rows.append(row)
    if a.quantize_gate:
        for layer in lm.model.layers:
            g = layer.mlp.gate
            if isinstance(g, nn.Linear) and not isinstance(g, nn.QuantizedLinear):
                layer.mlp.gate = nn.QuantizedLinear.from_linear(g, 64, 8)
        mx.eval(lm.parameters())
        res = generate(
            model, processor, prompt, max_tokens=24, temperature=0.0, verbose=False
        )
        rows.append(
            {"kind": "gate_q8_generate", "text": getattr(res, "text", str(res))}
        )
    a.out.write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
        + json.dumps({"complete": True})
        + "\n"
    )
    print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
