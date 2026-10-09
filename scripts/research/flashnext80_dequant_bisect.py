"""Which tensor class breaks stock mlx-vlm on the oQ4e pack?  Cumulative in-memory dequantization.

Loads the pack once, then for each class (in order) replaces its QuantizedLinear/QuantizedEmbedding modules
with dense bf16 modules holding the dequantized weights and generates an answer.  The first class after
which the answer contains "Paris" is (part of) the culprit.  Experts are never touched.

Run through gpuq only.
"""

import argparse
import json
import re
import sys
from pathlib import Path

CLASSES = (
    ("embed_head", r"(embed_tokens|lm_head)$"),
    ("hyper_conn", r"(hyper_connection|hyper_connection_mixer)\."),
    ("gdn", r"linear_attn\."),
    ("attn", r"self_attn\."),
    ("shared_expert", r"shared_expert"),
    ("ple_proj", r"\.ple\.(key_proj|value_proj)"),
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(a.model)
    prompt = apply_chat_template(
        processor,
        model.config,
        "What is the capital of France? Answer in one word.",
        num_images=0,
        enable_thinking=False,
    )
    rows = []

    def ask(tag, extra=None):
        res = generate(
            model, processor, prompt, max_tokens=16, temperature=0.0, verbose=False
        )
        text = getattr(res, "text", str(res))
        rows.append(
            {
                "kind": "answer",
                "after": tag,
                "text": text,
                "coherent": "Paris" in text,
                **(extra or {}),
            }
        )
        print(rows[-1], flush=True)
        return "Paris" in text

    ask("none")
    lm = model.language_model
    done = False
    for name, pattern in CLASSES:
        rx = re.compile(pattern)
        n = 0
        for path, mod in list(lm.named_modules()):
            if not rx.search(path):
                continue
            if isinstance(mod, nn.QuantizedLinear):
                w = mx.dequantize(
                    mod.weight,
                    mod.scales,
                    mod.biases,
                    mod.group_size,
                    mod.bits,
                    mode=getattr(mod, "mode", "affine"),
                )
                dense = nn.Linear(w.shape[1], w.shape[0], bias="bias" in mod)
                dense.weight = w.astype(mx.bfloat16)
                if "bias" in mod:
                    dense.bias = mod.bias
            elif isinstance(mod, nn.QuantizedEmbedding):
                w = mx.dequantize(
                    mod.weight,
                    mod.scales,
                    mod.biases,
                    mod.group_size,
                    mod.bits,
                    mode=getattr(mod, "mode", "affine"),
                )
                dense = nn.Embedding(w.shape[0], w.shape[1])
                dense.weight = w.astype(mx.bfloat16)
            else:
                continue
            parent_path, _, leaf = path.rpartition(".")
            parent = lm
            for part in parent_path.split(".") if parent_path else []:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            if leaf.isdigit():
                parent[int(leaf)] = dense
            else:
                setattr(parent, leaf, dense)
            n += 1
        mx.eval(lm.parameters())
        if ask(f"+{name}", {"modules_dequantized": n}):
            done = True
            break
    rows.append({"kind": "result", "fixed_by": rows[-1]["after"] if done else None})
    a.out.write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
        + json.dumps({"complete": True})
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
