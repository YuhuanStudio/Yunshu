"""Small same-device greedy repeatability smoke; no timing verdict."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.generate import generate_step
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(args.model)
    prompt = apply_chat_template(
        processor, model.config, "Reply briefly: 17 + 25 = ?", num_images=0
    )
    encoded = processor(text=prompt, return_tensors="mlx")
    results = []
    for _ in range(2):
        tokens = []
        for token, _ in generate_step(
            encoded["input_ids"],
            model,
            pixel_values=None,
            mask=None,
            max_tokens=args.max_tokens,
            temperature=0,
        ):
            tokens.append(int(token))
        mx.eval(model.parameters())
        results.append(tokens)
    same = results[0] == results[1] and bool(results[0])
    row = dict(
        complete=True,
        success=same,
        parity=same,
        device=os.environ.get("GPUQ_DEVICE", "m5"),
        model=Path(args.model).name,
        tokens=results,
        digest=hashlib.sha256(json.dumps(results[0]).encode()).hexdigest(),
        evidence="portability evidence (not M5)",
    )
    Path(args.out).write_text(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    if not same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
