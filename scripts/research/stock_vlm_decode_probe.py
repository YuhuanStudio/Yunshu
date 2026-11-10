#!/usr/bin/env python3
"""Stock mlx-vlm generate() on an image: prompt/decode tok/s split (the old OCR path), for the round-driver A/B.
argv: model image out_json [reps]. Run through gpuq."""

import json
import sys
import time


def split(prompt_s: float, gen_tokens: int, total_s: float) -> dict:
    return {
        "prompt_s": prompt_s,
        "decode_ms_per_token": (total_s - prompt_s) * 1e3 / max(gen_tokens, 1),
    }


def main() -> int:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model_path, image, out = sys.argv[1:4]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    model, proc = load(model_path)
    fmt = apply_chat_template(
        proc,
        model.config,
        [{"role": "user", "content": "Text Recognition:"}],
        num_images=1,
    )
    rows = []
    for i in range(reps + 1):
        t = time.perf_counter()
        r = generate(model, proc, fmt, [image], max_tokens=4096, verbose=False)
        dt = time.perf_counter() - t
        if r.generation_tokens < 50:
            raise SystemExit(f"too few tokens: {r.generation_tokens}")
        if i:
            rows.append(
                {
                    "total_s": dt,
                    "gen_tokens": r.generation_tokens,
                    "prompt_tokens": r.prompt_tokens,
                    "prompt_tps": r.prompt_tps,
                    "generation_tps": r.generation_tps,
                    **split(r.prompt_tokens / r.prompt_tps, r.generation_tokens, dt),
                }
            )
    json.dump(rows, open(out, "w"), indent=1)
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
