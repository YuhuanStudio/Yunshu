"""Bounded same-device Z-Image timing; submit only after the parity pilot succeeds."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from statistics import median


def plan():
    return [
        (rep, arm)
        for rep in range(3)
        for arm in (("ours", "mflux") if rep % 2 == 0 else ("mflux", "ours"))
    ]


def summarize(rows):
    import math

    if len(rows) != 6 or {(r["rep"], r["arm"]) for r in rows} != set(plan()):
        raise ValueError("three complete interleaved pairs required")
    if {r["device"] for r in rows} != {"M5"}:
        raise ValueError("mixed or incorrect device")
    if any(not math.isfinite(r["seconds"]) or r["seconds"] <= 0 for r in rows):
        raise ValueError("invalid timing")
    medians = {
        arm: median(r["seconds"] for r in rows if r["arm"] == arm)
        for arm in ("ours", "mflux")
    }
    return {
        "medians_seconds": medians,
        "mflux_over_ours": medians["mflux"] / medians["ours"],
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/Volumes/P5Plus/models/Z-Image-Turbo-MLX-4bit")
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def run(args):
    import math

    import numpy as np
    from PIL import Image
    from priorfix_mflux_reference import checkpoint_contract, load_reference

    from yunshu_engine.image_engine import ImageGenEngine

    checkpoint_contract(args.model)
    ours = ImageGenEngine(args.model)
    ours.load()
    mflux, reference_format = load_reference(args.model)
    prompt = "A red ball on a white table."
    rows, images = [], {}
    for rep, arm in plan():
        start = time.perf_counter()
        if arm == "ours":
            png = ours._run_pipeline(prompt, 256, 256, 2, 7)
            image = Image.open(io.BytesIO(png)).convert("RGB")
        else:
            generated = mflux.generate_image(
                seed=7, prompt=prompt, width=256, height=256, num_inference_steps=2
            )
            encoded = io.BytesIO()
            generated.image.save(encoded, format="PNG")
            image = Image.open(io.BytesIO(encoded.getvalue())).convert("RGB")
        elapsed = time.perf_counter() - start
        if not math.isfinite(elapsed) or elapsed <= 0 or image.size != (256, 256):
            raise ValueError("invalid first arm timing/output")
        # PIL image materialization completes GPU work before the timer ends.
        rows.append({"rep": rep, "arm": arm, "seconds": elapsed, "device": "M5"})
        print(json.dumps(rows[-1]), flush=True)
        images[arm] = np.asarray(image).astype(float)
    if images["ours"].shape != images["mflux"].shape:
        raise ValueError("output shape mismatch")
    return {
        "complete": True,
        "passed": True,
        "device": "M5",
        "rows": rows,
        "reference_format": reference_format,
        "output_pixel_rmse": float(
            np.mean((images["ours"] - images["mflux"]) ** 2) ** 0.5
        ),
        "output_equal": bool(np.array_equal(images["ours"], images["mflux"])),
        "workload": "256px, 2 steps, seed7, three interleaved warm-process pairs; excludes loading",
        **summarize(rows),
    }


def main():
    args = parser().parse_args()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan()}))
        return 0
    result = run(args)
    Path(args.out).write_text(json.dumps(result) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
