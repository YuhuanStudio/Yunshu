"""GPU-only bit parity and timing pilot for RMS + BF16 scale fusion."""

from __future__ import annotations

import json

SOURCE = r"""
uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.x;
float vals[4];
float ss = 0.0f;
for (uint i=0; i<4; ++i) {
    vals[i] = float(x[row * 128 + lane * 4 + i]);
    ss += vals[i] * vals[i];
}
ss = simd_sum(ss);
float inv = metal::precise::rsqrt(ss / 128.0f + 1e-6f);
for (uint i=0; i<4; ++i) {
    bfloat16_t normed = bfloat16_t(vals[i] * inv);
    out[row * 128 + lane * 4 + i] = bfloat16_t(float(normed) * float(bfloat16_t(INVERSE ? 0.0078125f : 0.08838834764831845f)));
}
"""


def main():
    import mlx.core as mx

    mx.random.seed(423)
    kernel = mx.fast.metal_kernel(
        name="coldfuse_norm_scale",
        input_names=["x"],
        output_names=["out"],
        source=SOURCE,
    )
    rows = []
    for tokens in (1, 64, 128, 512, 1984, 2048):
        for heads in (16, 48):
            x = mx.random.normal((1, tokens, heads, 128)).astype(mx.bfloat16)
            mx.eval(x)
            for scale in (128**-1, 128**-0.5):
                ref = scale * mx.fast.rms_norm(x, None, 1e-6)

                def run():
                    return kernel(
                        inputs=[x],
                        template=[("INVERSE", scale == 128**-1)],
                        grid=(tokens * heads * 32, 1, 1),
                        threadgroup=(32, 1, 1),
                        output_shapes=[x.shape],
                        output_dtypes=[x.dtype],
                    )[0]

                got = run()
                mx.eval(ref, got)
                mismatches = int(
                    mx.sum(ref.view(mx.uint16) != got.view(mx.uint16)).item()
                )
                row = dict(
                    tokens=tokens, heads=heads, scale=scale, mismatches=mismatches
                )
                print(json.dumps(row), flush=True)
                rows.append(row)
                if mismatches:
                    raise RuntimeError("norm fusion bit mismatch")
    print(json.dumps(dict(complete=True, cells=len(rows))), flush=True)


if __name__ == "__main__":
    main()
