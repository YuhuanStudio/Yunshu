"""Microbench: ways to write a K/V checkpoint prefix into a padded fresh buffer (APC restore).

Shapes follow the 27B full-attention layers (4 kv heads, head_dim 256, bf16) at 32832 rows
copied into capacity 33536. All variants must be bit-equal to the concatenate reference."""

import sys
import time

import mlx.core as mx


def variants():
    def concat(a, rows, pad):
        z = mx.zeros((*a.shape[:2], pad, a.shape[3]), dtype=a.dtype)
        return mx.concatenate([a[..., :rows, :], z], axis=2)

    def padop(a, rows, pad):
        return mx.pad(a[..., :rows, :], [(0, 0), (0, 0), (0, pad), (0, 0)])

    def slice_update(a, rows, pad):
        out = mx.zeros((*a.shape[:2], rows + pad, a.shape[3]), dtype=a.dtype)
        out[..., :rows, :] = a[..., :rows, :]
        return out

    def per_head(a, rows, pad):
        heads = []
        for h in range(a.shape[1]):
            z = mx.zeros((a.shape[0], 1, pad, a.shape[3]), dtype=a.dtype)
            heads.append(mx.concatenate([a[:, h : h + 1, :rows, :], z], axis=2))
        return mx.concatenate(heads, axis=1)

    def flat(a, rows, pad):
        b, h, _, d = a.shape
        src = a[..., :rows, :].reshape(b * h, rows * d)
        z = mx.zeros((b * h, pad * d), dtype=a.dtype)
        return mx.concatenate([src, z], axis=1).reshape(b, h, rows + pad, d)

    return dict(
        concat=concat,
        pad=padop,
        slice_update=slice_update,
        per_head=per_head,
        flat=flat,
    )


def main():
    rows, cap, n = 32832, 33536, 32
    pad = cap - rows
    srcs = [mx.random.normal((1, 4, cap, 256)).astype(mx.bfloat16) for _ in range(n)]
    mx.eval(srcs)
    for name, fn in variants().items():
        tiny = mx.zeros((1, 4, 600, 256), dtype=mx.bfloat16)
        mx.eval(fn(tiny, 512, 64))  # compile the copy kernels before timing
    for clear in (False, True):
        for name, fn in variants().items():
            ts = []
            for rep in range(4):
                if clear:
                    mx.clear_cache()  # next allocation must map fresh pages, like the engine after prompt_step
                mx.synchronize()
                t0 = time.perf_counter()
                outs = [fn(a, rows, pad) for a in srcs]
                mx.eval(outs)
                ts.append((time.perf_counter() - t0) * 1e3)
                if rep == 0:
                    ok = mx.array_equal(
                        outs[0], variants()["concat"](srcs[0], rows, pad)
                    ).item()
                    if not ok:
                        print("MISMATCH", name)
                        sys.exit(1)
                del outs
            mb = n * 2 * 1 * 4 * cap * 256 * 2 / 1e6
            print(
                f"clear={clear} {name:13s} min {min(ts):7.1f} ms  all {[round(t, 1) for t in ts]}  write {mb:.0f} MB",
                flush=True,
            )


main()
