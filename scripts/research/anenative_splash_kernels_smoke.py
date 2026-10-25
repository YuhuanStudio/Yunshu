"""GPU kernels of the Splash-style FFN split against numpy references (no ANE involved).

Surfaces are ordinary MLX arrays here (the kernels write through raw pointers). Fails fast.
"""

from __future__ import annotations

import sys

import numpy as np

from yunshu_engine.ane import ffn_kernels as K
from yunshu_engine.ane import ffn_program as P


def main():
    import mlx.core as mx

    rng = np.random.default_rng(0)
    signs = P.rotation_signs()
    sg = mx.array(signs)
    r128 = P.rotation_matrix(128, signs)
    r512 = P.rotation_matrix(512, signs)

    # ---- rotate + pack -------------------------------------------------------------
    hidden, rows = 5120, 70
    x = rng.normal(0, 1.0, (rows, hidden)).astype(np.float32)
    x[3, 5] = 40.0
    xb = mx.array(x).astype(mx.bfloat16)
    xf = np.array(xb.astype(mx.float32))
    tx = mx.zeros((1, 2048), dtype=mx.float16)
    (rotated,) = K.kernel("rotate")(
        inputs=[xb, sg, tx],
        template=[("H", hidden)],
        grid=(256 * rows, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[mx.float16],
    )
    mx.eval(rotated, tx)
    ref = (xf.reshape(rows, hidden // 128, 128) @ r128.T).reshape(rows, hidden)
    peak = np.abs(ref).max(axis=1)
    scale = np.maximum(peak, P.PEAK_FLOOR) / 127.0
    ref_q = ref / scale[:, None]
    got = np.array(rotated.astype(mx.float32))
    err = np.abs(got - ref_q).max()
    print("rotate max abs err (int8 units)", err, flush=True)
    assert err < 0.05, "rotate mismatch"
    tsc = np.array(tx.astype(mx.float32))[0, :rows]
    assert np.allclose(tsc, scale * 128.0, rtol=2e-3), "token scale mismatch"

    stride = 2048  # int8 surface width 2048 -> row bytes 2048
    surf = mx.zeros((2560, 2048), dtype=mx.int8)
    tiles = (rows + 31) // 32
    (marker,) = K.kernel("pack")(
        inputs=[rotated, surf],
        template=[("H", hidden), ("R", rows), ("CHANNEL0", 0), ("STRIDE", stride)],
        grid=(tiles * 32, (2560 // 32) * 8, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.uint32],
    )
    mx.eval(marker)
    packed = np.array(surf)[:, :rows].T
    expect = np.clip(np.rint(got), -127, 127).astype(np.int8)[:, :2560]
    bad = int((packed != expect).sum())
    print("pack mismatches", bad, flush=True)
    assert bad == 0, "pack mismatch"

    # ---- weights + row scales (4- and 5-bit, group 64) -----------------------------
    for bits in (4, 5):
        out_rows, inp, gs = 256, 2560, 64
        wdense = rng.normal(0, 0.05, (out_rows, inp)).astype(np.float32)
        wq, sc, bi = mx.quantize(mx.array(wdense), group_size=gs, bits=bits)
        mx.eval(wq, sc, bi)
        deq = np.array(mx.dequantize(wq, sc.astype(mx.float32), bi.astype(mx.float32), group_size=gs, bits=bits))
        for kk, block in ((1, 128), (4, 512)):
            rot = P.rotation_matrix(block, signs)
            wrot = (deq.reshape(out_rows, inp // block, block) @ rot.T).reshape(out_rows, inp)
            (rs,) = K.kernel("row_scale")(
                inputs=[wq, sc, bi, sg],
                template=[("K", kk), ("KIN", inp), ("BITS", bits), ("GS", gs), ("ROW0", 0), ("IN0", 0), ("WIDTH", inp - inp % block)],
                grid=(out_rows // 8 * 256, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(out_rows,)],
                output_dtypes=[mx.float16],
            )
            mx.eval(rs)
            width = inp - inp % block
            peak = np.abs(wrot[:, :width]).max(axis=1)
            rs_ref = np.maximum(peak, P.PEAK_FLOOR) / 127.0 * 128.0
            assert np.allclose(np.array(rs.astype(mx.float32)), rs_ref, rtol=3e-3), f"row scale bits={bits} k={kk}"
            surf = mx.zeros((out_rows, width), dtype=mx.int8)
            ssurf = mx.zeros((out_rows, 32), dtype=mx.float16)
            (m,) = K.kernel("weights")(
                inputs=[wq, sc, bi, rs, sg, surf, ssurf],
                template=[("K", kk), ("KIN", inp), ("BITS", bits), ("GS", gs), ("ROW0", 0), ("IN0", 0), ("STRIDE", width), ("SSTRIDE", 32)],
                grid=(out_rows // 8 * 256, width // (kk * 128), 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(1,)],
                output_dtypes=[mx.uint32],
            )
            mx.eval(m)
            rs_half = np.array(rs.astype(mx.float32))
            q_ref = np.clip(np.rint(wrot[:, :width] * (128.0 / rs_half[:, None])), -127, 127)
            diff = np.abs(np.array(surf).astype(np.int32) - q_ref.astype(np.int32))
            print(f"weights bits={bits} K={kk}: max diff {diff.max()} ties {(diff > 0).sum()}", flush=True)
            assert diff.max() <= 1 and (diff > 0).mean() < 1e-3, "weights mismatch"
            assert np.allclose(np.array(ssurf.astype(mx.float32))[:, 0], np.array(rs.astype(mx.float32))), "scale copy"

    # ---- join -----------------------------------------------------------------------
    hid, rws = 5120, 70
    stride_p = 2048
    part = rng.normal(0, 2.0, (hid + 1, stride_p)).astype(np.float16)
    part[hid, :] = rng.uniform(0.5, 2.0, stride_p).astype(np.float16)
    gpu = mx.array(rng.normal(0, 1, (rws, hid)).astype(np.float32)).astype(mx.bfloat16)
    tscale = mx.array(rng.uniform(0.5, 2.0, (1, 2048)).astype(np.float16))
    out, status = K.kernel("join")(
        inputs=[gpu, mx.array(part), tscale],
        template=[("H", hid), ("R", rws), ("PSTRIDE", stride_p)],
        grid=(((rws + 31) // 32) * 32, (hid // 32) * 8, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(rws, hid), (((rws + 31) // 32) * (hid // 32),)],
        output_dtypes=[mx.bfloat16, mx.uint32],
    )
    mx.eval(out, status)
    sc_tok = part[hid, :rws].astype(np.float32) * np.array(tscale)[0, :rws].astype(np.float32)
    ref = (np.array(gpu.astype(mx.float32)) + part[:hid, :rws].astype(np.float32).T * sc_tok[:, None])
    ref_bf = np.array(mx.array(ref).astype(mx.bfloat16).astype(mx.float32))
    mism = float((np.array(out.astype(mx.float32)) != ref_bf).mean())
    print("join mismatch fraction", mism, flush=True)
    assert mism < 2e-3, "join mismatch"
    assert int(np.array(status).max()) == 0, "status flagged finite data"
    part2 = part.copy()
    part2[7, 3] = np.inf
    _, status2 = K.kernel("join")(
        inputs=[gpu, mx.array(part2), tscale],
        template=[("H", hid), ("R", rws), ("PSTRIDE", stride_p)],
        grid=(((rws + 31) // 32) * 32, (hid // 32) * 8, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(rws, hid), (((rws + 31) // 32) * (hid // 32),)],
        output_dtypes=[mx.bfloat16, mx.uint32],
    )
    mx.eval(status2)
    assert int(np.array(status2).max()) == 1, "non-finite partial not flagged"
    print("kernels smoke OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
