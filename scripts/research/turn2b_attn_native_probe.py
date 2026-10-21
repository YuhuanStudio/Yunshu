"""Probe the non-D-split native attention body on short tails (gpuq only)."""

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
path = Path(
    str(
        importlib.metadata.distribution("mlx").locate_file(
            "mlx/include/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"
        )
    )
)
raw = path.read_text()
at = raw.index("void attention_nax(")
begin = raw.rfind("template <", 0, at)
end = raw.index("// clang-format off", at)
# The next declaration's template is preceded by the format marker.
body = raw[begin:end]
body = re.sub(r"\[\[.*?\]\]", "", body, flags=re.S)
body = body.replace(" void attention_nax(", " METAL_FUNC void turn2_native(")
body = body.replace("const constant Attn", "const thread Attn")
assert "METAL_FUNC void turn2_native(" in body
assert "[[" not in body
if a.dry_run:
    print(json.dumps(dict(body_bytes=len(body), variants=4, rows=532, keys=33300)))
    raise SystemExit(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import mlx.core as mx

from yunshu_engine.kernels import attn_prefill as ap

mx.random.seed(1022)
q = mx.random.normal((1, 532, 24, 256)).astype(mx.bfloat16).transpose(0, 2, 1, 3)
k = mx.random.normal((1, 4, 33300, 256)).astype(mx.bfloat16)
v = mx.random.normal(k.shape).astype(mx.bfloat16)
ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
mx.eval(ref)
for bq in (32, 64):
    for bk in (32, 64):
        h = ap.header(False, False)
        point = h.index("METAL_FUNC void yunshu_attn_dsplit")
        h = h[: h.rfind("template <", 0, point)] + body
        wm = bq // 16
        source = f"""
    AttnParams p;
    p.B=1;p.H=24;p.D=256;p.qL=532;p.kL=33300;p.gqa_factor=6;p.scale=0.0625f;
    p.NQ=(532+{bq}-1)/{bq};p.NK=(33300+{bk}-1)/{bk};
    p.NQ_aligned=532/{bq};p.NK_aligned=33300/{bk};
    p.qL_rem=532%{bq};p.kL_rem=33300%{bk};p.qL_off=33300-532;
    for(int i=0;i<3;i++){{p.Q_strides[i]=Q_strides[i];p.K_strides[i]=K_strides[i];p.V_strides[i]=V_strides[i];}}
    p.O_strides[0]=24*532*256;p.O_strides[1]=532*256;p.O_strides[2]=256;
    turn2_native<bfloat,{bq},{bk},256,256,{wm},1,float,float>(Q,K,V,O,&p,nullptr,nullptr,nullptr,
    thread_index_in_simdgroup,simdgroup_index_in_threadgroup,threadgroup_position_in_grid,thread_position_in_threadgroup);
    """
        fn = mx.fast.metal_kernel(
            name=f"turn2_native_{bq}_{bk}",
            input_names=["Q", "K", "V"],
            output_names=["O"],
            header=h,
            source=source,
            ensure_row_contiguous=False,
        )
        got = fn(
            inputs=[q, k, v],
            grid=((532 + bq - 1) // bq * 32, 24 * wm, 1),
            threadgroup=(32, wm, 1),
            output_shapes=[q.shape],
            output_dtypes=[mx.bfloat16],
        )[0]
        mx.eval(got)
        print(
            json.dumps(
                dict(
                    bq=bq,
                    bk=bk,
                    equal=bool(mx.array_equal(ref, got).item()),
                    mismatch=int(mx.sum(ref != got).item()),
                )
            ),
            flush=True,
        )
print(json.dumps(dict(complete=True)), flush=True)
