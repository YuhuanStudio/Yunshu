"""Raw peak of the M5 GPU tensor units (Metal Performance Primitives matmul2d) per input type.

One threadgroup loops a 64x64x64 matmul2d on cache-resident tiles (no DRAM traffic) many
times; reports TFLOPS (TOPS for int8) for half / bfloat / int8 inputs, and how it scales with
resident threadgroups. Shows whether an int8 tensor path (W8A8 / W4A8 prefill) would have
more headroom than the bf16 path MLX uses (~62 TFLOPS measured by gpu_compute.py).

    PYTHONPATH=scripts/research/hw python scripts/research/hw/tensor_unit_peak.py
"""

import mlx.core as mx
from _common import Out, timeit

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

SRC = """
  constexpr int M = 64, N = 64, K = 64;
  const uint tg = threadgroup_position_in_grid.x;
  auto at = tensor((device IN_T*)a, dextents<int, 2>{K, M}, array<int, 2>{1, K});
  auto bt = tensor((device IN_T*)b, dextents<int, 2>{K, N}, array<int, 2>{1, K});
  constexpr auto desc = matmul2d_descriptor(M, N, K, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto a0 = at.template slice<K, M>(0, 0);
  auto b0 = bt.template slice<K, N>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), ACC_T>();
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = ACC_T(0);
  for (int it = 0; it < ITERS; ++it) {
    op.run(a0, b0, acc);
  }
  ACC_T s = ACC_T(0);
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) s += acc[i];
  out[tg * 128 + thread_position_in_threadgroup.x] = float(s);
"""


def run(name, dt, in_t, acc_t, ntg, iters=2000):
    a = mx.ones((64, 64), dtype=dt)
    b = mx.ones((64, 64), dtype=dt)
    mx.eval(a, b)
    k = mx.fast.metal_kernel(
        name=f"tu_{name}",
        input_names=["a", "b"],
        output_names=["out"],
        source=SRC,
        header=HEADER,
    )

    def call():
        return k(
            inputs=[a, b],
            template=[("IN_T", dt), ("ACC_T", acc_t), ("ITERS", iters)],
            grid=(ntg * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(ntg * 128,)],
            output_dtypes=[mx.float32],
        )[0]

    t, _ = timeit(lambda: mx.eval(call()), 5, 2)
    flops = 2 * 64 * 64 * 64 * iters * ntg
    return t, flops / t / 1e12


def main():
    out = Out("tensor_unit_peak")
    for name, dt, in_t, acc_t in (
        ("half", mx.float16, "half", mx.float32),
        ("bf16", mx.bfloat16, "bfloat", mx.float32),
        ("int8", mx.int8, "int8_t", mx.int32),
    ):
        for ntg in (40, 160, 640, 2560):
            try:
                t, tf = run(name, dt, in_t, acc_t, ntg)
                out(
                    kind="tensor_unit",
                    dtype=name,
                    threadgroups=ntg,
                    ms=round(t * 1e3, 2),
                    TFLOPS_or_TOPS=round(tf, 1),
                )
            except Exception as e:  # noqa: BLE001
                out(kind="tensor_unit", dtype=name, threadgroups=ntg, err=str(e)[:300])
                break


if __name__ == "__main__":
    main()
