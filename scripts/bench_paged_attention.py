"""Bench: Metal PagedAttention decode vs MLX fallback.

Measures throughput for different (num_heads, head_dim, seq_len) configs.
"""
import time
import mlx.core as mx
from yunshu_engine.metal_kernels import MetalKernelManager


def bench(mgr, label, num_heads, head_dim, seq_len, kv_block_size, num_queries, iters=20):
    scale = 1.0 / (head_dim ** 0.5)
    num_blocks_needed = (seq_len + kv_block_size - 1) // kv_block_size
    num_blocks = num_blocks_needed + 2

    key_cache = mx.random.normal((num_blocks * kv_block_size, num_heads, head_dim)).astype(mx.float16)
    val_cache = mx.random.normal((num_blocks * kv_block_size, num_heads, head_dim)).astype(mx.float16)
    queries = mx.random.normal((num_queries, num_heads, head_dim)).astype(mx.float16)

    bt_rows = []
    sl_list = []
    for i in range(num_queries):
        nb = num_blocks_needed
        row = list(range(nb)) + [-1] * (num_blocks - nb)
        bt_rows.append(row)
        sl_list.append(seq_len)
    block_tables = mx.array(bt_rows, dtype=mx.int32)
    seq_lens = mx.array(sl_list, dtype=mx.int32)

    # Warmup
    for _ in range(3):
        if label == "Metal":
            out = mgr.paged_attention_decode(
                queries, key_cache, val_cache, block_tables, seq_lens,
                num_heads, head_dim, kv_block_size, scale,
            )
        else:
            out = mgr._fallback_paged_attention(
                queries, key_cache, val_cache, block_tables, seq_lens,
                num_heads, head_dim, kv_block_size, scale,
            )
        mx.eval(out)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(iters):
        if label == "Metal":
            out = mgr.paged_attention_decode(
                queries, key_cache, val_cache, block_tables, seq_lens,
                num_heads, head_dim, kv_block_size, scale,
            )
        else:
            out = mgr._fallback_paged_attention(
                queries, key_cache, val_cache, block_tables, seq_lens,
                num_heads, head_dim, kv_block_size, scale,
            )
        mx.eval(out)
    dt = time.perf_counter() - t0

    total_kv = num_queries * seq_len * num_heads * head_dim * 2  # K + V elements read
    qps = num_queries * iters / dt
    return dt / iters * 1000, qps


def main():
    mgr = MetalKernelManager()

    configs = [
        # (num_heads, head_dim, seq_len, kv_block_size, num_queries)
        (8,  64,  64,  16, 1),
        (8,  64,  256, 16, 1),
        (8,  64,  1024, 16, 1),
        (8,  128, 64,  16, 1),
        (8,  128, 256, 16, 1),
        (8,  128, 1024, 16, 1),
        (32, 128, 256, 16, 1),  # LLM-scale
        (32, 128, 1024, 16, 1),
        (8,  64,  256, 16, 8),  # batched
        (32, 128, 256, 16, 4),
    ]

    print(f"{'='*90}")
    print(f"  PagedAttention Decode: Metal Kernel vs MLX Fallback")
    print(f"{'='*90}")
    print(f"  {'Config':<35} {'Metal':>12} {'Fallback':>12} {'Speedup':>8}")
    print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*8}")

    for nh, hd, sl, kbs, nq in configs:
        label = f"h{nh}_d{hd}_seq{sl}_k{kbs}_q{nq}"
        try:
            metal_ms, metal_qps = bench(mgr, "Metal", nh, hd, sl, kbs, nq)
        except Exception as e:
            print(f"  {label:<35} FAILED: {e}")
            continue
        fallback_ms, fallback_qps = bench(mgr, "Fallback", nh, hd, sl, kbs, nq)

        speedup = fallback_ms / metal_ms if metal_ms > 0 else 0
        mark = " ★" if speedup > 1.05 else ""
        print(f"  {label:<35} {metal_ms:>10.2f}ms {fallback_ms:>10.2f}ms {speedup:>6.2f}x{mark}")

    print(f"{'='*90}")


if __name__ == "__main__":
    main()
