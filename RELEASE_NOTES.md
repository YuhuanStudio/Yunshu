# Yunshu Release Notes v0.0.1

## Release Checklist

- Tests: PASS (2,162 tests)
- Imports: PASS
- Routers: PASS (14 routers)
- Circular imports: NONE
- License: Apache 2.0
- README: PASS
- Packaging: PASS

## Benchmark Results (M3 Max, Qwen3.5-9B-4bit)

| Framework | MMLU Accuracy | Throughput | vs Baseline |
|-----------|:------------:|:----------:|:-----------:|
| **Yunshu** | **96%** | **44.4 tok/s** | **1.13x** |
| mlx-lm | 96% | 39.3 tok/s | 1.00x |
| oMLX | 96% | 41.8 tok/s | 1.06x |

## Code Statistics

- Source files: 160+
- Source lines: 38,000+
- Metal kernels: 6 (874 lines, Metal 3.1)
- TypeScript files: 12 (3,823 lines, Next.js 16)
- Tests: 2,162 passing (95+ test files)
- Packages: yunshu_api, yunshu_cli, yunshu_control, yunshu_engine, yunshu_gateway, yunshu_kv, yunshu_mesh, yunshu_sdk

## Per-Package Breakdown

- **yunshu_api**: Admin management API
- **yunshu_cli**: Command-line interface (Typer + Rich)
- **yunshu_control**: RBAC, scheduling, request queue
- **yunshu_engine**: 5-modality inference engine (40+ modules)
- **yunshu_gateway**: FastAPI HTTP server (14 routers, 6 middleware)
- **yunshu_kv**: 4-tier KV cache hierarchy (12 modules)
- **yunshu_mesh**: Compute mesh (mx.distributed, discovery, heartbeat)
- **yunshu_sdk**: Python SDK (OpenAI drop-in)

## Status

READY

---
*Generated for v0.0.1 release*
