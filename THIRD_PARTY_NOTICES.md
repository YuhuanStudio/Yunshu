# Third-party code

## oMLX — `python/yunshu_engine/kernels/omlx/`

Qwen3.5-family MTP verify kernels (`qwen35_verify_qmm.py`, `qwen35_gdn_prework.py`,
`qwen35_gdn_verify_fused.py`, `qwen35_verify_sdpa_split.py`, `qwen35_packed_linear.py`,
`qwen35_verify_linear.py`, `row_exact_qmv.py`, `module_cache.py`, `moe_verify_gather.py`) are
copied from https://github.com/jundot/omlx (`omlx/patches/`, per-file commits in `vendor.json`),
licensed under the Apache License 2.0. Only intra-package imports were changed. The files carry their own
upstream credits (MTPLX, dflash-mlx, Splash — Apache-2.0).

The exact source commit and our intended local changes for every vendored file are listed in
`vendor.json`; `just vendor-check` reports what changed upstream since.
