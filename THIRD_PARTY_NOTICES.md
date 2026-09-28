# Third-party code

## oMLX — `python/yunshu_engine/kernels/omlx/`

Qwen3.5-family MTP verify kernels (`qwen35_verify_qmm.py`, `qwen35_gdn_prework.py`,
`qwen35_gdn_verify_fused.py`, `qwen35_verify_sdpa_split.py`, `qwen35_packed_linear.py`,
`qwen35_verify_linear.py`) are copied from https://github.com/jundot/omlx at commit
`f0d8428acd3220c364177d1ea9593e4e15f94107` (`omlx/patches/`), licensed under the
Apache License 2.0. Only intra-package imports were changed, except
`qwen35_packed_linear.py`, which Yunshu modified (2026-09-28) to add opt-in 5-bit
packing and kernels (`YUNSHU_PACKED_5BIT=1`); its 4-bit kernels are unchanged. The
files carry their own upstream credits (MTPLX, dflash-mlx, Splash — Apache-2.0).
