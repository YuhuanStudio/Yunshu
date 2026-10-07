# Third-party code

`vendor.json` lists every piece of code Yunshu copied (`vendored`), rewrote from specific upstream files
(`derived`), reimplemented from an upstream idea (`inspired`) or monkeypatches at runtime (`patches`), with the
upstream repo, paths, base commit and license; `just vendor-check` reports what changed upstream since
(see `docs/guides/UPSTREAM_TRACKING.md`). Each such source file starts with an `# Upstream` / `# Patches upstream`
comment. Summary of upstreams and licenses:

| Upstream | License | Used as |
|---|---|---|
| jundot/omlx | Apache-2.0 | vendored `kernels/omlx/`; derived `ragged_attention.py` tile kernel, `mtp_patch.py`; inspired `spec_prefill.py`, `lane_layers.py`, `model_settings.py`, `mrope.py`, `mlx_executor.py`, `prefill_progress.py`, `yunshu_kv/mlx_cache.py`; derived `process_memory_enforcer.py` |
| ashhart/TensorFold | MIT | vendored `kernels/tensorfold/`; derived `lane_linear.py`, `int_code_linear.py`, `round_driver/allocate.py`, `copy_drafter.py`; inspired draft vocabulary, DFlash/tree drafting, tree verify, round driver, ordered chunk merge |
| Blaizzy/mlx-vlm | MIT | patched (speculative verifier, DFlash/MTP rounds, batch generator, audio patches, APC checkpoint policy and lookup provenance); derived `dflash_context.py`, `dflash_copy.py`, `mtp_lane.py`, `mtp_tree.py`, `round_driver/mtp.py`; tool-format registry, DFlash chain depth, structured-output wiring |
| ml-explore/mlx-lm | MIT | patched Qwen3.5 GatedDeltaNet (`n_confirmed_patch.py`); tool parsers; speculative verification (`spec_draft_verifier.py`), sampler order (`batched_engine.py`) |
| ml-explore/mlx | MIT | patched `nn.QuantizedLinear.__call__` for batch-invariant decode; derived native NAX prefill loader (`nax_prefill.py`) |
| vllm-project/vllm, vllm-omni | Apache-2.0 | derived `gemma4_assistant.py`; inspired Qwen3 reasoning ends at a tool-call marker (`vlm_engine.py`), `structured_outputs` request field and entrypoint request-validation / route checks (`schemas/structured_outputs.py`, `scripts/research/route_checks_vllm.py`), n-gram / suffix proposers, block pool, mRoPE state, TeaCache |
| huggingface/transformers | Apache-2.0 | inspired EmbeddingGemma 2 text tower (`embedding_gemma2.py`; the vision / audio towers are imported from mlx-vlm gemma4) |
| ggml-org/llama.cpp | MIT | inspired n-gram hash pool |
| sgl-project/sglang | Apache-2.0 | inspired radix-tree prefix matching (`kv_optimizations.py`) |
| waybarrios/vllm-mlx | Apache-2.0 | inspired warm-prompt preloading (`model_optimizations.py`) |
| incoai/splash | Apache-2.0 | inspired lossless (batch-invariant) speculative decoding; credited inside the vendored oMLX kernels |
| bstnxbt/dflash-mlx, youssofal/MTPLX | Apache-2.0 | credited inside the vendored oMLX kernels |

`nax_prefill.py` reuses installed MLX 0.32.3 Metal headers (Apple copyright, MIT) at tag `v0.32.3`, including `quantized_nax.h` and `steel/gemm/nax.h`. It preserves their dequantizer, NAX MAC and K accumulation, replacing only the global input loader for the lane layout and choosing measured M tiles. Runtime expansion preserves the original header copyright notices; source provenance is recorded in `vendor.json`.

## oMLX — `python/yunshu_engine/kernels/omlx/`

Qwen3.5-family MTP verify kernels (`qwen35_verify_qmm.py`, `qwen35_gdn_prework.py`,
`qwen35_gdn_verify_fused.py`, `qwen35_verify_sdpa_split.py`, `qwen35_packed_linear.py`,
`qwen35_verify_linear.py`, `row_exact_qmv.py`, `module_cache.py`, `moe_verify_gather.py`) are
copied from https://github.com/jundot/omlx (`omlx/patches/`, per-file commits in `vendor.json`),
licensed under the Apache License 2.0. Only intra-package imports were changed, except
`qwen35_packed_linear.py`, which Yunshu modified (2026-09-28) to route 5/6/8-bit
projections to the TensorFold integer-code matmul (the default when projections are packed); its 4-bit
kernels are unchanged. The files
carry their own upstream credits (MTPLX, dflash-mlx, Splash — Apache-2.0).

The token-tile partial kernel in `python/yunshu_engine/kernels/ragged_attention.py` (`_TILE`) is
adapted from oMLX's tensor-op verify attention (`_GQA_PARTIAL` in `qwen35_verify_sdpa_split.py`,
Apache-2.0): same MetalPerformancePrimitives matmul/online-softmax structure, with Yunshu's fixed
512-key chunks, per-row lengths, cache slots and a fixed 8-token row tile.

The exact source commit and our intended local changes for every vendored file are listed in
`vendor.json`; `just vendor-check` reports what changed upstream since.

## TensorFold — `python/yunshu_engine/kernels/tensorfold/`

`lane_qmm.py`, `lane_widen.py` (from `src/tensorfold/kernels/qwen/dense/v1/`) and `inputs.py`
(from `src/tensorfold/kernels/`) are copied from https://github.com/ashhart/TensorFold at commit
`34bae79ac97da6c3ab3fe10159cf49633ce8112a`. Only intra-package imports were changed. Yunshu calls
their integer-code tensor-unit matmul from `python/yunshu_engine/kernels/int_code_linear.py`
(the default for 5/6/8-bit projections when packed) and from `python/yunshu_engine/kernels/lane_linear.py`
(every projection under the round driver). `python/yunshu_engine/round_driver/allocate.py` adapts the
draft-allocation rule of TensorFold's `src/tensorfold/engine/allocate.py` (same commit).
TensorFold's own notices credit MLX (MIT) for code it derives.

```
MIT License

Copyright (c) 2026 TensorFold contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## llguidance — runtime dependency (`python/yunshu_engine/tool_call_grammar.py`)

Tool-call constrained decoding builds a Lark grammar (structural tag: free text, then
the call body) and runs it with llguidance's token-mask matcher and MLX bitmask apply.
Used as a library, not copied. Copyright (c) Microsoft Corporation, MIT license.
https://github.com/guidance-ai/llguidance

## Yunshu Console

The console imports the pinned YunUI package in
`frontend/vendor/yuhuanowo-yunui-0.2.18-analytics.tgz` (Apache-2.0); its source commit and SHA-256
are recorded in `frontend/vendor/README.md`, and other frontend dependencies are
locked by `frontend/pnpm-lock.yaml`. No oMLX dashboard source was copied into this
frontend. YunUI includes MIT-licensed Beautiful UI adaptations.

The build ships YunUI's license, upstream notices and retained Beautiful UI MIT
text under `console_static/licenses/yunui/`; Vite generates the bundled dependency
license catalogue at `console_static/licenses/bundled-dependencies.md`. The local
Geist and JetBrains Mono font assets carry their OFL texts under `console_static/fonts/`.
