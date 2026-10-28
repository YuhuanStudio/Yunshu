# Third-party code

`vendor.json` lists every piece of code Yunshu copied (`vendored`), rewrote from specific upstream files
(`derived`), reimplemented from an upstream idea (`inspired`) or monkeypatches at runtime (`patches`), with the
upstream repo, paths, base commit and license; `just vendor-check` reports what changed upstream since
(see `docs/guides/UPSTREAM_TRACKING.md`). Each such source file starts with an `# Upstream` / `# Patches upstream`
comment. Summary of upstreams and licenses:

| Upstream | License | Used as |
|---|---|---|
| jundot/omlx | Apache-2.0 | vendored `kernels/omlx/`; derived `ragged_attention.py` tile kernel; inspired `spec_prefill.py`, `routers/decisions.py` (route design only), `lane_layers.py`, `model_settings.py`, `mrope.py`, `mlx_executor.py`, `prefill_progress.py`, `yunshu_kv/mlx_cache.py`; derived `process_memory_enforcer.py` |
| ashhart/TensorFold | MIT | vendored `kernels/tensorfold/`; derived `lane_linear.py`, `int_code_linear.py`, `copy_drafter.py`; inspired draft vocabulary, DFlash/tree drafting, tree verify, ordered chunk merge |
| Blaizzy/mlx-vlm | MIT | patched (speculative verifier, Qwen4 sampled-verify logit stash, DFlash/MTP rounds, batch generator, audio patches, APC checkpoint policy and lookup provenance); derived `dflash_context.py`, patched `penalty_context.py`, `dflash_apc_window.py`, `dflash_copy.py`, `mtp_lane.py`, `mtp_tree.py`; tool-format registry, DFlash chain depth, structured-output wiring |
| ml-explore/mlx-lm | MIT | tool parsers; speculative verification (`spec_draft_verifier.py`), sampler order (`batched_engine.py`) |
| ml-explore/mlx | MIT | patched `nn.QuantizedLinear.__call__` for batch-invariant decode; derived native NAX prefill loader (`nax_prefill.py`) |
| vllm-project/vllm, vllm-omni | Apache-2.0 | derived `gemma4_assistant.py`; inspired Qwen3 reasoning ends at a tool-call marker (`vlm_engine.py`), `structured_outputs` request field and entrypoint request-validation / route checks (`schemas/structured_outputs.py`, `scripts/research/route_checks_vllm.py`), n-gram / suffix proposers, block pool, mRoPE state |
| Cloudflare/clef (model + reference `joint_schema_model.py`) | Apache-2.0 | derived `decision_engine.py` (joint schema head and record encoding re-implemented in MLX) |
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
(row-invariant projections for the speculative lane).
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

## Trafilatura

Web research uses the unmodified Python dependency trafilatura 2.1.0, licensed under
Apache-2.0. Source and license: https://github.com/adbar/trafilatura/tree/v2.1.0 .
No source is vendored. Earlier versions before 1.8 are not used.

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

## vLLM / SGLang — idea only (`python/yunshu_engine/gc_policy.py`)

Collecting once and calling `gc.freeze()` on the startup heap after model load follows vLLM's
`freeze_gc_heap` (vllm/utils/gc_utils.py, Apache-2.0) and SGLang's `freeze_gc`
(python/sglang/srt/utils/common.py, Apache-2.0). No code is copied; the code is written independently.
## mlx2 telemetry

`python/yunshu_engine/telemetry/apple.py` is derived from pierre427/mlx2
`src/mlx2/apple_telemetry.py`, revision `92ada04cc7a59c3263e214b0a1127924fedd52db`,
under Apache-2.0. Upstream NOTICE: “mlx2 is licensed under the Apache License, Version 2.0 (see LICENSE).”
The unrelated third-party adaptations listed in upstream NOTICE are not used by this telemetry module.
Source: https://github.com/pierre427/mlx2. Yunshu changes ownership cleanup and reporting integration.

## macmon Apple telemetry compatibility

`telemetry/clpc.py` derives the macOS 27 driver/counter catalog and scalar IOReport
descriptor format from vladkens/macmon `src/clpc.rs` / `src/sources.rs`, revision
`7df49f55d9a1b9072e31fc8ba991abda84593563`. `telemetry/apple.py` uses channel-family
and MTR sensor naming knowledge from `src/metrics.rs`. Source: https://github.com/vladkens/macmon.
Native lifetime handling is rewritten in Python; unknown OS/driver catalogs are rejected.

MIT License

Copyright (c) 2024 vladkens

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

### Qwen4 native MTP integration

`python/yunshu_engine/qwen4_mtp.py` uses the MIT-licensed mlx-vlm
Qwen4Exp MTP head and family-specific verifier directly (Blaizzy/mlx-vlm).
Prior art and reviewed revisions are recorded in `vendor.json`; no TensorFold
or Strata implementation was copied into this integration.
`python/yunshu_engine/ple_parallel.py` replaces mlx-vlm's (MIT)
`QuantizedMMapNGramEmbedding._read_rows` with a parallel `pread` version that returns
the same bytes (monkeypatch recorded in `vendor.json`).
`python/yunshu_engine/ple_scale.py` applies the per-table `ngram_embedding.weight_scale` that mlx-vlm
drops (rows scaled in float32, rounded once to bf16; recorded in `vendor.json`).

## aiortc — optional WebRTC dependency

The `webrtc` extra uses unmodified aiortc (BSD-3-Clause) and PyAV (BSD-3-Clause).
Their source and binaries are not vendored in Yunshu. The reviewed macOS arm64
PyAV 17.1 wheel links shared FFmpeg libraries whose `avcodec_license()` reports
"LGPL version 3 or later". FFmpeg and codec licensing remains part of the separate
installed dependency distribution; Yunshu does not bundle or relink those binaries.
Sources and dependency licenses: https://github.com/aiortc/aiortc and
https://github.com/PyAV-Org/PyAV . The SDK protocol adapters are independent code
checked against openai-python (Apache-2.0) and anthropic-sdk-python (MIT).

## mflux — independent image reference probe

The priorfix_mflux_reference probe uses the unchanged loader cache, HF mapper
and weight applier APIs of mflux-community/mflux (MIT, reviewed commit
465df30a3965b38c9c64fa853e26df140ddd5c55). The probe decodes downloaded diffusers
affine4 tensors before mapping; its reference uses floating arithmetic. No mflux
source is copied and the serving engine does not import this bridge.
https://github.com/mflux-community/mflux

## TensorFold flash_next v1 kernels — `python/yunshu_engine/kernels/tf_flash_next/`

`base.py`, `hc.py`, `experts.py` (from `src/tensorfold/kernels/qwen/flash_next/v1/`), `threads.py` and `inputs.py`
(from `src/tensorfold/kernels/`) are copied from https://github.com/ashhart/TensorFold at commit `17c73e1`
(v0.6.1), which is licensed under the Apache License, Version 2.0 (releases before 0.6.0 were MIT). Only
intra-package imports were changed and a credit line was added at the top of each file; the full license
text is at https://www.apache.org/licenses/LICENSE-2.0 and in TensorFold's `LICENSE`. TensorFold's NOTICE:
"TensorFold. Copyright 2026 TensorFold contributors. https://github.com/ashhart/TensorFold".

## Splash native ANE runtime (1.4.0)

`python/yunshu_engine/ane/native/{Program.hpp,Program.mm,Handoff.hpp,Handoff.mm}`
are adapted from [incoai/splash](https://github.com/incoai/splash), commit
`1482f6bbc2abf12375332725302019594150b6cb`, under Apache License 2.0.
`Support.hpp` adapts its awake-time clock semantics and `mil.py` derives the
IOSurface MIL ABI from `runtime/ops/AneFfn.cpp`. Modified for Yunshu: standalone
clang/ctypes runtime, explicit cache, local Metal adapter, constant slice weights
and fp16 underflow scaling. The upstream files had no per-file copyright header.
The Apache-2.0 license is reproduced in this repository's LICENSE.
This research runtime is not enabled in serving.

The optional native ANE MLX bridge (`ane/native/Bridge.mm`) uses the installed
MLX 0.32.3 C++ Metal command encoder and Python DLPack contracts (Apple, MIT;
tag `v0.32.3`). Its C ABI, watchdog and ownership implementation is Yunshu's;
no MLX implementation source is copied. The coupling is recorded in vendor.json.

`ane/native/ProgramInstrumentation.hpp` also adapts Splash's Apache-2.0
test-only fault declarations. Production builds do not expose the arming API.

## Splash ANE FFN split (1.4.0)

`python/yunshu_engine/ane/{ffn_program.py,ffn_kernels.py,ffn_split.py,ffn_calibration.py,ffn_startup.py}`
are a Python/MLX port of the dense-FFN Neural Engine split of
[incoai/splash](https://github.com/incoai/splash), commit
`1482f6bbc2abf12375332725302019594150b6cb` (`runtime/ops/AneFfn.{hpp,cpp}`,
`AneFfnCalibration.*`, `AneFfnMeasurement.cpp`, `runtime/engine/AneFfnStartup.cpp`,
`runtime/metal/kernels/prefill/ane_ffn.metal`, `runtime/metal/abi/AneFfn.h`), under Apache
License 2.0: the W8A8 MIL program with 2560-channel segments and a block-Hadamard
intermediate rotation, the GPU rotate/pack/weight-staging/join kernels, the double-buffered
one-layer-ahead weight staging, the timing model, share choice, least-chunk rule and breaker.
Modified for Yunshu: MLX custom kernels over MLX affine and Yunshu tiled quantized weights, an
MLX-stream shared-event bridge, calibration kept in a file. Enabled only by
`YUNSHU_ANE_FFN_SPLIT=splash` (off by default; lossy). The Apache-2.0 license is reproduced in
this repository's LICENSE.
