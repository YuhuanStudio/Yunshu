# Model support and validation


> Console docs: the maintained, translated version of this page is `/docs/guides/model-support` (source `frontend/docs/guides/model-support.mdx`). This file stays as the repository-side reference read by tooling.
> The console's Docs section carries this guide in English, 繁體中文 and 简体中文 ([`frontend/docs/guides/model-support.mdx`](../../frontend/docs/guides/model-support.mdx)); this file is its GitHub-facing counterpart.

Yunshu loads text models through mlx-lm and vision-language models through
mlx-vlm. Upstream loadability does not certify every Yunshu endpoint or feature.
The server's `/v1/models` capability contract describes the loaded model; check it
before using tools, structured output, logprobs or media. See
[API surface](API_SURFACE.md) and [agent compatibility](AGENT_COMPAT.md).

| Support scope | What it means | Evidence / limits |
|---|---|---|
| First tuned checkpoint | Qwen3.8-27B oQ4e-mtp | Dated decode, TTFT and correctness evidence in BENCHMARKS; do not transfer numbers to other quantizations |
| Tuned family | Qwen3.5-family models | Family-specific speculation and prompt reuse; exact feature support depends on checkpoint and backend |
| Generic text | Other mlx-lm models | Text serving path; upstream support alone is not an endpoint certification |
| Generic vision-language | Other mlx-vlm models | Shared vision-language serving path; model-specific media/template/capability restrictions apply |
| Decisions | Clef / Clef-flash MLX joint schema head | Separate forward-only engine; [guide](DECISIONS.md); other decision heads are rejected |
| Other modalities | Audio, embeddings, image/video generation | Optional extras and model-specific routes; see API and configuration docs |

For a new validation result, record exact model repository/revision, quantization,
upstream versions, Yunshu SHA, chip/macOS, tested routes, tools/schema/streaming/
stop/logprobs/cancel, and cold/partial/full prompt reuse. Mark each case passed,
failed or untested and link its receipt. A capability advertised by code and a
feature proven on hardware are distinct. The release gate supplies acceptance
checks; this page does not invent results for untested checkpoints.

## Per-checkpoint receipts (real server, 2026-10-24)

One `yunshu serve` per checkpoint on the M5 (default settings), then through the `openai` SDK: load, a chat completion, a stream with usage, a forced function call, and (when the model card advertises image input) a solid-red image the model must name. `untested` means the card does not advertise the capability; `not run` means no receipt exists. Script: `scripts/research/model_family_check.py`, gpuq jobs labelled `realcheck-fam-<model>`.

| Checkpoint | Family and path | Load | Chat | Stream | Tools | Vision | Notes |
|---|---|---|---|---|---|---|---|
| `Qwen3.8-27B oQ4e-mtp` | qwen3_5 (VLM runner, tuned) | not run | not run | not run | not run | not run | not run: the gpuq job was cancelled before it started (27B jobs waited behind higher-priority work) |
| `Qwen3.8-27B MLX-4bit (TensorFold)` | qwen3_5 (VLM runner) | not run | not run | not run | not run | not run | not run: the gpuq job was cancelled before it started (27B jobs waited behind higher-priority work) |
| `qwen3.8-27b-4bit (kapitsa)` | qwen3_5 (VLM runner) | not run | not run | not run | not run | not run | not run: the gpuq job was cancelled before it started (27B jobs waited behind higher-priority work) |
| `openjev-MLX-4bit` | qwen3_5 (VLM runner) | pass | pass | pass | pass | untested | vision: the model card does not advertise image input |
| `Qwen3.5-9B-MLX-4bit` | qwen3_5 (VLM runner, tuned family) | pass | pass | pass | pass | pass |  |
| `Qwen3.5-2B-MLX-bf16` | qwen3_5 (VLM runner, tuned family) | pass | pass | pass | pass | pass |  |
| `Qwen3.5-0.8B-MLX-bf16` | qwen3_5 (VLM runner, tuned family) | pass | pass | pass | pass | pass |  |
| `Qwen2.5-3B-Instruct-4bit` | qwen2 (text fast path) | pass | pass | pass | pass | untested | vision: the model card does not advertise image input |
| `gemma-4-e4b-it-bf16` | gemma4 (VLM runner) | pass | pass | pass | pass | pass | forced `tool_choice` refused with 400; `auto` call parsed |
| `gemma-4-e2b-it-4bit` | gemma4 (VLM runner) | pass | pass | pass | FAIL | pass | tools: BadRequestError: Error code: 400 - {'error': {'message': "Cannot guarantee forced tool_choice for this model a |
| `Qwen3-Omni-30B-A3B-Instruct-4bit` | qwen3_omni_moe (VLM runner) | pass | pass | pass | pass | pass |  |
| `Nemotron-3-Nano-Omni-30B-A3B-4bit` | nemotron_h_nano_omni (VLM runner) | pass | pass | pass | pass | pass |  |
| `d1-3B` | lfm2_vl (VLM runner) | pass | pass | pass | pass | pass |  |
| `d1-3B-MLX-8bit` | lfm2_vl (VLM runner) | pass | pass | pass | pass | pass |  |
| `Lance-3B-bf16` | qwen2_5_vl image specialist | FAIL | untested | untested | untested | untested | load: RuntimeError: server not ready after 174s; :     127.0.0.1:62413 - "GET /v1/yunshu/status HTTP/1; chat: load failed; stream: load failed; tools: load failed; vision: load failed |

`Lance-3B-bf16` is an image-to-image specialist, not a chat checkpoint: the server never became ready within 174 s (the reason was not captured; not re-run). Qwen3.5-4B (broken Hugging Face cache link), DeepSeek-V4-Flash and Qwen3.8-Flash-Next were not run.

## Qwen3.8-Flash-Next under 80 GB (qwen4_exp)

`Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` is 106 GB on disk, of which 32 GB is the per-layer n-gram embedding (PLE). Yunshu reads PLE rows from the SSD instead of loading the table, which leaves 74.3 GB of resident weights (the 1.5 GB MTP head included).

1. Serve the pack as it is: `yunshu serve -m <pack>`. Since 2026-10-28 the PLE table is read from the pack's own files by default (`YUNSHU_PLE_STORAGE=ssd`): the first load builds a small row manifest under `~/.yunshu/cache/ple/`, nothing is copied and the pack is untouched. Measured on Jundot oQ2.5e-mtp: peak memory 82.9 → 54.5 GB, identical text, speed unchanged (DFlash2 81.6, MTP 84.3 tok/s against 82.4 / 85.4 with a hand-built view, 3 interleaved reps). `YUNSHU_PLE_STORAGE=memory` loads the table like any weight.
2. A hand-built external-PLE view (`python scripts/research/flashnext80_prepare_view.py <pack> <pack>-ple-ssd`, hard links on the same volume) still works and is used as it is. The MTP head is used automatically (one draft per round); the APC SSD tier is on by default.

With a memory ceiling set, Yunshu keeps the freed-buffer pool at 0.5 GiB and leaves the vision tower unloaded until the first image request. Both are lossless; set `YUNSHU_PREFILL_BUFFER_CACHE_GB` to override the pool.

Historical external-PLE prototype measurements (including pre-fused-kernel runs), not current release speed or an 80 GB served-peak guarantee. The 0.1.5 snapshot measures Flash-Next at 32K / 64K: 67.050 / 56.750 tok/s; cold TTFT 33.243 / 68.247 s, warm 0.132 / 0.229 s, immediate follow-up 2.474 / 2.918 s. [Current evidence](../BENCHMARKS.md#where-yunshu-stands).

| Context (M5 Max 128 GB, MTP on, whole-system memory delta) | Steady | Cold-prefill peak | Decode |
|---|---:|---:|---:|
| 1K | 75-76 GB | 77 GB | 70 tok/s |
| 8K | 77 GB | 79 GB | 66 tok/s |
| 32K | 75 GB | 80 GB | 37-38 tok/s (before the fused kernels) |
| about 169K, SSD APC hit (second request) | 75 GB | 81 GB | 22 tok/s, first token in 1.2 s |
| about 169K, cold | 75 GB | 84-85 GB | 21 tok/s |

A cold prefill beyond roughly 32K needs about 83-85 GB at its peak (the extra is prefill working memory); the same prompt served from the SSD prompt cache stays near 81 GB. Output is identical with speculation on and off and with a prompt-cache hit or miss. Decode uses fused hyper-connection and router/expert Metal kernels (derived from TensorFold, Apache-2.0), 65 tok/s without speculation at 1K and 55 at 8K; MTP adds 8% at 1K and 20% at 8K because every extra verify row reads more experts. Thinking-mode MMLU-Pro (300 questions, greedy): 261/300 = 0.870 (0 parse failures; the 27B oQ4e gate protocol gives 0.833).

## Omni (audio, image, video in; speech out)

| Checkpoint | Serves | Real-server evidence (2026-10-06, `omnismall` jobs; M5 until the M3 allowlist reaches the queue daemon) |
|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` (3.3 GB, allowlisted for the M3) | audio, image, video in; text out | `real 2026-10-06 gemma-4-e2b-it-4bit`: spoken "pineapple" (Qwen3-TTS) answered on chat (stream and not), responses and `audio_url`; image + audio in one message ("red", "pineapple"); video (OpenCV frames; neither machine has ffmpeg); Realtime voice cascade with Qwen3-ASR + Qwen3-TTS on `/v1/realtime` and `/realtime`. Exact window checkpoints now reuse the prefix: M5 and M3 `apcomni-gemma-smoke-*` on `57bcba0e` hit 8,458/8,459 tokens beyond the 512-token window, including follow-up raw-ID parity; image-block Anthropic `cache_control` creates/reads 262 tokens. These are correctness receipts, not a general vision-quality score. |
| `Qwen3-Omni-30B-A3B-Instruct-4bit` (20 GB, M5 only) | audio, image, video in; text + speech out | `real 2026-10-06 Qwen3-Omni-30B-A3B-Instruct-4bit`: `/v1/omni/speech/stream` (text, audio-in, image-in), native speech-to-speech on both Realtime sockets, repeated audio hits the prefix cache (112 of 113 tokens). Repeated-image APC fixed with complete dense checkpoints: M5 `apcomni-omni-smoke-1007d` (`d0497e10`) hits 104/105 tokens and 96/127 on a follow-up; raw IDs equal cold and baseline. Image-block Anthropic `cache_control` creates/reads 54 tokens. Image+audio together remains unverified by this cache fix. |
| Qwen2.5-Omni | not served | mlx-vlm 0.7.6 has no `qwen2_5_omni` model; there is no small speech-out omni checkpoint Yunshu can serve, so speech out stays on the M5 with Qwen3-Omni. |

2026-10-07 cache receipts live under `/Volumes/P5Plus/yunshu-build/verify/runs/apcomni-*`;
long-context timing and the 200-item image-conditioned cache gate are recorded separately.
