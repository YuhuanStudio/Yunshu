# Model support and validation

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

## Qwen3.8-Flash-Next under 80 GB (qwen4_exp)

`Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` is 106 GB on disk, of which 32 GB is the per-layer n-gram embedding (PLE). Yunshu reads PLE rows from the SSD instead of loading the table, which leaves 74.3 GB of resident weights (the 1.5 GB MTP head included).

1. Build the external-PLE view once (hard links, nothing is copied, the original pack is untouched; source and view must be on the same volume): `python scripts/research/flashnext80_prepare_view.py <pack> <pack>-ple-ssd`.
2. Serve the view: `YUNSHU_MAX_MEMORY_GB=74.5 yunshu serve -m <pack>-ple-ssd`. The MTP head is used automatically (one draft per round); the APC SSD tier is on by default.

With a memory ceiling set, Yunshu keeps the freed-buffer pool at 0.5 GiB and leaves the vision tower unloaded until the first image request. Both are lossless; set `YUNSHU_PREFILL_BUFFER_CACHE_GB` to override the pool.

| Context (M5 Max 128 GB, MTP on, whole-system memory delta) | Steady | Cold-prefill peak | Decode |
|---|---:|---:|---:|
| 1K | 75-76 GB | 77 GB | 45-50 tok/s |
| 8K | 77 GB | 79 GB | 42 tok/s |
| 32K | 75 GB | 80 GB | 37-38 tok/s |
| about 169K, SSD APC hit (second request) | 75 GB | 81 GB | 22 tok/s, first token in 1.2 s |
| about 169K, cold | 75 GB | 84-85 GB | 21 tok/s |

A cold prefill beyond roughly 32K needs about 83-85 GB at its peak (the extra is prefill working memory); the same prompt served from the SSD prompt cache stays near 81 GB. Output is identical with speculation on and off and with a prompt-cache hit or miss. MTP gains about 20% here because every extra verify row reads more experts. Thinking-mode MMLU-Pro (300 questions, greedy): TBD.

## Omni (audio, image, video in; speech out)

| Checkpoint | Serves | Real-server evidence (2026-10-06, `omnismall` jobs; M5 until the M3 allowlist reaches the queue daemon) |
|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` (3.3 GB, allowlisted for the M3) | audio, image, video in; text out | `real 2026-10-06 gemma-4-e2b-it-4bit`: spoken "pineapple" (Qwen3-TTS) answered on chat (stream and not), responses and `audio_url`; image + audio in one message ("red", "pineapple"); video (OpenCV frames; neither machine has ffmpeg); Realtime voice cascade with Qwen3-ASR + Qwen3-TTS on `/v1/realtime` and `/realtime`. No prefix cache (sliding-window KV): a repeat is identical but not cached. |
| `Qwen3-Omni-30B-A3B-Instruct-4bit` (20 GB, M5 only) | audio, image, video in; text + speech out | `real 2026-10-06 Qwen3-Omni-30B-A3B-Instruct-4bit`: `/v1/omni/speech/stream` (text, audio-in, image-in), native speech-to-speech on both Realtime sockets, repeated audio hits the prefix cache (112 of 113 tokens). Open: a repeated image (alone or with audio) gets no cache hit. |
| Qwen2.5-Omni | not served | mlx-vlm 0.7.6 has no `qwen2_5_omni` model; there is no small speech-out omni checkpoint Yunshu can serve, so speech out stays on the M5 with Qwen3-Omni. |
