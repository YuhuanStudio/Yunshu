# Model support and validation

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
| Other modalities | Audio, embeddings, image/video generation | Optional extras and model-specific routes; see API and configuration docs |

For a new validation result, record exact model repository/revision, quantization,
upstream versions, Yunshu SHA, chip/macOS, tested routes, tools/schema/streaming/
stop/logprobs/cancel, and cold/partial/full prompt reuse. Mark each case passed,
failed or untested and link its receipt. A capability advertised by code and a
feature proven on hardware are distinct. The release gate supplies acceptance
checks; this page does not invent results for untested checkpoints.

## Omni (audio, image, video in; speech out)

| Checkpoint | Serves | Real-server evidence (2026-10-06, `omnismall` jobs; M5 until the M3 allowlist reaches the queue daemon) |
|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` (3.3 GB, allowlisted for the M3) | audio, image, video in; text out | `real 2026-10-06 gemma-4-e2b-it-4bit`: spoken "pineapple" (Qwen3-TTS) answered on chat (stream and not), responses and `audio_url`; image + audio in one message ("red", "pineapple"); video (OpenCV frames; neither machine has ffmpeg); Realtime voice cascade with Qwen3-ASR + Qwen3-TTS on `/v1/realtime` and `/realtime`. No prefix cache (sliding-window KV): a repeat is identical but not cached. |
| `Qwen3-Omni-30B-A3B-Instruct-4bit` (20 GB, M5 only) | audio, image, video in; text + speech out | `real 2026-10-06 Qwen3-Omni-30B-A3B-Instruct-4bit`: `/v1/omni/speech/stream` (text, audio-in, image-in), native speech-to-speech on both Realtime sockets, repeated audio hits the prefix cache (112 of 113 tokens). Open: a repeated image (alone or with audio) gets no cache hit. |
| Qwen2.5-Omni | not served | mlx-vlm 0.7.6 has no `qwen2_5_omni` model; there is no small speech-out omni checkpoint Yunshu can serve, so speech out stays on the M5 with Qwen3-Omni. |
