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
