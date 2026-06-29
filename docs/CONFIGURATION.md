# Configuration

Yunshu is configured by environment variables (plus a few `yunshu serve` flags).
This is the **curated, user-facing** set — the knobs you actually reach for. The
codebase has many more internal/experimental flags; the ones here are the stable,
supported surface.

Booleans accept `1`/`true`/`yes` (case-insensitive); anything else (or unset) is off.

## Model selection

| Variable | Default | Meaning |
|---|---|---|
| `YUNSHU_MODEL` | _(unset)_ | Path/id of the single model to serve (single-model mode). The chat endpoint serves it under **any** requested model name. |
| `YUNSHU_OMNI_MODEL` | _(unset)_ | Path/id of the Qwen3-Omni model backing the native speech endpoints (`/v1/omni/speech/stream`, realtime omni). Loads independently of `YUNSHU_MODEL`. |
| `YUNSHU_MULTI_MODEL` | off | Enable multi-model mode: discover + load models on demand from `YUNSHU_MODELS_DIR`. Ignored if `YUNSHU_MODEL` is set (single-model wins). |
| `YUNSHU_MODELS_DIR` | _(unset)_ | Directory of model subfolders to auto-discover in multi-model mode. Each subfolder name becomes its model id. |

## Server & runtime

| Variable | Default | Meaning |
|---|---|---|
| `YUNSHU_REALTIME_OMNI` | off | Use the native Qwen3-Omni Thinker→Talker path on the `WS /v1/realtime` socket (speech-to-speech). Off → the realtime socket uses the ASR→LLM→TTS cascade. |
| `YUNSHU_OMNI_PRELOAD` | on | Warm the omni model at boot (compiles Talker kernels so the first request isn't cold). Set `0` to defer loading to first use. |
| `YUNSHU_DEFAULT_MAX_TOKENS` | `512` | Default completion length when a request omits `max_tokens`. |
| `YUNSHU_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins. |

`yunshu serve` flags: `-m/--model <path>` (= `YUNSHU_MODEL`), `--port <n>`, `--host <addr>`.

## Auth

Inference endpoints (chat/completions, embeddings, …) are open by default for easy
local use. Admin endpoints (model load/unload, profiling, dashboard) are denied
until a token is set.

| Variable | Default | Meaning |
|---|---|---|
| `YUNSHU_AUTH_TOKEN` | _(unset)_ | Bearer token enabling the admin endpoints. Unset → admin endpoints are denied; inference stays open. |
| `YUNSHU_AUTH_DISABLED` | off | Disable auth entirely (admin endpoints open too). Local-only convenience — do not expose the server publicly with this on. |

## Decoding & optimization

All of these are **safe defaults / opt-in** — Yunshu never silently changes output
quality. The single-request fast path (`mlx-lm generate_step`) with KV-prefix
caching is always on.

| Variable | Default | Meaning |
|---|---|---|
| `YUNSHU_NGRAM_DEFAULT` | on (`1`) | n-gram speculative decode on greedy requests. **Lossless** — the verifier accepts only the model's own argmax. Set `0` to opt out. |
| `YUNSHU_SPEC_PROPOSER` | `ngram` | Speculative proposer family: `ngram` (default, fastest on M-series) or `suffix` (SuffixDecoding; lossless but slower here). |
| `YUNSHU_TOP_N_SIGMA` | `0` (off) | Server-wide top-nσ sampler (ACL 2025) — keep only logits within n·σ of the max. Per-request `"top_n_sigma"` on `/v1/chat/completions` overrides this. |
| `YUNSHU_QUANT_MODE` | off | In-memory weight quant at load: `mxfp4` / `nvfp4` / `mxfp8` / `affine` (via mlx-lm `nn.quantize`). |
| `YUNSHU_KV_QUANT_BITS` | auto | Force KV-cache quant bits (`2`/`3`/`4`/`8`). Unset → auto-engages only when the cache would dominate bandwidth (very long context). |
| `YUNSHU_JUMP_FORWARD` | off | Jump-forward decoding for JSON-schema/grammar-constrained output (emits FSM-forced structural tokens without a per-token forward). |
| `YUNSHU_GPU_SAMPLER` | off | On-GPU Gumbel-max sampler (no per-token GPU→CPU sync). Default numpy sampler avoids mlx-lm's PRNG-compile-cache trap. |
| `YUNSHU_SPEC_PREFILL` | off | Sparse speculative prefill (needs a draft model). |
| `YUNSHU_ENGINE_LOOP` | off | Legacy continuous-batching loop instead of the single-request fast path. Not the supported path — for experiments only. |

## Embeddings

| Variable | Default | Meaning |
|---|---|---|
| `YUNSHU_ANE_EMBEDDINGS` | off | Compute embeddings on the Apple Neural Engine via CoreML (lower latency, less GPU contention) when available. |

---

This reference is curated. To see every flag the code reads:

```bash
grep -rhoE "YUNSHU_[A-Z0-9_]+" python/ | sort -u
```

Anything not listed above is internal/experimental and may change without notice.
