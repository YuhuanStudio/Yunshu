# Inference features


> Console docs: the maintained, translated version of this page is `/docs/internals/speculative-decoding` (source `frontend/docs/internals/speculative-decoding.mdx`). This file stays as the repository-side reference read by tooling.
What the engine does between the API and the GPU, and the settings that control it. Every
`YUNSHU_*` setting below is listed with its default in [Configuration](../CONFIGURATION.md); run
`yunshu config` to see the effective values. No performance number is given here: measured results
are in [Benchmarks](../BENCHMARKS.md) and [PERF_TREND](../reports/PERF_TREND.md).

## Which path serves a model

| Model | Path | Speculation | Prefix cache |
|---|---|---|---|
| Qwen3.5 / 3.6 / 3.8 family (mlx-vlm) | VLM batch runner, single-request speculative lane | DFlash2, MTP, prompt-copy | exact hybrid checkpoints (APC) |
| Other mlx-vlm models | same runner | none | APC where the cache layout allows it (including checkpoint-specific exact-window support; see model support) |
| Text-only mlx-lm models | single-request fast path | optional n-gram / suffix, Gemma 4 assistant, external draft (experimental) | mlx-lm prompt cache |

`/v1/models` reports what the loaded model supports.

## Lossless speculative decoding

- **DFlash2** block drafter: used automatically when a drafter matching the model is installed.
  `YUNSHU_SPEC_TREE` (`auto` by default) picks the draft-tree verifier for eligible greedy requests.
- **MTP**: the checkpoint's own multi-token-prediction head (`YUNSHU_MTP`, on by default; block size
  `YUNSHU_MTP_BLOCK_SIZE`).
- **Prompt-copy drafting**: proposes the continuation of text that already occurred in the prompt
  (`YUNSHU_SPEC_COPY_ROWS`, default 32 verify rows; `0` disables).
- **Batch-invariant kernels** make verification arithmetic identical to ordinary decode, so greedy
  output with speculation on is token-identical to speculation off.
- Override with `YUNSHU_VLM_DRAFT=mtp|off|<drafter path>`.
- Text models: `YUNSHU_NGRAM_DEFAULT=1` enables lossless n-gram speculation on greedy requests
  (`YUNSHU_SPEC_PROPOSER` = `ngram` or `suffix`); a request can also ask for it with `spec_decode`.
  Gemma 4 text models can use an assistant drafter directory via `YUNSHU_GEMMA4_ASSISTANT`.
- Experimental: `YUNSHU_ROUND_DRIVER` ([guide](ROUND_DRIVER.md)), `YUNSHU_MTP_ROW_EXACT`,
  `YUNSHU_SPEC_UNVERIFIED` with `YUNSHU_DRAFT_MODEL`. Experimental flags may be removed after measurement.
- Counters: `GET /v1/yunshu/spec-decode`, Prometheus `yunshu_spec_decode_*`, and per-request
  `x_yunshu.speculative`.

Limit: the speculative lane serves one request at a time; concurrent requests use the shared batch
without drafting.

## Prefix cache (APC)

Exact checkpoints of attention KV plus recurrent state, keyed by text tokens and by image-pixel or
audio-feature hashes, in tiers: HOT (RAM), optional WARM (compressed RAM), SSD (survives restarts),
optional extra storage tiers.

```bash
YUNSHU_VLM_APC_MEMORY_GB=12 YUNSHU_VLM_APC_DISK_GB=64 yunshu serve
yunshu cache status        # what is on disk
yunshu cache gc            # clean it
```

The console Cache page and `GET /v1/yunshu/cache` show entries, tiers, hits and lifecycle events;
`POST /v1/yunshu/cache/clear` drops the resident entries (refused while a request runs). Tier
details: [KV cache matrix](KV_CACHE_MATRIX.md). Responses report cached tokens in `usage`.
WARM int8/int4 encodings and KV precision are lossy and off by default.

## Structured output and constrained decoding

`response_format` (`json_object`, strict `json_schema`), `regex`, `choice` and Lark `grammar` are
enforced while decoding, including tool-call arguments. By default an in-house allowlist sampler is
used, with llguidance for schemas outside its subset. Options: `YUNSHU_GRAMMAR_BITMASK=1`
(xgrammar-style bitmask engine), `YUNSHU_JUMP_FORWARD=1` (emit grammar-forced tokens without a
forward pass), `YUNSHU_TOOL_GRAMMAR=1` (mask tool-call bodies). Unsupported constructs are an error,
not silently ignored. Responses report `x_yunshu.structured_output` (`requested`, `enforced`, engine).
Limit: enforcement guarantees the grammar, not that a `max_tokens` cut-off output is complete.

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"YOUR_MODEL","messages":[{"role":"user","content":"Name a color."}],
  "response_format":{"type":"json_schema","json_schema":{"name":"c","strict":true,
    "schema":{"type":"object","properties":{"color":{"type":"string"}},"required":["color"],"additionalProperties":false}}}}'
```

## Lossy memory options (off by default)

| Setting | Effect |
|---|---|
| `YUNSHU_KV_PRECISION=int8` | int8 KV for the Qwen3.5-family shared decode batch; smaller memory and bandwidth, small attention error |
| `YUNSHU_VLM_APC_WARM=int8\|int4` | compressed WARM prefix tier (`lossless` is exact) |
| `YUNSHU_QUANT_MODE` | quantize weights at load (mxfp4, nvfp4, mxfp8, affine) |

A lossy option never becomes a default because of a benchmark.

## Other engine options

- `YUNSHU_GPU_SAMPLER=1`: on-GPU sampling for text models. `YUNSHU_OVERLAP` (experimental): CPU/GPU overlap.
- `YUNSHU_PREFILL_STEP_SIZE`: prefill chunk size (lower it to cap the activation peak on small Macs).
- Large mixture-of-experts checkpoints load with quantization bits and group size derived from the
  tensor shapes when a community pack's config disagrees (DeepSeek-V4-Flash packs); custom TurboQuant
  packs are refused with an explicit error. Whether a given large checkpoint fits is shown by
  `GET /v1/yunshu/models/{id}/fit` and the console Models page.
- Memory accounting: `GET /v1/yunshu/memory` lists what holds memory; all `*_gb` fields are binary GiB
  ([units](API_EXTENSIONS.md#memory-units)).

## Split-invariant prefill and idle keep-warm

On the qualified Qwen3.8-27B / M5 Max / MLX 0.32.3 path (0.1.6 default) prefill uses one kernel
family for every piece length: NAX GEMM padded to 64 rows, BQ32 attention for any query length,
chunked GDN padded to 64 with identity updates. Prompt pieces and APC checkpoints are cut at
absolute 64-token boundaries (a new APC arithmetic namespace, so 0.1.5 checkpoints are not reused).
Cold, cached and rehearsed prefixes therefore yield the same tokens. Cold outputs may differ from
0.1.5 within the accuracy gate (n=240: 132 vs 131 correct). Related settings:
`YUNSHU_PREFILL_MATMUL`, `YUNSHU_PREFILL_GDN`, `YUNSHU_PREFILL_ATTENTION`.

Idle keep-warm: after a request the engine holds the wired-memory limit with a 1 Hz heartbeat
(a tiny matmul touching no model array) for up to 30 s, until the idle trim gives it back. Without
it Metal stops keeping the 27B weights resident after about 2 s and the next request re-pins them
(~200 ms). SSD restore reads checkpoints with MLX's native safetensors reader.

Deployment sizing (27B, 60-minute soak, 766 requests, 0 errors, peak 38.8 GiB, 0 pageouts): keep
the defaults on 128 GB hosts. On tight hosts set `YUNSHU_VLM_APC_MEMORY_GB=4` and
`YUNSHU_PREFILL_BUFFER_CACHE_GB=1` (peak 37.2 -> 31.5 GiB). `YUNSHU_MAX_MEMORY_GB` is only needed
for multi-model hosts.

## Next-turn rehearsal

In 0.1.5, `YUNSHU_NEXT_TURN_PREFILL=stride` is the default for eligible VLM text chats.
After a reply, idle work prefills full canonical atoms of the assistant reply and stores
APC checkpoints; the next request still computes the remainder and its new suffix.
Stride rehearsal skips boundaries at or before the completed prompt checkpoint: a short
reply that adds no full atom does not trigger a replay of an older prefix.
It costs idle GPU time and another cached prefix. A matching request can adopt work
already in progress (since 0.1.6 rehearsal starts 0.3 s after a reply and stops at the last 64-aligned position at or before the assistant end); a diverging request can wait for the current atom.
`off` disables it. `boundary` and `exact` add assistant-end splits and can change
multi-turn floating-point rounding; they are explicit opt-ins. This does not eliminate
the measured [immediate follow-up latency gap](../BENCHMARKS.md#where-yunshu-stands).

When a shared batch drains to one eligible greedy request, the runner can promote it
to the speculative lane at a step boundary. Concurrent rows do not draft together.
