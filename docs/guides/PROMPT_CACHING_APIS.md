# Prompt-caching APIs — the three vendor paradigms

There are three distinct ways the major API vendors expose prompt caching. Yunshu
implements the first two over the serving path's prefix cache: mlx-vlm APC on the
VLM runner, `KVPrefixCache` on the text fast path — see [KV_CACHE_MATRIX](KV_CACHE_MATRIX.md).

| paradigm | vendor | client surface | who decides what's cached |
|---|---|---|---|
| **Automatic / implicit** | OpenAI | none — just resend the prefix | the server (prefix match) |
| **Explicit hints (breakpoints)** | Anthropic | `cache_control: {type: ephemeral}` | client marks, server caches |
| **Explicit named objects** | Google Gemini | not offered | client creates + references |

Both offered paradigms read the active engine's automatic prefix cache; the hints and
usage accounting are handled by the gateway. VLM APC stores exact hybrid checkpoints;
main adds optional WARM and lower storage tiers without another client API.

---

## 1. OpenAI — automatic implicit caching

No API surface. Resend the same prefix and the server reuses its KV. Reported in
the response:

```json
"usage": { "prompt_tokens": 3273,
           "prompt_tokens_details": { "cached_tokens": 3264 } }
```

- Handler: `routers/chat.py` (`usage.prompt_tokens_details.cached_tokens`),
  streaming via `streaming.py::format_openai_usage_chunk`, Responses API via
  `routers/responses.py` (`input_tokens_details.cached_tokens`).
- Source: the engine's `GenerationOutput.cached_tokens` (the active prefix-cache hit).
- **Verified live:** call 1 `cached=0`, call 2 (same prefix) `cached=3264/3273`.

## 2. Anthropic — explicit breakpoint hints

The client marks cache breakpoints with `cache_control` on a system/message block:

```json
"system": [{"type":"text","text":"<long doc>","cache_control":{"type":"ephemeral"}}]
```

Response usage splits the prompt into written-vs-read:

```json
"usage": { "input_tokens": 1,
           "cache_creation_input_tokens": <written this turn>,
           "cache_read_input_tokens": <served from cache> }
```

- Handler: `routers/anthropic.py` — `_extract_cache_control_hints()` parses the
  breakpoints, passes token positions to the engine
  (the text fast path stores explicit prefix entries; the VLM runner
  uses APC checkpoint positions), and builds `cache_creation_input_tokens` /
  `cache_read_input_tokens` from the engine's `cached_tokens`.
- **Verified live:** `cache_read_input_tokens` populated on a repeat with the same
  `cache_control` block.

## 3. Google Gemini — explicit named context cache (removed)

Yunshu no longer offers the `/v1/cachedContents` handle API or the `cached_content` request
field. The automatic prefix cache (section 1) and Anthropic `cache_control` (section 2)
cover the same reuse without a separate write/read protocol.

## Design note — why one cache per serving path, two offered surfaces

The actual reuse uses automatic prefix matching in the active engine's cache. The vendor "paradigms" are just different **control surfaces +
accounting** over it:
- OpenAI = no control surface (resend prefix).
- Anthropic = hint where the durable breakpoints are (so they survive eviction
  preferentially).

So adding a new vendor's caching API is a thin gateway layer, not new engine work.
