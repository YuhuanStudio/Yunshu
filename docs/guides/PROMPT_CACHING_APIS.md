# Prompt-caching APIs — the three vendor paradigms

There are three distinct ways the major API vendors expose prompt caching. Yunshu
implements the first two on top of the same underlying KV machinery (the 4-tier
`KVPrefixCache` + the fast-path prefix reuse — see `KV_CACHE_MATRIX.md`).

| paradigm | vendor | client surface | who decides what's cached |
|---|---|---|---|
| **Automatic / implicit** | OpenAI | none — just resend the prefix | the server (prefix match) |
| **Explicit hints (breakpoints)** | Anthropic | `cache_control: {type: ephemeral}` | client marks, server caches |
| **Explicit named objects** | Google Gemini | not offered | client creates + references |

Both offered paradigms READ the same automatic `KVPrefixCache`; they differ only in the WRITE
surface and the usage accounting.

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
- Source: the engine's `GenerationOutput.cached_tokens` (the KVPrefixCache hit).
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
  (`batched_engine._kv_breakpoint_token_positions`, which stores a prefix entry at
  each breakpoint), and builds `cache_creation_input_tokens` /
  `cache_read_input_tokens` from the engine's `cached_tokens`.
- **Verified live:** `cache_read_input_tokens` populated on a repeat with the same
  `cache_control` block.

## 3. Google Gemini — explicit named context cache (removed)

Yunshu no longer offers the `/v1/cachedContents` handle API or the `cached_content` request
field. The automatic prefix cache (section 1) and Anthropic `cache_control` (section 2)
cover the same reuse without a separate write/read protocol.

## Design note — why one engine, three surfaces

The actual KV reuse is ONE mechanism (automatic prefix matching in
`KVPrefixCache`). The vendor "paradigms" are just different **control surfaces +
accounting** over it:
- OpenAI = no control surface (resend prefix).
- Anthropic = hint where the durable breakpoints are (so they survive eviction
  preferentially).

So adding a new vendor's caching API is a thin gateway layer, not new engine work.
