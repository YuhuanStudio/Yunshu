# Prompt-caching APIs — the three vendor paradigms

There are three distinct ways the major API vendors expose prompt caching. Yunshu
now implements all three on top of the same underlying KV machinery (the 4-tier
`KVPrefixCache` + the fast-path prefix reuse — see `KV_CACHE_MATRIX.md`).

| paradigm | vendor | client surface | who decides what's cached |
|---|---|---|---|
| **Automatic / implicit** | OpenAI | none — just resend the prefix | the server (prefix match) |
| **Explicit hints (breakpoints)** | Anthropic | `cache_control: {type: ephemeral}` | client marks, server caches |
| **Explicit named objects** | Google Gemini | `POST /v1/cachedContents` → handle | client creates + references |

All three READ the same automatic `KVPrefixCache`; they differ only in the WRITE
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

## 3. Google Gemini — explicit named context cache (NEW)

The client explicitly **WRITES** a cache object and **READS** it by reference.

**Write** — create a handle (warms the KV prefix cache):
```
POST /v1/cachedContents
{ "model": "...", "system_instruction": "<long doc>", "ttl_seconds": 3600 }
→ { "name": "cachedContents/<id>", "usageMetadata": {"totalTokenCount": 3253},
    "expireTime": "...", "ttl": "3600.0s" }
```
(Also accepts OpenAI-style `messages` or Gemini-style `contents`.)

**Read** — reference the handle in a chat request; its stored content is prepended
so the automatic KVPrefixCache serves the warmed prefix:
```
POST /v1/chat/completions
{ "model":"...", "cached_content":"cachedContents/<id>",
  "messages":[{"role":"user","content":"..."}] }
→ usage.prompt_tokens_details.cached_tokens ≈ the cached doc length
```

**Manage:** `GET /v1/cachedContents` (list), `GET/PATCH/DELETE
/v1/cachedContents/{id}` (get / update-TTL / delete).

- Store: `explicit_cache.py::ExplicitContextCache` (name→content+TTL, thread-safe,
  TTL + capacity eviction, read-count). Router: `routers/cached_contents.py`.
  Read wiring: `routers/chat.py::_prepend_cached_content`.
- The handle governs *validity/TTL*; the KV tier governs *residency* — if the KV
  was LRU-evicted before a read it transparently re-prefills once.
- **Verified live (Qwen2.5-3B):** WRITE → handle (3253 tok); READ via
  `cached_content` → `cached_tokens=3200/3267` reused; GET/LIST/PATCH/DELETE work.

---

## Design note — why one engine, three surfaces

The actual KV reuse is ONE mechanism (automatic prefix matching in
`KVPrefixCache`). The vendor "paradigms" are just different **control surfaces +
accounting** over it:
- OpenAI = no control surface (resend prefix).
- Anthropic = hint where the durable breakpoints are (so they survive eviction
  preferentially).
- Gemini = a named, TTL'd handle the client manages explicitly.

So adding a new vendor's caching API is a thin gateway layer, not new engine work.
Tests: `tests/unit/test_explicit_cache.py`; live verification scripts use the
gateway on `:8011`.
