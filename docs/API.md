# API reference

Yunshu exposes an **OpenAI/Anthropic-compatible** HTTP API plus a few Yunshu-specific
endpoints for the native-omni and retrieval surfaces. Base URL is `http://<host>:<port>/v1`
(default port 8000). Auth: inference endpoints are open by default; local model lifecycle
operations are gated by `YUNSHU_AUTH_TOKEN` (see [CONFIGURATION.md](CONFIGURATION.md)). The complete route-by-route
status, the parameter coverage and the removed list are in [guides/API_SURFACE.md](guides/API_SURFACE.md).

For the standard OpenAI/Anthropic endpoints, use your existing SDK unchanged — the
shapes match the upstream spec. This page is a concise endpoint overview with non-standard examples; the linked
route matrix includes Files, Batches, Conversations, compaction and WebSocket details.

## OpenAI-compatible

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/chat/completions` | Chat. Tool-calling, JSON-schema/grammar (`response_format`), streaming, `logprobs`. Plus extras: `top_n_sigma`, `min_p`, `xtc_probability`/`xtc_threshold`, `spec_decode`. |
| POST | `/v1/completions` | Legacy text completion. |
| POST | `/v1/responses` | Responses API (+ `GET/POST /v1/responses/{id}`, `/cancel`, `DELETE`). |
| POST | `/v1/embeddings` | Text **and multimodal** embeddings — see [below](#post-v1embeddings-multimodal). |
| GET | `/v1/models`, `/v1/models/{id}` | List / describe loaded models. |
| POST | `/v1/audio/transcriptions`, `/v1/audio/translations` | ASR (Whisper / mlx-audio). |
| POST | `/v1/audio/speech` | TTS → audio. |
| POST | `/v1/images/generations` | Diffusion image gen (+ `edits`, `variations`, `generations/stream`). Inline `<lora:name:weight>` applies on every image route. |
| POST | `/tokenize`, `/detokenize` (also under `/v1`) | vLLM-schema tokenizer: `prompt` or chat `messages`; returns `count`, `max_model_len`, `tokens`. |
| POST | `/v1/responses/input_tokens` | Input token count for a Responses request. |

## Ollama-compatible

`/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/ps`, `/api/version` (NDJSON streaming),
verified with the `ollama` SDK. `pull` / `create` / `copy` / `delete` answer 501.

## Anthropic-compatible

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/messages` | Anthropic Messages API (system/messages, streaming). |
| POST | `/v1/messages/count_tokens` | Token counting. |

## Yunshu-specific

### POST `/v1/omni/speech/stream` — native speech-to-speech

Streams Qwen3-Omni Thinker text + Talker audio as Server-Sent Events. Uses a served Qwen3-Omni model or a separate model selected with
`YUNSHU_OMNI_MODEL`.

```jsonc
// request
{
  "text": "Say hello in one sentence.",  // text turn / system instruction
  "speaker": "Ethan",                     // optional; unknown speaker → 400 with the valid list
  "image_path": "scene.png",              // optional image input: local path, data: URI, or http(s) URL
  "audio_path": "question.wav"            // optional native speech-IN: local path, data: URI, or http(s) URL
}
```

SSE events: `{"type":"text","delta":"…"}`, `{"type":"audio","delta":"<base64 pcm16>","sr":24000}`,
`{"type":"done", …}`, then `data: [DONE]`. See [examples/quickstart.py](../examples/quickstart.py).

### WS `/v1/realtime` — OpenAI-Realtime voice

Bidirectional voice agent over WebSocket (OpenAI-Realtime event protocol). With
`YUNSHU_REALTIME_OMNI=1` it runs the native Thinker→Talker path (speech-in → speech-out);
otherwise an ASR→LLM→TTS cascade. Multi-turn conversation context is preserved.
Tool-calling works on **both** paths — a tool turn emits `function_call` items and the
spoken JSON is suppressed (the voice doesn't read the tool call aloud).
See [examples/talk.py](../examples/talk.py) for a live mic↔speaker client.

### POST `/v1/rerank` — reranking

Cohere-style rerank. Uses a **true cross-encoder** when a `Qwen3-VL-Reranker` model is
loaded (joint query↔doc scoring), otherwise a bi-encoder cosine fallback.

```jsonc
// request — query/documents are str OR {text?, image?} objects (multimodal needs a cross-encoder)
{
  "model": "Qwen3-VL-Reranker-2B-8bit",
  "query": "What is the capital of France?",
  "documents": ["Paris is the capital of France.", {"image": "data:image/png;base64,…"}],
  "top_n": 3,
  "return_documents": true,
  "instruction": "Retrieve documents that answer the question."  // optional
}
// response: {"results": [{"index", "relevance_score", "document"?}], "usage": {…}}
```

### POST `/v1/embeddings` (multimodal)

Standard OpenAI embeddings, extended: `input` items may be `{text?, image?, instruction?}`
objects (image = url/path/data-uri) for a `Qwen3-VL-Embedding` model — text, image, and
cross-modal vectors land in one shared space. Optional top-level `instruction`. See
[examples/multimodal_embeddings.py](../examples/multimodal_embeddings.py).

### Other

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/score`, `/v1/pooling`, `/v1/classify` | Similarity / pooled embeddings / zero-shot classification (`classify` is embedding-cosine + softmax, not a trained classifier). |
| POST | `/v1/ocr` | OCR (GLM-OCR via mlx-vlm). |
| POST/GET | `/v1/mcp`, `/v1/mcp/sse`, `/v1/mcp/tools` | Model Context Protocol server + client surface. |

## Local model lifecycle and diagnostics

Denied unless `YUNSHU_AUTH_TOKEN` is set (or `YUNSHU_AUTH_DISABLED=1` locally): model lifecycle
(`POST /v1/models/load`, `/v1/models/unload/{id}`) and the `/debug/*` diagnostics, which are only
mounted with `YUNSHU_DEBUG_ROUTES=1`.

---

Health: `GET /health`. Prometheus metrics: `GET /metrics`. Endpoint shapes are verified against the
routers in `python/yunshu_gateway/routers/`.
