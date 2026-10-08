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
| GET/POST/DELETE | `/v1/chat/completions/{id}` | Retrieve, update metadata or delete a completion created with `store: true`; list at `GET /v1/chat/completions`, input messages at `GET /v1/chat/completions/{id}/messages`. |
| POST | `/v1/evals` | [Local Evals](guides/EVALS.md): definitions, runs, cancellation and output items. |
| POST | `/v1/realtime/client_secrets` | Ephemeral keys for the Realtime WebSocket; beta session routes are also served. Transcription-only configuration is echoed but not executed; transcription secrets cannot create model responses. |
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
Native MLX model `pull`, alias `copy` / `create`, and `delete` are supported; registry
upload (`push`) remains unsupported. GGUF imports and arbitrary Modelfiles are rejected.

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

### POST `/v1/decisions` — typed decisions (OpenAI Decisions API)

Answers user-defined questions about shared evidence with probabilities, not prose; one forward pass of a
decision model (Cloudflare Clef in MLX format), nothing is generated. Same shape as `client.decisions.create(...)`
in the `openai` SDK (3.26+).

```python
d = client.decisions.create(
    model="Clef-MLX-4bit",
    input="The forecast says heavy rain all day.",
    questions=[
        {"type": "predicate", "instructions": "Is it raining?", "name": "rain"},
        {"type": "choice", "instructions": "Pick the activity.", "choices": [
            {"value": "hike", "description": "outdoors"}, {"value": "museum", "description": "indoors"}]},
        {"type": "score", "instructions": "How wet?", "levels": [{"label": "dry"}, {"label": "wet"}]},
    ],
)
```

`answers` come back in question order: `predicate` (`probability`), `choice` (`choice`, `confidence`, per-value
`probabilities`; values keep their JSON type) and `score` (`score`, the probability-weighted level index, plus per-level
`probabilities`). A question whose logits are not finite is answered `{"type": "refusal"}`. `input` may hold inline
`input_image` parts (base64 data URLs only). The model sees choices sorted by value, so the order you send them in does
not move the probabilities. The input (state, images and schema) is limited to 16384 tokens; longer is a 400.
`POST /v1/systemone` takes the TypeSafe Jev / System One wire (`state`, map-keyed `questions` with `noul` / `choice` /
`score` and `criteria`) on the same engine.

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

**EmbeddingGemma 2** (`google/embeddinggemma-2`, 768-d, 8K tokens; text, image, audio, video and
mixes in one space). Extensions on the OpenAI request, all optional, SDKs keep working:

| Field | Meaning |
|---|---|
| `input` item `{text?, image?, audio?, video?, task?, instruction?}` | Media is a URL, file path or data URI (one value or a list; audio = PCM WAV, any rate, resampled to 16 kHz mono; `video` = a list of frames, sampled by the caller). `text` may carry `<\|image\|>` / `<\|audio\|>` / `<\|video\|>` markers that place each media item in order; without markers the order is the key order of the item. |
| `messages` | vLLM-style chat form, ONE embedding per request: content parts `text`, `image_url`, `input_audio` (base64 WAV), `audio_url`, `{"type":"video","frames":[...]}`. Use `input` or `messages`, not both. |
| `task` | A prompt of the model (`SearchQuery`, `Document`, `QuestionAnswering`, `FactChecking`, `CodeRetrieval`, `Classification`, `Clustering`, `SentenceSimilarity`); unset = the raw text, as `SentenceTransformer.encode` without a prompt. A per-item `task` wins. Applies to text only. |
| `instruction` | A literal text prefix instead of a named task. |
| `dimensions` | Matryoshka: the leading values, re-normalised (128, 256, 512, 768 are the trained sizes). |

`usage.prompt_tokens` counts the real sequence (text tokens plus image / frame / audio soft tokens;
280 per image, 25 per second of audio). More than 8192 tokens is a 400. The vision and audio towers
load on the first image / audio request (about 0.3 / 0.6 GB more), the text tower alone is 0.55 GB.

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

## Web retrieval

Search and fetch server tools are described in [Web search](guides/WEB_SEARCH.md).
The Tavily-compatible API has its own `/tavily` base (no `/v1`); see [Tavily](guides/TAVILY.md).
