# Multimodal and retrieval endpoints

Optional extras install the dependencies (`pip install 'yunshu[audio]'`, `[vision]`, `[generation]`,
`[embeddings]`, `[omni]`). Each route needs a model that supports it; `/v1/models` says which. Coverage and
verification status per route: [API surface](API_SURFACE.md). Request shapes: [API reference](../API.md).

| Capability | Route | Extra | Notes |
|---|---|---|---|
| Images, audio and video as chat input | `/v1/chat/completions`, `/v1/responses`, `/v1/messages` | none (mlx-vlm models) | media enters the prefix-cache key |
| Qwen3-Omni speech-to-speech | `POST /v1/omni/speech/stream`, [example](../../examples/talk.py) | `omni` | text and audio out |
| Realtime voice | `WS /v1/realtime` (`/realtime`) | `audio` | server VAD, barge-in, native omni speech or an ASR, LLM, TTS cascade; ephemeral keys via `POST /v1/realtime/client_secrets`; optional WebRTC (`yunshu[webrtc]`) |
| Speech to text | `/v1/audio/transcriptions`, `/v1/audio/translations` | `audio` | translation needs a Whisper model |
| Text to speech | `/v1/audio/speech`, `GET /v1/audio/voices` | `audio` | local voice enrollment from audio samples for reference-audio models |
| OCR | `POST /v1/ocr` (GLM-OCR) | `vision` | `yunshu ocr`; also in the console |
| Image generation, editing, variations | `/v1/images/generations`, `/edits`, `/variations` | `generation` | `yunshu image`, `image-edit`, `image-variations` |
| Embeddings | `POST /v1/embeddings` | `embeddings` | text and, for supported models, image, audio and video ([example](../../examples/multimodal_embeddings.py)) |
| Rerank, score, classify, pooling | `/v1/rerank`, `/v1/score`, `/v1/classify`, `/v1/pooling` | `embeddings` | cross-encoder rerank and trained classification heads, vLLM semantics |
| Typed decisions | `/v1/decisions`, `/v1/systemone` | none | [guide](DECISIONS.md) |

## Examples

```bash
yunshu speak "Hello from Yunshu" -o hello.wav     # needs a TTS model served
yunshu transcribe hello.wav                       # needs an ASR model
yunshu ocr page.png
yunshu embed "a sentence"
yunshu rerank "query" "doc one" "doc two" --top-n 1
```

## Limits

- Speech, ASR, TTS, image generation and OCR are supported capabilities, not the tuned path; features
  differ per checkpoint and are verified per model (see [model support](MODEL_SUPPORT.md)).
- Realtime transcription-only session configuration is echoed but not executed as a transcription-only session.
- Models with sliding-window KV (for example Gemma 4) have no prefix cache.
- A repeated image alone does not yet hit the cache on Qwen3-Omni.
