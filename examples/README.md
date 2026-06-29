# Examples

Runnable, copy-paste scripts against a local Yunshu server. Each is self-contained
and prints what it's doing. Start a server first (see the [main README](../README.md)),
then run a script.

| Script | What it shows | Needs |
|---|---|---|
| [`quickstart.py`](quickstart.py) | Every endpoint at a glance — native omni speech-out (writes `omni_out.wav`), chat, streaming chat, plus the text/vision/ASR/TTS surface | `requests`, `openai`; omni section needs `YUNSHU_OMNI_MODEL` |
| [`realtime_voice.py`](realtime_voice.py) | Bidirectional **speech-to-speech** over the OpenAI-Realtime WebSocket — streams a spoken WAV in, assembles the model's spoken reply into `reply.wav` | `websockets`; server with `YUNSHU_OMNI_MODEL` + `YUNSHU_REALTIME_OMNI=1` |
| [`multimodal_embeddings.py`](multimodal_embeddings.py) | Text / image / **cross-modal** embeddings in one shared space, plus true **cross-encoder reranking** (including image documents) | `requests`; a `Qwen3-VL-Embedding` + `Qwen3-VL-Reranker` model, multi-model mode |

## Running them

```bash
pip install openai requests websockets

# 1) all-endpoints tour (omni section needs an omni model on the server)
python examples/quickstart.py

# 2) speak in, hear the model speak back (one native S2S turn)
python examples/realtime_voice.py question.wav            # ws://localhost:8000
python examples/realtime_voice.py question.wav ws://host:8000

# 3) multimodal embeddings + reranking (pass an image to run the cross-modal parts)
python examples/multimodal_embeddings.py path/to/an_image.png
```

Each script has a header comment with the exact server command it expects. The
flagship speech-to-speech paths (`quickstart.py` omni section, `realtime_voice.py`)
need a `Qwen3-Omni` model; the rest run against any `mlx-lm` / `mlx-vlm` model.
