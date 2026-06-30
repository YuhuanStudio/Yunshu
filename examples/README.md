# Examples

Three small, self-contained scripts. Start a Yunshu server first (see the
[main README](../README.md)), then run one.

## 🗣️ `talk.py` — actually talk to it

A real spoken conversation: **speak into your mic, hear the model speak back**,
live, in a loop. This is the flagship — native speech-to-speech (Qwen3-Omni),
your voice in as raw audio, the model's voice out, no text step in between.

```bash
# serve a Qwen3-Omni model — native voice is automatic (same model, no extra memory):
yunshu serve -m /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit --port 8000

pip install sounddevice numpy websockets    # sounddevice bundles PortAudio on macOS
python examples/talk.py
```

Each turn: **press Enter, speak, press Enter again** to send. The model replies
out loud and the conversation remembers what was said. Ctrl-C to quit.

## ⚡ `quickstart.py` — does my server work?

No microphone. A 30-second HTTP tour: streams a spoken reply to `omni_out.wav`,
then a text chat and a streaming chat. Run this first to confirm the server is up.

```bash
pip install openai requests
python examples/quickstart.py
```

The speech-to-speech section needs `YUNSHU_OMNI_MODEL` set on the server; the
text sections work against any `mlx-lm` model.

## 🔎 `multimodal_embeddings.py` — search across text and images

Embeds text, images, and cross-modal (text↔image) into one shared vector space,
and reranks documents (including image documents) with a true cross-encoder.

```bash
# server in multi-model mode with both retrieval models available:
#   $MODELS/Qwen3-VL-Embedding-2B-8bit
#   $MODELS/Qwen3-VL-Reranker-2B-8bit
YUNSHU_MULTI_MODEL=1 YUNSHU_MODELS_DIR=$MODELS yunshu serve --port 8000

pip install requests
python examples/multimodal_embeddings.py path/to/an_image.png
```

Pass an image path to run the cross-modal and image-reranking sections; without
one, it runs the text-only embedding part.
