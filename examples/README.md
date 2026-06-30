# Examples

Three small, self-contained scripts. Start a Yunshu server first (see the
[main README](../README.md)), then run one.

## 🗣️ `talk.py` — actually talk to it

A real spoken conversation: **speak into your mic, hear the model speak back**,
live, in a loop. This is the flagship — native speech-to-speech (Qwen3-Omni),
your voice in as raw audio, the model's voice out, no text step in between.

```bash
# serve a Qwen3-Omni model — native voice is automatic (same model, no extra memory):
uv run yunshu serve -m /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit --port 8000

# sounddevice bundles PortAudio on macOS:
uv run --with sounddevice --with numpy --with websockets python examples/talk.py
```

Each turn: **press Enter, speak, press Enter again** to send. The model replies
out loud and the conversation remembers what was said. Ctrl-C to quit.

## ⚡ `quickstart.py` — does my server work?

No microphone. A 30-second HTTP tour: streams a spoken reply to `omni_out.wav`,
then a text chat and a streaming chat. Run this first to confirm the server is up.

```bash
uv run --with openai --with requests python examples/quickstart.py
```

The speech-to-speech section needs an omni model served (it reuses the served
model — no extra config); the text sections work against any `mlx-lm` model.

## 🔎 `multimodal_embeddings.py` — search across text and images

Embeds text, images, and cross-modal (text↔image) into one shared vector space,
and reranks documents (including image documents) with a true cross-encoder.

```bash
# server in multi-model mode with both retrieval models available:
#   $MODELS/Qwen3-VL-Embedding-2B-8bit
#   $MODELS/Qwen3-VL-Reranker-2B-8bit
YUNSHU_MULTI_MODEL=1 YUNSHU_MODELS_DIR=$MODELS uv run yunshu serve --port 8000

uv run --with requests python examples/multimodal_embeddings.py path/to/an_image.png
```

Pass an image path to run the cross-modal and image-reranking sections; without
one, it runs the text-only embedding part.

## 🎬 `video.py` — generate a video from a prompt

Wan 2.x / LTX-2 text-to-video (and image-to-video) on-device. Returns an MP4.

```bash
# needs the video extra; a model named *wan*/*ltx*/*video* routes to the video engine:
uv sync --extra video
uv run yunshu serve -m /path/to/Wan2.2-TI2V-5B-mlx --port 8000

uv run --with requests python examples/video.py "a fluffy cat in a sunny garden"
```

Writes `video_out.mp4`. Video diffusion is heavy — pass fewer `--frames` / `--steps`
(or a smaller `--width`/`--height`) to iterate faster, and `--image photo.jpg` to
animate a still (I2V).
