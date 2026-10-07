"""Official-recipe reference vectors for EmbeddingGemma 2 (sentence-transformers >= 6.1, torch fp32 on
CPU: no MLX / Metal, so it runs outside gpuq). Used as the independent oracle for the served vectors
(route_checks_media embed check) and as a CLI:

    egemma2_reference.py MODEL_DIR --out ref.json [--dims 256] [--task SearchQuery]

Items are a str (text) or {"text"?, "image"?, "audio"?, "video"?} with media given as file paths
(one or a list). sentence-transformers is imported from $EGEMMA2_ST_PATH when it is not installed.
The pure helpers (media loading, truncation, case building) are unit-tested on CPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import wave

SAMPLE_RATE = 16000


def l2(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def truncate(v, dims):
    """Matryoshka: keep the leading `dims` values, re-normalise (the model card's rule)."""
    return l2(list(v)[:dims]) if dims else list(v)


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def load_wav_16k(path):
    """Mono float32 samples in [-1, 1] at 16 kHz from a PCM16 WAV (resampled linearly if needed)."""
    import numpy as np

    with wave.open(path) as w:
        rate, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: want 16-bit PCM, got {sw * 8}-bit")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE:
        n = round(len(x) * SAMPLE_RATE / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(
            np.float32
        )
    return x


def to_st_input(item):
    """str | dict with paths -> the sentence-transformers input (audio decoded to arrays)."""
    if isinstance(item, str):
        return item
    out = {}
    for k, v in item.items():
        if k == "audio":
            v = [load_wav_16k(p) for p in v] if isinstance(v, list) else load_wav_16k(v)
        elif k == "video":  # a video is a list of frame files: hand over decoded frames
            from PIL import Image

            vids = v if isinstance(v, list) and v and isinstance(v[0], list) else [v]
            v = [[Image.open(f).convert("RGB") for f in frames] for frames in vids]
            v = v if len(v) > 1 else v[0]
        out[k] = v
    return out


def reference_embed_hf(model_dir, items, task=None, dims=None):
    """The same recipe on transformers alone (EmbeddingGemma2Model fp32 on CPU, mean pooling over
    the attention mask, L2): for hosts without sentence-transformers (the M3 laptop). Checked
    against sentence-transformers on the M5 (cosine > 0.9999)."""
    import torch
    from transformers import AutoModel, AutoProcessor

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "python"))
    from yunshu_engine.embedding_gemma2 import build_text, resolve_prompt

    with open(os.path.join(model_dir, "config_sentence_transformers.json")) as f:
        prompts = json.load(f)["prompts"]
    prefix = resolve_prompt(prompts, task, None)
    proc = AutoProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).eval()
    out = []
    for it in items:
        text, media = build_text({"text": it} if isinstance(it, str) else it, prefix)
        kw = {"text": [text], "return_tensors": "pt"}
        if "image" in media:
            kw["images"] = [[_pil(x) for x in media["image"]]]
        if "video" in media:
            kw["videos"] = [[_pil(f) for f in frames] for frames in media["video"]]
        if "audio" in media:
            kw["audio"] = [load_wav_16k(a) for a in media["audio"]]
        enc = proc(**kw)
        with torch.no_grad():
            h = model(**enc).last_hidden_state[0]
        m = enc["attention_mask"][0].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(0) / m.sum()
        out.append(truncate([float(x) for x in v], dims))
    return out


def _pil(x):
    from PIL import Image

    return x if hasattr(x, "convert") else Image.open(x).convert("RGB")


def reference_embed(model_dir, items, task=None, dims=None):
    """Vectors from sentence-transformers (official; the transformers-only recipe when it is not
    installed). `task` is a prompt name from config_sentence_transformers.json (SearchQuery,
    Document, ...); None encodes the raw text."""
    extra = os.environ.get("EGEMMA2_ST_PATH")
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    import torch

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return reference_embed_hf(model_dir, items, task, dims)

    m = SentenceTransformer(
        model_dir, device="cpu", model_kwargs={"torch_dtype": torch.float32}
    )
    out = []
    for it in items:
        v = m.encode(to_st_input(it), prompt_name=task if isinstance(it, str) else None)
        out.append(truncate([float(x) for x in v], dims))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--items", required=True, help="JSON file: list of str | dict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", type=int)
    ap.add_argument("--task")
    a = ap.parse_args()
    with open(a.items) as f:
        items = json.load(f)
    vecs = reference_embed(a.model_dir, items, a.task, a.dims)
    with open(a.out, "w") as f:
        json.dump(vecs, f)
    print(f"wrote {len(vecs)} vectors of {len(vecs[0])} to {a.out}")


if __name__ == "__main__":
    main()
