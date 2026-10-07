# Upstream (inspired): huggingface/transformers (Apache-2.0) models/embedding_gemma2; Blaizzy/mlx-vlm (MIT) gemma4 vision/audio towers (imported)
"""EmbeddingGemma 2 (Google, ``embedding_gemma2``) in MLX: text, image, audio and video embeddings.

The text tower is a 24-layer bidirectional Gemma-4-style encoder (per-layer-embedding gate, local /
global attention with different head sizes, q/k/v norms, scale 1.0) followed by mean pooling, a
512 -> 768 projection and L2 normalisation. Images / video frames / audio go through the Gemma 4
vision and audio towers that mlx-vlm already ships (imported, not copied) and are scattered into the
text sequence at their placeholder tokens. The text tower is written here from the architecture of
the Hugging Face ``transformers`` implementation (Apache-2.0).

Loading is strict: a model parameter without a checkpoint tensor raises. The vision / audio towers
load lazily from the same safetensors file on the first image / audio / video request.
Run in bfloat16 or float32; float16 overflows (model card).
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Any

import numpy as np

# Prompts of the model's sentence-transformers config (config_sentence_transformers.json).
DEFAULT_PROMPTS = {
    "SearchQuery": "task: search result | query: ",
    "Document": "title: none | text: ",
    "QuestionAnswering": "task: question answering | query: ",
    "FactChecking": "task: fact checking | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
}
MAX_TOKENS = 8192
MRL_DIMS = (128, 256, 512, 768)
PLACEHOLDERS = {"image": "<|image|>", "video": "<|video|>", "audio": "<|audio|>"}


# ── pure helpers (CPU, unit-tested) ─────────────────────────────────────────────────────


def resolve_prompt(prompts: dict[str, str], task: str | None, instruction: str | None):
    """Text prefix: an explicit ``instruction`` wins, else the named task, else none."""
    if instruction:
        return instruction
    if not task:
        return ""
    if task not in prompts:
        raise ValueError(f"unknown task {task!r}; one of {sorted(prompts)}")
    return prompts[task]


def build_text(item: dict[str, Any], prefix: str = "") -> tuple[str, dict[str, list]]:
    """Item {"text"?, "image"?, "video"?, "audio"?} -> (text with placeholders, media lists).

    A text that already carries ``<|image|>`` / ``<|video|>`` / ``<|audio|>`` places each media
    item at its marker, in order (the model card's interleaving). Otherwise the sequence follows
    the key order of the item, like the official chat template: {"text", "image"} is the text then
    the image; {"image", "text"} the image then the text. The prefix applies to text only.
    """
    media: dict[str, list] = {}
    for kind in PLACEHOLDERS:
        v = item.get(kind)
        if v is None:
            continue
        media[kind] = list(v) if isinstance(v, (list, tuple)) else [v]
    text = item.get("text") or ""
    if any(m in text for m in PLACEHOLDERS.values()):
        for kind, ph in PLACEHOLDERS.items():
            have, want = text.count(ph), len(media.get(kind, []))
            if have != want:
                raise ValueError(
                    f"text has {have} {ph} markers but {want} {kind} inputs were given"
                )
        return prefix + text, media
    parts: list[str] = []
    for key in item:
        if key == "text":
            parts.append(prefix + text if text else "")
        elif key in PLACEHOLDERS and key in media:
            parts.append(PLACEHOLDERS[key] * len(media[key]))
    s = "".join(parts)
    if not s:
        raise ValueError("an input needs text or at least one image / audio / video")
    return s, media


def truncate_normalize(v: np.ndarray, dims: int | None) -> np.ndarray:
    """Matryoshka: leading ``dims`` values, re-normalised (the model card's rule)."""
    if dims is not None:
        if dims not in MRL_DIMS:
            raise ValueError(f"dimensions must be one of {MRL_DIMS}, got {dims}")
        v = v[..., :dims]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.asarray(v / np.maximum(n, 1e-12))


def sliding_mask(length: int, window: int) -> np.ndarray:
    """Bidirectional sliding window: |i - j| <= window."""
    idx = np.arange(length)
    return np.abs(idx[:, None] - idx[None, :]) <= window


def plan_batches(lengths: list[int], budget: int = 16384) -> list[list[int]]:
    """Group item indexes (shortest first) so padded tokens per batch stay within ``budget``."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and (len(cur) + 1) * lengths[i] > budget:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


# ── model ───────────────────────────────────────────────────────────────────────────────


def _build():
    import mlx.core as mx
    import mlx.nn as nn

    class Attention(nn.Module):
        def __init__(self, cfg, idx):
            super().__init__()
            lc = cfg.get("per_layer_config", {}).get(f"{idx:02d}") or {}
            self.head_dim = lc.get("head_dim", cfg["head_dim"])
            self.kv_heads = lc.get("num_key_value_heads", cfg["num_key_value_heads"])
            self.heads = cfg["num_attention_heads"]
            self.eps = cfg["rms_norm_eps"]
            h = cfg["hidden_size"]
            self.q_proj = nn.Linear(h, self.heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(h, self.kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(h, self.kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.heads * self.head_dim, h, bias=False)
            self.q_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
            kind = cfg["layer_types"][idx]
            self.sliding = kind == "sliding_attention"
            self.theta = cfg["rope_parameters"][kind]["rope_theta"]

        def __call__(self, x, mask):
            b, n, _ = x.shape
            q = self.q_proj(x).reshape(b, n, self.heads, self.head_dim)
            k = self.k_proj(x).reshape(b, n, self.kv_heads, self.head_dim)
            v = self.v_proj(x).reshape(b, n, self.kv_heads, self.head_dim)
            q = self.q_norm(q).transpose(0, 2, 1, 3)
            k = self.k_norm(k).transpose(0, 2, 1, 3)
            v = mx.fast.rms_norm(v, None, self.eps).transpose(0, 2, 1, 3)
            q = mx.fast.rope(
                q,
                self.head_dim,
                traditional=False,
                base=self.theta,
                scale=1.0,
                offset=0,
            )
            k = mx.fast.rope(
                k,
                self.head_dim,
                traditional=False,
                base=self.theta,
                scale=1.0,
                offset=0,
            )
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
            return self.o_proj(o.transpose(0, 2, 1, 3).reshape(b, n, -1))

    class MLP(nn.Module):
        def __init__(self, h, i):
            super().__init__()
            self.gate_proj = nn.Linear(h, i, bias=False)
            self.up_proj = nn.Linear(h, i, bias=False)
            self.down_proj = nn.Linear(i, h, bias=False)

        def __call__(self, x):
            return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))

    class PLEBlock(nn.Module):
        def __init__(self, h, p, eps):
            super().__init__()
            self.per_layer_input_gate = nn.Linear(h, p, bias=False)
            self.per_layer_projection = nn.Linear(p, h, bias=False)
            self.post_per_layer_input_norm = nn.RMSNorm(h, eps=eps)

        def __call__(self, x, per_layer_input):
            g = nn.gelu_approx(self.per_layer_input_gate(x)) * per_layer_input
            return x + self.post_per_layer_input_norm(self.per_layer_projection(g))

    class Layer(nn.Module):
        def __init__(self, cfg, idx):
            super().__init__()
            h, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
            self.self_attn = Attention(cfg, idx)
            self.mlp = MLP(h, cfg["intermediate_size"])
            self.input_layernorm = nn.RMSNorm(h, eps=eps)
            self.post_attention_layernorm = nn.RMSNorm(h, eps=eps)
            self.pre_feedforward_layernorm = nn.RMSNorm(h, eps=eps)
            self.post_feedforward_layernorm = nn.RMSNorm(h, eps=eps)
            self.layer_scalar = mx.ones((1,))
            self.ple_block = PLEBlock(h, cfg["hidden_size_per_layer_input"], eps)

        def __call__(self, x, ple_in, mask):
            x = x + self.post_attention_layernorm(
                self.self_attn(self.input_layernorm(x), mask)
            )
            x = x + self.post_feedforward_layernorm(
                self.mlp(self.pre_feedforward_layernorm(x))
            )
            return self.ple_block(x, ple_in) * self.layer_scalar

    class PLE(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.n = cfg["num_hidden_layers"]
            self.p = cfg["hidden_size_per_layer_input"]
            self.scale = cfg["hidden_size"] ** -0.5
            self.per_layer_model_projection = nn.Linear(
                cfg["hidden_size"], self.n * self.p, bias=False
            )
            self.per_layer_projection_norm = nn.RMSNorm(self.p, eps=cfg["rms_norm_eps"])

        def __call__(self, embeds):
            x = self.per_layer_model_projection(embeds) * self.scale
            return self.per_layer_projection_norm(
                x.reshape(*embeds.shape[:-1], self.n, self.p)
            )

    class TextModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.window = cfg["sliding_window"]
            self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
            self.layers = [Layer(cfg, i) for i in range(cfg["num_hidden_layers"])]
            self.norm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
            self.ple = PLE(cfg)
            self.embedding_projection = nn.Linear(
                cfg["hidden_size"], cfg["embedding_dim"], bias=False
            )

        def embed(self, ids):
            e = self.embed_tokens(ids)
            # bf16 rounds sqrt(512)=22.627 to 22.625 in the reference too
            return e * mx.array(math.sqrt(self.cfg["hidden_size"])).astype(e.dtype)

        def __call__(self, embeds, valid):
            """embeds [B, L, H], valid [B, L] bool -> mean-pooled, projected [B, embedding_dim] fp32."""
            b, n, _ = embeds.shape
            ple = self.ple(embeds)
            full = mx.broadcast_to(valid[:, None, None, :], (b, 1, n, n))
            idx = mx.arange(n)
            near = (mx.abs(idx[:, None] - idx[None, :]) <= self.window)[None, None]
            local = full & near
            x = embeds
            for i, layer in enumerate(self.layers):
                x = layer(
                    x, ple[:, :, i, :], local if layer.self_attn.sliding else full
                )
            x = self.norm(x).astype(mx.float32)
            w = valid[:, :, None].astype(mx.float32)
            pooled = (x * w).sum(axis=1) / w.sum(axis=1)
            # projecting after the mean equals projecting every token first (linear, no bias)
            return pooled @ self.embedding_projection.weight.astype(mx.float32).T

    class Embedder(nn.Module):
        def __init__(self, in_dim, out_dim, eps):
            super().__init__()
            self.embedding_projection = nn.Linear(in_dim, out_dim, bias=False)
            self.eps = eps

        def __call__(self, x):
            return self.embedding_projection(mx.fast.rms_norm(x, None, self.eps))

    return TextModel, Embedder


def _sanitize_audio(weights: dict, channels: tuple) -> dict:
    """PyTorch conv layouts -> MLX (same rule as mlx-vlm's gemma4 sanitize)."""
    out = {}
    for k, v in weights.items():
        if "subsample_conv_projection" in k and "conv.weight" in k and v.ndim == 4:
            expect = 1 if ".layer0." in k else channels[0]
            if v.shape[-1] != expect:
                v = v.transpose(0, 2, 3, 1)
        if "depthwise_conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
            v = v.transpose(0, 2, 1)
        out[k] = v
    return out


class EmbeddingGemma2:
    """Loaded model: ``embed_items`` -> unit vectors (numpy float32, [N, 768 or dims])."""

    def __init__(self, model_dir: str, dtype: str = "bfloat16"):
        import mlx.core as mx

        self.dir = model_dir
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)
        if self.config.get("model_type") != "embedding_gemma2":
            raise ValueError(f"{model_dir}: not an embedding_gemma2 checkpoint")
        if dtype not in ("bfloat16", "float32"):
            raise ValueError(
                "EmbeddingGemma 2 overflows float16: use bfloat16 or float32"
            )
        self.dtype = getattr(mx, dtype)
        self.tc = self.config["text_config"]
        self.prompts = dict(DEFAULT_PROMPTS)
        st = os.path.join(model_dir, "config_sentence_transformers.json")
        if os.path.isfile(st):
            with open(st) as f:
                self.prompts.update(json.load(f).get("prompts", {}))
        self._files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not self._files:
            raise FileNotFoundError(f"no safetensors in {model_dir}")
        self._weights: dict | None = None
        self.vision: Any = None
        self.embed_vision: Any = None
        self.audio: Any = None
        self.embed_audio: Any = None
        self._processor = None
        text_cls, self._embedder_cls = _build()
        self.text = text_cls(self.tc)
        self._strict_load(self.text, "language_model.")

    # -- loading ------------------------------------------------------------------------
    def _all_weights(self) -> dict:
        import mlx.core as mx

        if self._weights is None:  # lazy mmap: only the tensors that are used get read
            w: dict = {}
            for f in self._files:
                loaded: Any = mx.load(f)
            w.update(loaded)
            self._weights = w
        return self._weights

    def _strict_load(self, module, prefix: str, extra=None):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        w = {
            k[len(prefix) :]: v
            for k, v in self._all_weights().items()
            if k.startswith(prefix)
        }
        if extra:
            w = extra(w)
        flat: Any = tree_flatten(module.parameters())
        want = {str(k) for k, _ in flat}
        missing = sorted(want - set(w))
        if missing:
            raise ValueError(
                f"{self.dir}: {len(missing)} {prefix}* parameters have no checkpoint tensor "
                f"(e.g. {missing[:3]}); refusing to serve random weights"
            )
        module.load_weights(
            [
                (k, w[k] if k.endswith("layer_scalar") else w[k].astype(self.dtype))
                for k in want
            ],
            strict=True,
        )
        mx.eval(module.parameters())

    def _ensure_vision(self):
        if self.vision is not None:
            return
        vc = self.config.get("vision_config")
        if not vc:
            raise ValueError("this checkpoint has no vision tower")
        from mlx_vlm.models.gemma4.config import VisionConfig
        from mlx_vlm.models.gemma4.vision import VisionModel

        vision = VisionModel(VisionConfig.from_dict(vc))
        self._strict_load(vision, "vision_tower.")
        emb = self._embedder_cls(
            vc["hidden_size"], self.tc["hidden_size"], vc["rms_norm_eps"]
        )
        self._strict_load(emb, "embed_vision.")
        self.vision, self.embed_vision = vision, emb

    def _ensure_audio(self):
        if self.audio is not None:
            return
        ac = self.config.get("audio_config")
        if not ac:
            raise ValueError("this checkpoint has no audio tower")
        from mlx_vlm.models.gemma4.audio import AudioEncoder
        from mlx_vlm.models.gemma4.config import AudioConfig

        cfg = AudioConfig.from_dict(ac)
        audio = AudioEncoder(cfg)
        ch = tuple(cfg.subsampling_conv_channels)
        self._strict_load(audio, "audio_tower.", lambda w: _sanitize_audio(w, ch))
        emb = self._embedder_cls(
            ac.get("output_proj_dims") or ac["hidden_size"],
            self.tc["hidden_size"],
            ac["rms_norm_eps"],
        )
        self._strict_load(emb, "embed_audio.")
        self.audio, self.embed_audio = audio, emb

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.dir)
        return self._processor

    # -- encoding -----------------------------------------------------------------------
    def _prepare(self, item: dict, prefix: str):
        """-> (ids, {"image": feats[n,512], ...}) for one item; features in placeholder order."""
        import mlx.core as mx

        text, media = build_text(item, prefix)
        kw: dict[str, Any] = {"text": [text], "return_tensors": "np"}
        if "image" in media:
            kw["images"] = [[_load_image(x) for x in media["image"]]]
        if "video" in media:
            kw["videos"] = [[_load_video(x) for x in media["video"]]]
        if "audio" in media:
            kw["audio"] = [_load_audio(x) for x in media["audio"]]
        out = self.processor(**kw)
        ids = out["input_ids"][0]
        feats: dict[str, Any] = {}
        if "image" in media:
            self._ensure_vision()
            f = self.vision(
                mx.array(out["pixel_values"]), mx.array(out["image_position_ids"])
            )
            feats["image"] = self.embed_vision(f)[0]
        if "video" in media:
            self._ensure_vision()
            f = self.vision(
                mx.array(out["pixel_values_videos"]),
                mx.array(out["video_position_ids"]),
            )
            feats["video"] = self.embed_vision(f)[0]
        if "audio" in media:
            self._ensure_audio()
            pad = mx.array(~out["input_features_mask"].astype(bool))  # True = padding
            enc, bad = self.audio(mx.array(out["input_features"]), pad)
            keep = mx.array(np.nonzero(~np.array(bad).reshape(-1))[0])
            feats["audio"] = self.embed_audio(enc).reshape(-1, self.tc["hidden_size"])[
                keep
            ]
        return ids, feats

    def embed_items(
        self,
        items: list[str | dict],
        task: str | None = None,
        instruction: str | None = None,
        dims: int | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """Unit vectors [N, dims or 768] and the token count of every item."""
        import mlx.core as mx

        prefix = resolve_prompt(self.prompts, task, instruction)
        prepared = [
            self._prepare({"text": it} if isinstance(it, str) else it, prefix)
            for it in items
        ]
        lengths = [len(p[0]) for p in prepared]
        for i, n in enumerate(lengths):
            if n > MAX_TOKENS:
                raise ValueError(
                    f"input {i} is {n} tokens; the context is {MAX_TOKENS}"
                )
        out = np.zeros((len(items), self.tc["embedding_dim"]), dtype=np.float32)
        for group in plan_batches(lengths):
            width = max(lengths[g] for g in group)
            ids = np.zeros((len(group), width), dtype=np.int64)
            valid = np.zeros((len(group), width), dtype=bool)
            for r, g in enumerate(group):
                ids[r, : lengths[g]] = prepared[g][0]
                valid[r, : lengths[g]] = True
            embeds = self._merge(ids, [prepared[g][1] for g in group])
            vec = self.text(embeds, mx.array(valid))
            mx.eval(vec)
            out[group] = np.array(vec)
        return truncate_normalize(out, dims).astype(np.float32), lengths

    def _merge(self, ids: np.ndarray, feats: list[dict]):
        """Token embeddings with every placeholder row replaced by its soft-token feature."""
        import mlx.core as mx

        tokens = {
            "image": self.config["image_token_id"],
            "video": self.config["video_token_id"],
            "audio": self.config["audio_token_id"],
        }
        soft = np.zeros(ids.shape, dtype=bool)
        for t in tokens.values():
            soft |= ids == t
        embeds = self.text.embed(mx.array(np.where(soft, self.tc["pad_token_id"], ids)))
        if not soft.any():
            return embeds
        rows = []
        for r in range(ids.shape[0]):
            row = embeds[r]
            for kind, f in feats[r].items():
                pos = np.nonzero(ids[r] == tokens[kind])[0]
                if len(pos) != f.shape[0]:
                    raise ValueError(
                        f"{kind}: {f.shape[0]} features for {len(pos)} placeholder tokens"
                    )
                idx = mx.array(pos)
                row[idx] = f.astype(row.dtype)
            rows.append(row)
        return mx.stack(rows)


# ── media loading (path / URL / data URI / bytes) ────────────────────────────────────────


def _read_bytes(x) -> bytes:
    import base64

    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if not isinstance(x, str):
        raise ValueError(f"unsupported media value of type {type(x).__name__}")
    if x.startswith("data:"):
        return base64.b64decode(x.split(",", 1)[1])
    if x.startswith(("http://", "https://")):
        import urllib.request

        try:
            with urllib.request.urlopen(x, timeout=30) as r:  # noqa: S310
                return bytes(r.read())
        except OSError as e:
            raise ValueError(f"cannot fetch {x[:60]!r}: {e}") from None
    try:
        with open(x[7:] if x.startswith("file://") else x, "rb") as f:
            return f.read()
    except OSError as e:
        raise ValueError(f"cannot read media {x[:60]!r}: {e.strerror or e}") from None


def _load_image(x):
    import io

    from PIL import Image

    if hasattr(x, "convert"):
        return x.convert("RGB")
    try:
        return Image.open(io.BytesIO(_read_bytes(x))).convert("RGB")
    except OSError as e:  # PIL.UnidentifiedImageError is an OSError
        raise ValueError(f"cannot decode image: {e}") from None


def _load_video(x):
    """A video is a list of frames (image values); decoding a video file is the caller's job."""
    if isinstance(x, (list, tuple)):
        return [_load_image(f) for f in x]
    raise ValueError(
        "video input must be a list of frames (image paths / data URIs / URLs), sampled by the caller"
    )


def _load_audio(x) -> np.ndarray:
    """16 kHz mono float32 from PCM16 WAV bytes / path / data URI, or an already-decoded array."""
    import io
    import wave

    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    raw = _read_bytes(x)
    try:
        with wave.open(io.BytesIO(raw)) as w:
            rate, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            data = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise ValueError(f"audio must be a PCM WAV file ({e})") from None
    if sw != 2:
        raise ValueError(f"audio must be 16-bit PCM WAV, got {sw * 8}-bit")
    a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if rate != 16000:
        n = round(len(a) * 16000 / rate)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(
            np.float32
        )
    return a
