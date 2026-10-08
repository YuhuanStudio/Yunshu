"""EmbeddingGemma 2: the pure parts (prompts, placeholders, Matryoshka, batching, media loading)."""

import base64
import io
import wave

import numpy as np
import pytest

from yunshu_engine import embedding_gemma2 as eg

P = eg.DEFAULT_PROMPTS


def test_prompt_precedence():
    assert eg.resolve_prompt(P, None, None) == ""
    assert eg.resolve_prompt(P, "SearchQuery", None) == "task: search result | query: "
    assert eg.resolve_prompt(P, "SearchQuery", "custom: ") == "custom: "
    with pytest.raises(ValueError, match="unknown task"):
        eg.resolve_prompt(P, "Nope", None)


def test_build_text_key_order_and_prefix_on_text_only():
    s, m = eg.build_text({"text": "a red ball", "image": "x.png"}, "p: ")
    assert s == "p: a red ball<|image|>" and m == {"image": ["x.png"]}
    s, _ = eg.build_text({"image": ["a", "b"], "text": "hi"}, "p: ")
    assert s == "<|image|><|image|>p: hi"
    s, _ = eg.build_text({"audio": "a.wav"}, "p: ")
    assert s == "<|audio|>"


def test_build_text_interleaved_markers_checked():
    s, m = eg.build_text(
        {"text": "x <|image|> y <|video|>", "image": "i", "video": ["f"]}
    )
    assert s == "x <|image|> y <|video|>" and m["video"] == ["f"]
    with pytest.raises(ValueError, match="markers"):
        eg.build_text({"text": "x <|image|> y <|image|>", "image": "i"})
    with pytest.raises(ValueError, match="at least one"):
        eg.build_text({})


def test_matryoshka_truncates_and_renormalises():
    v = np.arange(1, 769, dtype=np.float32)[None]
    v = v / np.linalg.norm(v)
    out = eg.truncate_normalize(v, 256)
    assert out.shape == (1, 256) and abs(np.linalg.norm(out) - 1) < 1e-6
    assert np.allclose(out[0], v[0, :256] / np.linalg.norm(v[0, :256]))
    assert eg.truncate_normalize(v, None).shape == (1, 768)
    with pytest.raises(ValueError, match="dimensions"):
        eg.truncate_normalize(v, 300)


def test_sliding_mask_is_symmetric_and_inclusive():
    m = eg.sliding_mask(6, 2)
    assert m[0, 2] and not m[0, 3] and (m == m.T).all() and m.diagonal().all()


def test_plan_batches_cover_all_and_respect_budget():
    lengths = [5, 3000, 40, 7, 4000, 6]
    groups = eg.plan_batches(lengths, budget=8192)
    assert sorted(i for g in groups for i in g) == list(range(6))
    assert all(
        len(g) * max(lengths[i] for i in g) <= 8192 or len(g) == 1 for g in groups
    )
    assert eg.plan_batches([], 10) == []


def _wav(rate, ch, n):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.ones(n * ch, dtype=np.int16) * 16384).tobytes())
    return b.getvalue()


def test_audio_loading_mono_16k_from_data_uri_and_resample():
    uri = "data:audio/wav;base64," + base64.b64encode(_wav(8000, 2, 800)).decode()
    a = eg._load_audio(uri)
    assert a.dtype == np.float32 and len(a) == 1600 and abs(a[0] - 0.5) < 1e-3
    with pytest.raises(ValueError, match="PCM WAV"):
        eg._load_audio(b"not a wav")


def test_video_needs_frames():
    with pytest.raises(ValueError, match="list of frames"):
        eg._load_video("clip.mp4")


def test_float16_refused_before_any_weights_are_read(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type": "embedding_gemma2", "text_config": {}}'
    )
    with pytest.raises(ValueError, match="float16"):
        eg.EmbeddingGemma2(str(tmp_path), dtype="float16")
    (tmp_path / "config.json").write_text('{"model_type": "gemma3"}')
    with pytest.raises(ValueError, match="not an embedding_gemma2"):
        eg.EmbeddingGemma2(str(tmp_path))


def test_undecodable_media_is_a_value_error_not_an_internal_error(tmp_path):
    """The served check found a 500: PIL's UnidentifiedImageError escaped the 400 mapping."""
    with pytest.raises(ValueError, match="cannot decode image"):
        eg._load_image("data:image/png;base64," + base64.b64encode(b"nope").decode())
    with pytest.raises(ValueError, match="cannot read media"):
        eg._load_image(str(tmp_path / "missing.png"))
    with pytest.raises(ValueError, match="cannot read media"):
        eg._load_audio(str(tmp_path / "missing.wav"))


def test_model_manager_routes_embedding_gemma2(tmp_path):
    import json

    from yunshu_engine import model_manager as mm

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "embedding_gemma2",
                "vision_config": {},
                "architectures": ["EmbeddingGemma2Model"],
            }
        )
    )
    assert mm._detect_model_type(str(tmp_path)) == mm.ModelType.EMBEDDING
    assert mm._embedding_engine_class(str(tmp_path)).__name__ == "GemmaEmbeddingEngine"
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl", "vision_config": {}})
    )
    assert mm._embedding_engine_class(str(tmp_path)).__name__ == "VLEmbeddingEngine"


def test_weight_loader_merges_every_shard_and_memoizes(monkeypatch):
    import sys
    import types

    calls = []
    shards = {
        "model-00001.safetensors": {"language_model.embed_tokens.weight": object()},
        "model-00002.safetensors": {"language_model.layers.0.weight": object()},
    }

    def load(path):
        calls.append(path)
        return shards[path]

    fake_core = types.ModuleType("mlx.core")
    fake_core.load = load
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    model = eg.EmbeddingGemma2.__new__(eg.EmbeddingGemma2)
    model._files = list(shards)
    model._weights = None

    weights = model._all_weights()
    assert weights == {
        key: value for shard in shards.values() for key, value in shard.items()
    }
    assert calls == list(shards)
    assert model._all_weights() is weights
    assert calls == list(shards)
