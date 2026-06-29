"""Unit tests for model discovery."""

import json
from pathlib import Path

import pytest

from yunshu_engine.model_discovery import (
    detect_model_type,
    discover_models,
    discover_models_from_dirs,
    estimate_model_size,
)


def _make_model_dir(
    base: Path, name: str, config: dict, files: dict | None = None
) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "config.json", "w") as f:
        json.dump(config, f)
    if files:
        for fname, content in files.items():
            if isinstance(content, bytes):
                (d / fname).write_bytes(content)
            else:
                (d / fname).write_text(content)
    return d


class TestDetectModelType:
    def test_llm_default(self, tmp_path):
        _make_model_dir(tmp_path, "qwen", {"model_type": "qwen2"})
        assert detect_model_type(tmp_path / "qwen") == "llm"

    def test_vlm_from_architecture(self, tmp_path):
        _make_model_dir(
            tmp_path,
            "qwen-vl",
            {
                "architectures": ["Qwen2VLForConditionalGeneration"],
            },
        )
        assert detect_model_type(tmp_path / "qwen-vl") == "vlm"

    def test_vlm_model_type_is_vlm(self, tmp_path):
        # qwen2_vl is a known VLM model type in mlx-lm's registry
        _make_model_dir(tmp_path, "vlm-model", {"model_type": "qwen2_vl"})
        assert detect_model_type(tmp_path / "vlm-model") == "vlm"

    def test_vlm_from_vision_config(self, tmp_path):
        _make_model_dir(
            tmp_path,
            "model",
            {
                "model_type": "qwen2_vl",
                "vision_config": {"hidden_size": 1024},
            },
        )
        assert detect_model_type(tmp_path / "model") == "vlm"

    def test_tts_from_architecture(self, tmp_path):
        _make_model_dir(
            tmp_path,
            "tts",
            {
                "architectures": ["Qwen3TTSForConditionalGeneration"],
            },
        )
        assert detect_model_type(tmp_path / "tts") == "audio_tts"

    def test_tts_from_model_type(self, tmp_path):
        _make_model_dir(tmp_path, "tts2", {"model_type": "kokoro"})
        assert detect_model_type(tmp_path / "tts2") == "audio_tts"

    def test_asr_from_architecture(self, tmp_path):
        _make_model_dir(
            tmp_path,
            "whisper",
            {
                "architectures": ["WhisperForConditionalGeneration"],
            },
        )
        assert detect_model_type(tmp_path / "whisper") == "audio_stt"

    def test_image_gen_from_model_type(self, tmp_path):
        _make_model_dir(tmp_path, "flux", {"model_type": "flux"})
        assert detect_model_type(tmp_path / "flux") == "image_gen"

    def test_image_gen_from_architecture(self, tmp_path):
        _make_model_dir(
            tmp_path,
            "sd3",
            {
                "architectures": ["SD3Transformer2DModel"],
            },
        )
        assert detect_model_type(tmp_path / "sd3") == "image_gen"

    def test_no_config_falls_back_to_name(self, tmp_path):
        d = tmp_path / "my-tts-model"
        d.mkdir()
        assert detect_model_type(d) == "audio_tts"

    def test_no_config_llm_default(self, tmp_path):
        d = tmp_path / "random-model"
        d.mkdir()
        assert detect_model_type(d) == "llm"

    def test_name_heuristic_vlm(self, tmp_path):
        d = tmp_path / "something-vision"
        d.mkdir()
        assert detect_model_type(d) == "vlm"

    def test_name_heuristic_asr(self, tmp_path):
        d = tmp_path / "whisper-large"
        d.mkdir()
        assert detect_model_type(d) == "audio_stt"

    def test_name_heuristic_image(self, tmp_path):
        d = tmp_path / "z-image-1"
        d.mkdir()
        assert detect_model_type(d) == "image_gen"


class TestEstimateModelSize:
    def test_no_files(self, tmp_path):
        d = tmp_path / "empty-model"
        d.mkdir()
        assert estimate_model_size(d) == 0

    def test_safetensors(self, tmp_path):
        d = _make_model_dir(
            tmp_path, "model", {}, {"model.safetensors": b"\x00" * 1000}
        )
        size = estimate_model_size(d)
        assert size >= 1000

    def test_includes_overhead(self, tmp_path):
        d = _make_model_dir(
            tmp_path, "model", {}, {"model.safetensors": b"\x00" * 1000}
        )
        size = estimate_model_size(d)
        assert size > 1000  # 5% overhead


class TestDiscoverModels:
    def test_empty_dir(self, tmp_path):
        models = discover_models(tmp_path)
        assert models == {}

    def test_single_model(self, tmp_path):
        _make_model_dir(tmp_path, "qwen-7b", {"model_type": "qwen2"})
        models = discover_models(tmp_path)
        assert "qwen-7b" in models
        assert models["qwen-7b"].model_type == "llm"
        assert models["qwen-7b"].engine_type == "batched"

    def test_multiple_models(self, tmp_path):
        _make_model_dir(tmp_path, "llama", {"model_type": "llama"})
        _make_model_dir(
            tmp_path, "qwen-vl", {"model_type": "qwen2_vl", "vision_config": {}}
        )
        models = discover_models(tmp_path)
        assert len(models) == 2
        assert models["llama"].model_type == "llm"
        assert models["qwen-vl"].model_type == "vlm"

    def test_nested_directories(self, tmp_path):
        org = tmp_path / "mlx-community"
        org.mkdir()
        _make_model_dir(org, "Qwen2.5-7B", {"model_type": "qwen2"})
        models = discover_models(tmp_path)
        assert "Qwen2.5-7B" in models

    def test_skips_hidden_dirs(self, tmp_path):
        _make_model_dir(tmp_path, ".hidden", {"model_type": "qwen2"})
        models = discover_models(tmp_path)
        assert models == {}

    def test_nonexistent_dir_raises(self):
        with pytest.raises(ValueError):
            discover_models(Path("/nonexistent/path"))


class TestDiscoverModelsFromDirs:
    def test_multiple_dirs(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        _make_model_dir(dir1, "model-a", {"model_type": "qwen2"})
        _make_model_dir(dir2, "model-b", {"model_type": "llama"})
        models = discover_models_from_dirs([dir1, dir2])
        assert "model-a" in models
        assert "model-b" in models

    def test_first_wins_on_conflict(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        _make_model_dir(dir1, "model", {"model_type": "qwen2"})
        _make_model_dir(dir2, "model", {"model_type": "llama"})
        models = discover_models_from_dirs([dir1, dir2])
        assert models["model"].config_model_type == "qwen2"

    def test_nonexistent_dirs_skipped(self, tmp_path):
        models = discover_models_from_dirs([tmp_path / "nope"])
        assert models == {}
