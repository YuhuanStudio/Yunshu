"""Audio input to a VLM without an audio encoder is refused (2026-11-09).

Before: Qwen3.5 (no audio tower) accepted audio_paths and silently dropped the audio.
"""

import json
from pathlib import Path

import pytest

from yunshu_engine.vlm_engine import VLMEngine, _config_has_audio

MODELS = Path("/Volumes/P5Plus/models")


def test_config_markers():
    assert not _config_has_audio({"vision_config": {}, "text_config": {}})
    assert _config_has_audio({"audio_config": {}})
    assert _config_has_audio({"thinker_config": {"audio_config": {}}})
    assert _config_has_audio({"sound_config": {}})  # Nemotron Omni naming


@pytest.mark.parametrize(
    "name, audio",
    [
        ("Jundot/Qwen3.8-27B-oQ4e-mtp", False),
        ("Qwen3-Omni-30B-A3B-Instruct-4bit", True),
        ("Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit", True),
        ("gemma-4-e2b-it-4bit", True),
    ],
)
def test_real_checkpoint_configs(name, audio):
    cfg = MODELS / name / "config.json"
    if not cfg.exists():
        pytest.skip("checkpoint not on this machine")
    assert _config_has_audio(json.loads(cfg.read_text())) is audio


def test_audio_refused_without_an_audio_encoder():
    eng = VLMEngine.__new__(VLMEngine)
    eng._batch_runner = object()
    eng._has_vision = True
    eng._has_audio = False
    eng._processor = object()
    eng._model_path = "qwen35"
    with pytest.raises(ValueError, match="no audio encoder"):
        eng._check_request_supported([], ["a.wav"], {})
    eng._check_request_supported(["a.png"], [], {})  # images still fine
