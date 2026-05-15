from __future__ import annotations
"""Model-specific preprocessors for multi-modal input handling (vllm-omni pattern).

vllm-omni has 17+ model-specific input processors for different architectures.
Yunshu implements a registry of preprocessors that handle:
  - Audio token encoding (Qwen3-Omni, CosyVoice)
  - Image preprocessing (LLaVA, Qwen-VL, custom)
  - Video frame extraction (Wan2.2, LTX2)
  - OCR document layout (GLM-OCR)
  - Speech feature extraction (Whisper, Parakeet)

Each preprocessor handles architecture-specific input format requirements.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class PreprocessorType(Enum):
    AUDIO = auto()
    IMAGE = auto()
    VIDEO = auto()
    OCR = auto()
    SPEECH = auto()
    MULTIMODAL = auto()


@dataclass
class PreprocessedInput:
    """Result of preprocessing a raw input for a specific model."""
    input_type: PreprocessorType
    model_family: str
    # Token representations
    token_ids: list[int] = field(default_factory=list)
    # Embedding tensors (for models that accept pre-computed embeddings)
    embeddings: Any = None  # mx.array or None
    # Attention mask modifications
    attention_mask: Any = None
    # Position ID offsets (for mRoPE or variable-length inputs)
    position_offsets: list[int] = field(default_factory=list)
    # Metadata
    original_tokens: int = 0
    processed_tokens: int = 0
    # Features extracted from input
    features: dict = field(default_factory=dict)
    # Errors/warnings from preprocessing
    warnings: list[str] = field(default_factory=list)


class ModelPreprocessor(ABC):
    """Base class for model-specific input preprocessors."""

    @property
    @abstractmethod
    def model_family(self) -> str:
        """The model family this preprocessor handles."""

    @property
    @abstractmethod
    def input_type(self) -> PreprocessorType:
        """The type of input this preprocessor handles."""

    @abstractmethod
    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        """Preprocess raw input for the model."""

    def detect_model(self, model_config: dict) -> bool:
        """Detect if the given model config matches this preprocessor."""
        return False


class QwenOmniAudioPreprocessor(ModelPreprocessor):
    """Handles Qwen3-Omni audio token encoding.

    Qwen3-Omni uses special <audio_start>/<audio_end> tokens and
    codec-based audio tokenization. This preprocessor:
    1. Detects audio content in messages
    2. Encodes audio to token IDs using the model's codec
    3. Wraps tokens with start/end markers
    """

    model_family = "qwen3_omni"
    input_type = PreprocessorType.AUDIO

    # Qwen3-Omni special tokens
    AUDIO_START = "<|audio_bos|>"
    AUDIO_END = "<|audio_eos|>"

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        audio_tokens = []
        warnings = []

        if isinstance(raw_input, list) and raw_input and isinstance(raw_input[0], int):
            # Pre-tokenized input — pass through directly
            audio_tokens = raw_input
        elif isinstance(raw_input, dict):
            audio_tokens = raw_input.get("token_ids", [])
        else:
            # Raw audio bytes — real codec not yet implemented
            # Return empty token_ids to prevent fake tokens replacing real prompt
            warnings.append("Qwen3-Omni audio codec not yet integrated — pass pre-tokenized input")
            estimated_tokens = 0
            if isinstance(raw_input, (bytes, bytearray)):
                estimated_tokens = len(raw_input) // 320
            return PreprocessedInput(
                input_type=PreprocessorType.AUDIO,
                model_family=self.model_family,
                token_ids=[],  # SAFE: no fake tokens
                original_tokens=estimated_tokens,
                processed_tokens=0,
                features={"codec": "qwen3_omni", "estimated_tokens": estimated_tokens},
                warnings=warnings,
            )

        return PreprocessedInput(
            input_type=PreprocessorType.AUDIO,
            model_family=self.model_family,
            token_ids=audio_tokens,
            original_tokens=len(audio_tokens),
            processed_tokens=len(audio_tokens),
            features={"codec": "qwen3_omni"},
            warnings=warnings,
        )

    def detect_model(self, model_config: dict) -> bool:
        return any(
            x in model_config.get("model_type", "").lower()
            for x in ("qwen3_omni", "qwen2_5_omni")
        )


class CosyVoicePhonemePreprocessor(ModelPreprocessor):
    """Handles CosyVoice phoneme-based TTS preprocessing.

    CosyVoice uses phoneme encoding with duration prediction.
    This preprocessor converts text to phoneme sequences.
    """

    model_family = "cosyvoice"
    input_type = PreprocessorType.SPEECH

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        text = ""
        if isinstance(raw_input, str):
            text = raw_input
        elif isinstance(raw_input, dict):
            text = raw_input.get("text", "")

        estimated_phonemes = len(text) * 3

        return PreprocessedInput(
            input_type=PreprocessorType.SPEECH,
            model_family=self.model_family,
            token_ids=[],  # Placeholder: phoneme encoder not yet integrated
            original_tokens=len(text.split()),
            processed_tokens=estimated_phonemes,
            features={"phoneme_count": estimated_phonemes},
            warnings=["CosyVoice phoneme encoder not yet integrated"],
        )

    def detect_model(self, model_config: dict) -> bool:
        return "cosyvoice" in model_config.get("model_type", "").lower()


class LLaVAImagePreprocessor(ModelPreprocessor):
    """Handles LLaVA-family image preprocessing.

    LLaVA uses CLIP vision encoder + projector. Images are:
    1. Resized to square (336x336 or 384x384)
    2. Encoded through vision encoder
    3. Projected to language model dimension via MLP projector
    """

    model_family = "llava"
    input_type = PreprocessorType.IMAGE

    def __init__(self, image_size: int = 336, patch_size: int = 14) -> None:
        self._image_size = image_size
        self._patch_size = patch_size

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        num_patches = (self._image_size // self._patch_size) ** 2

        return PreprocessedInput(
            input_type=PreprocessorType.IMAGE,
            model_family=self.model_family,
            token_ids=[],  # Placeholder: vision encoder not yet integrated
            original_tokens=1,
            processed_tokens=num_patches,
            features={
                "image_size": self._image_size,
                "patch_size": self._patch_size,
                "num_patches": num_patches,
            },
            warnings=["LLaVA vision encoder not yet integrated — VLM engine uses mlx_vlm directly"],
        )

    def detect_model(self, model_config: dict) -> bool:
        model_type = model_config.get("model_type", "").lower()
        return model_type in ("llava", "llava_next", "llava_onevision")


class QwenVLImagePreprocessor(ModelPreprocessor):
    """Handles Qwen-VL family image preprocessing.

    Qwen-VL uses ViT + abstractor. Supports dynamic resolution
    with vit-house-keeping tokens.
    """

    model_family = "qwen_vl"
    input_type = PreprocessorType.IMAGE

    def __init__(self, min_pixels: int = 56*56, max_pixels: int = 28*28*4*1280) -> None:
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        resolution = kwargs.get("resolution", 448)
        patch_size = 14
        num_patches = (resolution // patch_size) ** 2

        return PreprocessedInput(
            input_type=PreprocessorType.IMAGE,
            model_family=self.model_family,
            token_ids=[],  # Placeholder: ViT encoder not yet integrated
            original_tokens=1,
            processed_tokens=num_patches,
            features={
                "resolution": resolution,
                "num_patches": num_patches,
                "min_pixels": self._min_pixels,
                "max_pixels": self._max_pixels,
            },
            warnings=["Qwen-VL ViT encoder not yet integrated — VLM engine uses mlx_vlm directly"],
        )

    def detect_model(self, model_config: dict) -> bool:
        model_type = model_config.get("model_type", "").lower()
        return any(
            x in model_type
            for x in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl")
        )


class WanVideoPreprocessor(ModelPreprocessor):
    """Handles Wan2.2 video frame preprocessing.

    Wan2.2 expects video frames as temporal tensors with specific
    normalization and channel ordering.
    """

    model_family = "wan_video"
    input_type = PreprocessorType.VIDEO

    def __init__(self, num_frames: int = 16, frame_size: int = 512) -> None:
        self._num_frames = num_frames
        self._frame_size = frame_size

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        num_frames = kwargs.get("num_frames", self._num_frames)
        frame_size = kwargs.get("frame_size", self._frame_size)

        tokens_per_frame = (frame_size // 16) ** 2
        total_tokens = num_frames * tokens_per_frame

        return PreprocessedInput(
            input_type=PreprocessorType.VIDEO,
            model_family=self.model_family,
            token_ids=[],  # Placeholder: video encoder not yet integrated
            original_tokens=num_frames,
            processed_tokens=total_tokens,
            features={
                "num_frames": num_frames,
                "frame_size": frame_size,
                "tokens_per_frame": tokens_per_frame,
                "temporal_encoding": "3d_conv",
            },
            warnings=["Wan video encoder not yet integrated"],
        )

    def detect_model(self, model_config: dict) -> bool:
        return "wan" in model_config.get("model_type", "").lower()


class GLMOCRPreprocessor(ModelPreprocessor):
    """Handles GLM-OCR document layout preprocessing.

    GLM-OCR processes document images with specific layout-aware
    token encoding for OCR tasks.
    """

    model_family = "glm_ocr"
    input_type = PreprocessorType.OCR

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        max_patches = kwargs.get("max_patches", 1024)
        num_patches = min(max_patches, 1024)

        return PreprocessedInput(
            input_type=PreprocessorType.OCR,
            model_family=self.model_family,
            token_ids=[],  # Placeholder: OCR layout encoder not yet integrated
            original_tokens=1,
            processed_tokens=num_patches,
            features={
                "max_patches": max_patches,
                "layout_aware": True,
            },
            warnings=["GLM-OCR layout encoder not yet integrated"],
        )

    def detect_model(self, model_config: dict) -> bool:
        return "glm" in model_config.get("model_type", "").lower() and "ocr" in model_config.get("model_type", "").lower()


class DeepSeekOCRPreprocessor(ModelPreprocessor):
    """Handles DeepSeek-OCR document preprocessing."""

    model_family = "deepseek_ocr"
    input_type = PreprocessorType.OCR

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        max_patches = kwargs.get("max_patches", 2048)
        return PreprocessedInput(
            input_type=PreprocessorType.OCR,
            model_family=self.model_family,
            token_ids=[],  # Placeholder: DeepSeek OCR encoder not yet integrated
            original_tokens=1,
            processed_tokens=max_patches,
            features={"max_patches": max_patches},
            warnings=["DeepSeek OCR encoder not yet integrated"],
        )

    def detect_model(self, model_config: dict) -> bool:
        return "deepseek" in model_config.get("model_type", "").lower() and "ocr" in model_config.get("model_type", "").lower()


class WhisperSpeechPreprocessor(ModelPreprocessor):
    """Handles Whisper-family speech feature extraction.

    Converts raw audio to mel spectrogram features for ASR models.
    """

    model_family = "whisper"
    input_type = PreprocessorType.SPEECH

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        max_audio_seconds: float = 30.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._n_mels = n_mels
        self._max_seconds = max_audio_seconds

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        audio_length = 0
        if isinstance(raw_input, (bytes, bytearray)):
            audio_length = len(raw_input) // 2  # 16-bit samples
        elif isinstance(raw_input, dict):
            audio_length = raw_input.get("audio_length", 0)

        num_mel_frames = min(
            audio_length // 160,
            int(self._max_seconds * self._sample_rate / 160),
        )

        return PreprocessedInput(
            input_type=PreprocessorType.SPEECH,
            model_family=self.model_family,
            token_ids=[],  # Placeholder: mel spectrogram not yet computed
            original_tokens=audio_length,
            processed_tokens=num_mel_frames,
            features={
                "sample_rate": self._sample_rate,
                "n_mels": self._n_mels,
                "mel_frames": num_mel_frames,
            },
            warnings=["Whisper mel spectrogram not yet integrated"],
        )

    def detect_model(self, model_config: dict) -> bool:
        return "whisper" in model_config.get("model_type", "").lower()


class PreprocessorRegistry:
    """Registry of model-specific preprocessors (vllm-omni pattern).

    Auto-detects the correct preprocessor from model config and
    routes inputs through the appropriate preprocessing pipeline.
    """

    def __init__(self) -> None:
        self._preprocessors: dict[str, ModelPreprocessor] = {}
        self._type_index: dict[PreprocessorType, list[ModelPreprocessor]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for cls in [
            QwenOmniAudioPreprocessor,
            CosyVoicePhonemePreprocessor,
            LLaVAImagePreprocessor,
            QwenVLImagePreprocessor,
            WanVideoPreprocessor,
            GLMOCRPreprocessor,
            DeepSeekOCRPreprocessor,
            WhisperSpeechPreprocessor,
        ]:
            instance = cls()
            self.register(instance)

    def register(self, preprocessor: ModelPreprocessor) -> None:
        self._preprocessors[preprocessor.model_family] = preprocessor
        self._type_index.setdefault(preprocessor.input_type, []).append(preprocessor)

    def get(self, model_family: str) -> ModelPreprocessor | None:
        return self._preprocessors.get(model_family)

    def detect(self, model_config: dict) -> ModelPreprocessor | None:
        for preprocessor in self._preprocessors.values():
            if preprocessor.detect_model(model_config):
                return preprocessor
        return None

    def preprocess(self, raw_input: Any, model_family: str = "", model_config: dict | None = None, **kwargs) -> PreprocessedInput:
        """Preprocess input for a specific model."""
        preprocessor = None
        if model_family:
            preprocessor = self.get(model_family)
        if preprocessor is None and model_config:
            preprocessor = self.detect(model_config)

        if preprocessor is None:
            return PreprocessedInput(
                input_type=PreprocessorType.MULTIMODAL,
                model_family="unknown",
                warnings=[f"No preprocessor found for family={model_family}"],
            )

        return preprocessor.preprocess(raw_input, **kwargs)

    def list_preprocessors(self) -> list[dict]:
        return [
            {
                "family": p.model_family,
                "type": p.input_type.name,
                "class": type(p).__name__,
            }
            for p in self._preprocessors.values()
        ]

    def get_stats(self) -> dict:
        return {
            "registered": len(self._preprocessors),
            "by_type": {
                t.name: len(ps) for t, ps in self._type_index.items()
            },
        }
