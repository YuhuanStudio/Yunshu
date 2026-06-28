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

    def __init__(self) -> None:
        self._audio_start_id: int | None = None
        self._audio_end_id: int | None = None
        self._audio_token_id: int | None = None

    def configure_from_model_config(self, model_config: dict) -> None:
        thinker = model_config.get("thinker_config", {})
        if isinstance(thinker, dict):
            self._audio_start_id = thinker.get("audio_start_token_id", self._audio_start_id)
            self._audio_end_id = thinker.get("audio_end_token_id", self._audio_end_id)
            self._audio_token_id = thinker.get("audio_token_id", self._audio_token_id)

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
        model_type = model_config.get("model_type", "").lower()
        return any(
            x in model_type
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


class Qwen3TTSPreprocessor(ModelPreprocessor):
    """Handles Qwen3-TTS audio token encoding for text-to-speech.

    Qwen3-TTS uses dedicated TTS tokens: tts_bos_token_id, tts_eos_token_id,
    tts_pad_token_id. Config-driven token IDs from model config.
    """

    model_family = "qwen3_tts"
    input_type = PreprocessorType.SPEECH

    DEFAULT_TTS_BOS_ID = 151672
    DEFAULT_TTS_EOS_ID = 151673
    DEFAULT_TTS_PAD_ID = 151671

    def __init__(self) -> None:
        self._tts_bos_id = self.DEFAULT_TTS_BOS_ID
        self._tts_eos_id = self.DEFAULT_TTS_EOS_ID
        self._tts_pad_id = self.DEFAULT_TTS_PAD_ID

    def configure_from_model_config(self, model_config: dict) -> None:
        self._tts_bos_id = model_config.get("tts_bos_token_id", self._tts_bos_id)
        self._tts_eos_id = model_config.get("tts_eos_token_id", self._tts_eos_id)
        self._tts_pad_id = model_config.get("tts_pad_token_id", self._tts_pad_id)

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        warnings = []
        token_ids: list[int] = []

        if isinstance(raw_input, list) and raw_input and isinstance(raw_input[0], int):
            token_ids = [self._tts_bos_id] + raw_input + [self._tts_eos_id]
        elif isinstance(raw_input, dict):
            if "token_ids" in raw_input:
                token_ids = [self._tts_bos_id] + raw_input["token_ids"] + [self._tts_eos_id]
            else:
                text = raw_input.get("text", "")
                estimated_tokens = len(text) * 3
                warnings.append("Qwen3-TTS codec not yet integrated — pass pre-tokenized input")
                return PreprocessedInput(
                    input_type=PreprocessorType.SPEECH,
                    model_family=self.model_family,
                    token_ids=[],
                    original_tokens=len(text.split()),
                    processed_tokens=estimated_tokens,
                    features={"codec": "qwen3_tts_12hz", "estimated_tokens": estimated_tokens},
                    warnings=warnings,
                )
        elif isinstance(raw_input, str):
            estimated_tokens = len(raw_input) * 3
            warnings.append("Qwen3-TTS codec not yet integrated — pass pre-tokenized input")
            return PreprocessedInput(
                input_type=PreprocessorType.SPEECH,
                model_family=self.model_family,
                token_ids=[],
                original_tokens=len(raw_input.split()),
                processed_tokens=estimated_tokens,
                features={"codec": "qwen3_tts_12hz", "estimated_tokens": estimated_tokens},
                warnings=warnings,
            )

        return PreprocessedInput(
            input_type=PreprocessorType.SPEECH,
            model_family=self.model_family,
            token_ids=token_ids,
            original_tokens=max(0, len(token_ids) - 2),
            processed_tokens=len(token_ids),
            features={
                "codec": "qwen3_tts_12hz",
                "tts_bos_id": self._tts_bos_id,
                "tts_eos_id": self._tts_eos_id,
                "tts_pad_id": self._tts_pad_id,
            },
            warnings=warnings,
        )

    def detect_model(self, model_config: dict) -> bool:
        return model_config.get("model_type", "").lower() == "qwen3_tts"


class Qwen3ASRPreprocessor(ModelPreprocessor):
    """Handles Qwen3-ASR audio token encoding for speech recognition.

    Uses same audio token framework as Qwen3-Omni with different audio_token_id.
    Reads token IDs from thinker_config in model config.
    """

    model_family = "qwen3_asr"
    input_type = PreprocessorType.AUDIO

    DEFAULT_AUDIO_START_ID = 151669
    DEFAULT_AUDIO_END_ID = 151670
    DEFAULT_AUDIO_TOKEN_ID = 151676

    def __init__(self) -> None:
        self._audio_start_id = self.DEFAULT_AUDIO_START_ID
        self._audio_end_id = self.DEFAULT_AUDIO_END_ID
        self._audio_token_id = self.DEFAULT_AUDIO_TOKEN_ID

    def configure_from_model_config(self, model_config: dict) -> None:
        thinker = model_config.get("thinker_config", {})
        if isinstance(thinker, dict):
            self._audio_start_id = thinker.get("audio_start_token_id", self._audio_start_id)
            self._audio_end_id = thinker.get("audio_end_token_id", self._audio_end_id)
            self._audio_token_id = thinker.get("audio_token_id", self._audio_token_id)

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        warnings = []
        token_ids: list[int] = []

        if isinstance(raw_input, list) and raw_input and isinstance(raw_input[0], int):
            token_ids = [self._audio_start_id] + raw_input + [self._audio_end_id]
        elif isinstance(raw_input, dict):
            if "token_ids" in raw_input:
                token_ids = [self._audio_start_id] + raw_input["token_ids"] + [self._audio_end_id]
            else:
                estimated_tokens = raw_input.get("audio_length", 0) // 320
                warnings.append("Qwen3-ASR audio codec not yet integrated — pass pre-tokenized input")
                return PreprocessedInput(
                    input_type=PreprocessorType.AUDIO,
                    model_family=self.model_family,
                    token_ids=[],
                    original_tokens=estimated_tokens,
                    processed_tokens=0,
                    features={"codec": "qwen3_asr", "estimated_tokens": estimated_tokens},
                    warnings=warnings,
                )
        elif isinstance(raw_input, (bytes, bytearray)):
            estimated_tokens = len(raw_input) // 320
            warnings.append("Qwen3-ASR audio codec not yet integrated — pass pre-tokenized input")
            return PreprocessedInput(
                input_type=PreprocessorType.AUDIO,
                model_family=self.model_family,
                token_ids=[],
                original_tokens=estimated_tokens,
                processed_tokens=0,
                features={"codec": "qwen3_asr", "estimated_tokens": estimated_tokens},
                warnings=warnings,
            )

        return PreprocessedInput(
            input_type=PreprocessorType.AUDIO,
            model_family=self.model_family,
            token_ids=token_ids,
            original_tokens=max(0, len(token_ids) - 2),
            processed_tokens=len(token_ids),
            features={
                "codec": "qwen3_asr",
                "audio_start_id": self._audio_start_id,
                "audio_end_id": self._audio_end_id,
                "audio_token_id": self._audio_token_id,
            },
            warnings=warnings,
        )

    def detect_model(self, model_config: dict) -> bool:
        return model_config.get("model_type", "").lower() == "qwen3_asr"


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


class PhiVisionPreprocessor(ModelPreprocessor):
    """Handles Phi-3/4 vision preprocessing.

    Phi-3-Vision uses dynamic image resolution with crop-based encoding.
    """

    model_family = "phi_vision"
    input_type = PreprocessorType.IMAGE

    def __init__(self, image_size: int = 336, num_crops: int = 4) -> None:
        self._image_size = image_size
        self._num_crops = num_crops

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        num_crops = kwargs.get("num_crops", self._num_crops)
        patch_size = 14
        base_patches = (self._image_size // patch_size) ** 2
        total_patches = base_patches * (1 + num_crops)

        return PreprocessedInput(
            input_type=PreprocessorType.IMAGE,
            model_family=self.model_family,
            token_ids=[],  # Placeholder: vision encoder not yet integrated
            original_tokens=1,
            processed_tokens=total_patches,
            features={
                "image_size": self._image_size,
                "num_crops": num_crops,
                "total_patches": total_patches,
            },
            warnings=["Phi vision encoder not yet integrated"],
        )

    def detect_model(self, model_config: dict) -> bool:
        model_type = model_config.get("model_type", "").lower()
        return "phi3_vision" in model_type or "phi4_vision" in model_type


class InternVLImagePreprocessor(ModelPreprocessor):
    """Handles InternVL-family image preprocessing.

    InternVL uses dynamic resolution with pixel shuffle and
    supports interleaved image-text inputs.
    """

    model_family = "internvl"
    input_type = PreprocessorType.IMAGE

    def __init__(self, image_size: int = 448, patch_size: int = 14, downsample_ratio: float = 0.5) -> None:
        self._image_size = image_size
        self._patch_size = patch_size
        self._downsample_ratio = downsample_ratio

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        image_size = kwargs.get("image_size", self._image_size)
        patch_size = self._patch_size
        num_patches = (image_size // patch_size) ** 2
        # InternVL downsamples via pixel shuffle
        effective_patches = int(num_patches * self._downsample_ratio)

        return PreprocessedInput(
            input_type=PreprocessorType.IMAGE,
            model_family=self.model_family,
            token_ids=[],  # Placeholder: vision encoder not yet integrated
            original_tokens=1,
            processed_tokens=effective_patches,
            features={
                "image_size": image_size,
                "num_patches": num_patches,
                "downsample_ratio": self._downsample_ratio,
                "effective_patches": effective_patches,
            },
            warnings=["InternVL vision encoder not yet integrated"],
        )

    def detect_model(self, model_config: dict) -> bool:
        model_type = model_config.get("model_type", "").lower()
        return "internvl" in model_type


class CohereVisionPreprocessor(ModelPreprocessor):
    """Handles Cohere Command-R Vision image preprocessing.

    Cohere2VisionForConditionalGeneration uses ViT + adapter layers.
    """

    model_family = "cohere_vision"
    input_type = PreprocessorType.IMAGE

    def __init__(self, image_size: int = 384, patch_size: int = 14) -> None:
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
            warnings=["Cohere vision encoder not yet integrated"],
        )

    def detect_model(self, model_config: dict) -> bool:
        model_type = model_config.get("model_type", "").lower()
        return "cohere2" in model_type and "vision" in model_type


class LTXVideoPreprocessor(ModelPreprocessor):
    """Handles LTX-Video frame preprocessing.

    LTX-Video uses VAE-based temporal encoding for video generation.
    """

    model_family = "ltx_video"
    input_type = PreprocessorType.VIDEO

    def __init__(self, num_frames: int = 25, frame_size: int = 512) -> None:
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
                "temporal_encoding": "vae",
            },
            warnings=["LTX video encoder not yet integrated"],
        )

    def detect_model(self, model_config: dict) -> bool:
        model_type = model_config.get("model_type", "").lower()
        return "ltx" in model_type and "video" in model_type


class InternVLVideoPreprocessor(ModelPreprocessor):
    """Handles InternVL video frame preprocessing.

    InternVL processes video as temporal image sequences.
    """

    model_family = "internvl_video"
    input_type = PreprocessorType.VIDEO

    def __init__(self, num_frames: int = 8, image_size: int = 448, patch_size: int = 14) -> None:
        self._num_frames = num_frames
        self._image_size = image_size
        self._patch_size = patch_size

    def preprocess(self, raw_input: Any, **kwargs) -> PreprocessedInput:
        num_frames = kwargs.get("num_frames", self._num_frames)
        image_size = kwargs.get("image_size", self._image_size)
        patch_size = self._patch_size

        patches_per_frame = (image_size // patch_size) ** 2
        total_tokens = num_frames * patches_per_frame

        return PreprocessedInput(
            input_type=PreprocessorType.VIDEO,
            model_family=self.model_family,
            token_ids=[],  # Placeholder: video encoder not yet integrated
            original_tokens=num_frames,
            processed_tokens=total_tokens,
            features={
                "num_frames": num_frames,
                "image_size": image_size,
                "patches_per_frame": patches_per_frame,
                "total_tokens": total_tokens,
            },
            warnings=["InternVL video encoder not yet integrated"],
        )

    def detect_model(self, model_config: dict) -> bool:
        model_type = model_config.get("model_type", "").lower()
        return "internvl" in model_type and "video" in model_type


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
            Qwen3TTSPreprocessor,
            Qwen3ASRPreprocessor,
            LLaVAImagePreprocessor,
            QwenVLImagePreprocessor,
            WanVideoPreprocessor,
            GLMOCRPreprocessor,
            DeepSeekOCRPreprocessor,
            WhisperSpeechPreprocessor,
            PhiVisionPreprocessor,
            # More specific preprocessors MUST come before broader ones:
            # InternVLVideoPreprocessor checks "internvl" + "video"
            # and must be tried before InternVLImagePreprocessor which
            # matches any model_type containing "internvl".
            InternVLVideoPreprocessor,
            InternVLImagePreprocessor,
            CohereVisionPreprocessor,
            LTXVideoPreprocessor,
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
