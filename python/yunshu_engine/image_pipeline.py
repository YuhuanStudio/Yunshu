from __future__ import annotations

"""Image generation pipeline registry — pluggable diffusion architecture support.

Provides a registry that maps model paths/names to the correct pipeline
implementation. Currently supports:
- Z-Image (Turbo/Dev) — ZImageTransformer + Qwen3 text encoder + VAE
- Flux (Dev/Schnell) — placeholder for future Flux pipeline support
- Flux2 (Klein) — placeholder for future Flux2 pipeline support
- Qwen-Image — placeholder for future Qwen-Image pipeline support

Architecture:
  DiffusionPipeline: Abstract base for all image generation pipelines
  PipelineRegistry: Maps model paths → pipeline classes
  AutoPipeline: Auto-detects pipeline type from model directory structure

Integration:
  - ImageGenEngine delegates to the appropriate pipeline via registry
  - Model manager detects pipeline type during model loading
  - Gateway endpoints remain the same regardless of pipeline type
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineType(StrEnum):
    """Supported diffusion pipeline architectures."""
    Z_IMAGE = "z_image"
    FLUX = "flux"
    FLUX2 = "flux2"
    QWEN_IMAGE = "qwen_image"
    UNKNOWN = "unknown"


@dataclass
class PipelineInfo:
    """Metadata about a registered pipeline."""
    pipeline_type: PipelineType
    name: str
    description: str
    supported_features: list[str] = field(default_factory=list)
    default_steps: int = 4
    default_size: tuple[int, int] = (1024, 1024)
    latent_channels: int = 16
    vae_scale: int = 8


# ── Pipeline registry ──


_PIPELINE_REGISTRY: dict[str, PipelineInfo] = {}


def register_pipeline(name: str, info: PipelineInfo) -> None:
    _PIPELINE_REGISTRY[name.lower()] = info


def get_pipeline_info(name: str) -> PipelineInfo | None:
    return _PIPELINE_REGISTRY.get(name.lower())


def list_pipelines() -> dict[str, PipelineInfo]:
    return dict(_PIPELINE_REGISTRY)


# Register known pipelines
register_pipeline("z-image-turbo-mlx-4bit", PipelineInfo(
    pipeline_type=PipelineType.Z_IMAGE,
    name="Z-Image-Turbo-MLX-4bit",
    description="Z-Image Turbo 4-bit quantized, MLX native",
    supported_features=["text2img", "img2img", "inpaint", "controlnet", "depth", "lora", "teacache", "tiling"],
    default_steps=4,
    latent_channels=16,
    vae_scale=8,
))

register_pipeline("z-image-turbo", PipelineInfo(
    pipeline_type=PipelineType.Z_IMAGE,
    name="Z-Image-Turbo",
    description="Z-Image Turbo, full precision",
    supported_features=["text2img", "img2img", "inpaint", "controlnet", "depth", "lora", "teacache", "tiling"],
    default_steps=4,
    latent_channels=16,
    vae_scale=8,
))

register_pipeline("z-image-dev", PipelineInfo(
    pipeline_type=PipelineType.Z_IMAGE,
    name="Z-Image-Dev",
    description="Z-Image Dev, full precision",
    supported_features=["text2img", "img2img", "inpaint", "controlnet", "depth", "lora", "teacache", "tiling"],
    default_steps=28,
    latent_channels=16,
    vae_scale=8,
))

register_pipeline("flux-dev", PipelineInfo(
    pipeline_type=PipelineType.FLUX,
    name="FLUX.1-dev",
    description="FLUX.1 Dev, dual-stream DiT (placeholder)",
    supported_features=["text2img", "controlnet", "depth", "lora"],
    default_steps=28,
    latent_channels=16,
    vae_scale=8,
))

register_pipeline("flux-schnell", PipelineInfo(
    pipeline_type=PipelineType.FLUX,
    name="FLUX.1-schnell",
    description="FLUX.1 Schnell, dual-stream DiT (placeholder)",
    supported_features=["text2img", "lora"],
    default_steps=4,
    latent_channels=16,
    vae_scale=8,
))

register_pipeline("flux2-klein", PipelineInfo(
    pipeline_type=PipelineType.FLUX2,
    name="Flux2Klein",
    description="Flux2 Klein, next-gen dual-stream DiT (placeholder)",
    supported_features=["text2img", "inpaint", "controlnet", "depth", "lora"],
    default_steps=28,
    latent_channels=16,
    vae_scale=8,
))

register_pipeline("qwen-image", PipelineInfo(
    pipeline_type=PipelineType.QWEN_IMAGE,
    name="Qwen-Image",
    description="Qwen-Image, Qwen-based DiT (placeholder)",
    supported_features=["text2img", "inpaint", "lora"],
    default_steps=28,
    latent_channels=16,
    vae_scale=8,
))


# ── Auto-detection ──


def detect_pipeline_type(model_path: str) -> PipelineType:
    """Auto-detect pipeline type from model directory structure.

    Detection strategy:
    1. Check for known model name patterns
    2. Check directory structure (transformer/, vae/, text_encoder/)
    3. Check transformer config.json for architecture hints
    """
    path_lower = model_path.lower()
    path = Path(model_path)

    # Name-based detection
    if "flux2" in path_lower or "klein" in path_lower:
        return PipelineType.FLUX2
    if "flux" in path_lower:
        return PipelineType.FLUX
    if "qwen-image" in path_lower or "qwen_image" in path_lower:
        return PipelineType.QWEN_IMAGE
    if "z-image" in path_lower or "zimage" in path_lower:
        return PipelineType.Z_IMAGE

    # Directory-based detection
    if (path / "transformer" / "config.json").exists():
        try:
            import json
            config = json.loads((path / "transformer" / "config.json").read_text())
            arch = config.get("_class_name", "").lower()
            model_type = config.get("model_type", "").lower()

            if "flux2" in arch or "flux2" in model_type:
                return PipelineType.FLUX2
            if "flux" in arch or "flux" in model_type:
                return PipelineType.FLUX
            if "qwen" in arch or "qwen" in model_type:
                return PipelineType.QWEN_IMAGE
        except Exception:
            logger.debug("transformer config.json read failed", exc_info=True)

    # Default to Z-Image if it has the right directory structure
    if all((path / d).is_dir() for d in ("transformer", "vae", "text_encoder")):
        return PipelineType.Z_IMAGE

    return PipelineType.UNKNOWN


def get_pipeline_info_for_path(model_path: str) -> PipelineInfo:
    """Get pipeline info for a model path, creating a default if not registered."""
    pipeline_type = detect_pipeline_type(model_path)

    # Try to find a registered pipeline matching the path
    path_name = Path(model_path).name.lower()
    info = get_pipeline_info(path_name)
    if info is not None:
        return info

    # Try matching by pipeline type
    for _name, info in _PIPELINE_REGISTRY.items():
        if info.pipeline_type == pipeline_type:
            return info

    # Default
    return PipelineInfo(
        pipeline_type=pipeline_type,
        name=Path(model_path).name,
        description=f"Auto-detected {pipeline_type.value} pipeline",
        supported_features=["text2img"],
        default_steps=4,
        latent_channels=16,
        vae_scale=8,
    )


def create_pipeline_for_path(model_path: str, config=None):
    """Create the appropriate pipeline instance for a model path.

    Currently returns ImageGenEngine for all types (Z-Image only fully supported).
    Other types will be supported as their implementations are completed.
    """
    pipeline_type = detect_pipeline_type(model_path)

    if pipeline_type == PipelineType.Z_IMAGE:
        from .image_engine import ImageGenEngine
        return ImageGenEngine(model_path, config)

    if pipeline_type in (PipelineType.FLUX, PipelineType.FLUX2):
        logger.warning(f"Pipeline type {pipeline_type.value} not yet natively supported, "
                       f"falling back to Z-Image engine. Check model compatibility.")
        from .image_engine import ImageGenEngine
        return ImageGenEngine(model_path, config)

    if pipeline_type == PipelineType.QWEN_IMAGE:
        logger.warning(f"Pipeline type {pipeline_type.value} not yet natively supported, "
                       f"falling back to Z-Image engine.")
        from .image_engine import ImageGenEngine
        return ImageGenEngine(model_path, config)

    # Unknown type — try Z-Image engine as best effort
    logger.warning(f"Unknown pipeline type for {model_path}, attempting Z-Image engine")
    from .image_engine import ImageGenEngine
    return ImageGenEngine(model_path, config)
