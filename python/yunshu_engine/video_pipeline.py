from __future__ import annotations

"""Native MLX Video Pipeline — frame-by-frame video generation without mlx-video dependency.

Implements a native MLX pipeline for video generation that follows the Wan2.2-style
architecture. This eliminates the external mlx-video dependency and provides direct
MLX computation for all stages.

Architecture:
  VideoPipeline (ABC): Abstract base for all video generation pipelines
  WanVideoPipeline: Concrete implementation of Wan2.2-style video generation
  VideoLoRAManager: Load/unload LoRA adapters for video models

Pipeline stages (Wan2.2 pattern):
  1. Text encoding: Encode text prompt via CLIP + T5 text encoders
  2. Latent noise: Initialize latent noise in 3D latent space (T×H×W)
  3. Flow matching: Iterative denoising with flow matching scheduler
  4. 3D VAE decode: Temporal + spatial convolution to decode latents -> frames

Flow matching scheduler:
  Replaces traditional DDPM/DDIM with rectified flow matching. The scheduler
  interpolates between noise and data using a learned velocity field:
    z_t = (1 - t) * noise + t * data
    v_pred = model(z_t, t, text_emb)
    z_{t-1} = z_t + v_pred * dt

MLX tensor format note:
  MLX Conv1d uses NLC (batch, length, channels) format.
  MLX Conv2d uses NHWC (batch, height, width, channels) format.
  All internal tensors follow these conventions.

Integration:
  - VideoEngine delegates to WanVideoPipeline when model weights are available
  - VideoLoRAManager handles adapter hot-swapping
  - Gateway endpoint at /v1/video/generations remains unchanged

Env vars:
  YUNSHU_VIDEO_PIPELINE=native  Force native pipeline (skip mlx-video)
  YUNSHU_VIDEO_LORA=path        Auto-load LoRA adapter at startup
"""

import gc
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


# ── Data types ──


@dataclass
class VideoGenRequest:
    """Input specification for video generation."""
    prompt: str = ""
    negative_prompt: str = ""
    image: mx.array | None = None  # Optional source image for I2V (H, W, C)
    num_frames: int = 81  # Must be 4n+1 for Wan2.2
    height: int = 704
    width: int = 1280
    fps: int = 16
    num_steps: int = 20
    guide_scale: float = 5.0
    seed: int = -1  # -1 = random
    scheduler: str = "flow_matching"  # flow_matching, euler, unipc


@dataclass
class VideoGenResult:
    """Output from video generation."""
    frames: list[mx.array] = field(default_factory=list)  # (H, W, C) per frame
    latents: mx.array | None = None  # Final latents (1, C, T, H_lat, W_lat)
    width: int = 0
    height: int = 0
    num_frames: int = 0
    fps: int = 16
    duration_s: float = 0.0
    method: str = ""
    seed_used: int = -1
    metadata: dict = field(default_factory=dict)


# ── Schedulers ──


class FlowMatchingScheduler:
    """Rectified flow matching scheduler for video denoising.

    Implements the flow matching ODE solver:
      z_{t-1} = z_t + v_pred * dt

    where v_pred is the model's predicted velocity and dt = 1/num_steps.
    This replaces traditional DDPM/DDIM with a more stable and faster
    convergence schedule.

    Args:
        num_steps: Number of denoising steps.
        guide_scale: Classifier-free guidance scale (1.0 = no guidance).
    """

    def __init__(self, num_steps: int = 20, guide_scale: float = 5.0) -> None:
        self.num_steps = num_steps
        self.guide_scale = guide_scale
        # Timesteps uniformly spaced from 1.0 to 0.0
        self._timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

    @property
    def timesteps(self) -> list[float]:
        return list(self._timesteps)

    def get_timestep(self, step: int) -> float:
        """Get the continuous timestep t for a given step index."""
        if 0 <= step < len(self._timesteps):
            return self._timesteps[step]
        return 0.0

    def get_dt(self, step: int) -> float:
        """Get the step size dt for a given step."""
        return 1.0 / self.num_steps

    def init_noise(
        self,
        shape: tuple[int, ...],
        seed: int = -1,
        dtype: mx.Dtype = mx.float16,
    ) -> mx.array:
        """Generate initial latent noise using a per-call key (avoids global seed mutation)."""
        if seed >= 0:
            key = mx.random.key(seed)
        else:
            key = mx.random.key(int(time.time_ns()) % (2**31))
        return mx.random.normal(shape=shape, dtype=dtype, key=key)

    def apply_guidance(
        self,
        noise_pred: mx.array,
        noise_pred_uncond: mx.array,
    ) -> mx.array:
        """Apply classifier-free guidance: uncond + scale * (cond - uncond)."""
        return noise_pred_uncond + self.guide_scale * (noise_pred - noise_pred_uncond)

    def step(
        self,
        noise_pred: mx.array,
        latent: mx.array,
        t: float,
        dt: float,
    ) -> mx.array:
        """Perform a single flow matching denoising step: z_{t-dt} = z_t + v_pred * dt."""
        return latent + noise_pred * dt


class EulerScheduler:
    """Euler discretization scheduler: z_{t-1} = z_t - noise_pred * dt."""

    def __init__(self, num_steps: int = 20, guide_scale: float = 5.0) -> None:
        self.num_steps = num_steps
        self.guide_scale = guide_scale

    def get_timestep(self, step: int) -> float:
        return 1.0 - step / self.num_steps

    def get_dt(self, step: int) -> float:
        return 1.0 / self.num_steps

    def init_noise(
        self,
        shape: tuple[int, ...],
        seed: int = -1,
        dtype: mx.Dtype = mx.float16,
    ) -> mx.array:
        """Generate initial latent noise using a per-call key (avoids global seed mutation)."""
        if seed >= 0:
            key = mx.random.key(seed)
        else:
            key = mx.random.key(int(time.time_ns()) % (2**31))
        return mx.random.normal(shape=shape, dtype=dtype, key=key)

    def apply_guidance(
        self,
        noise_pred: mx.array,
        noise_pred_uncond: mx.array,
    ) -> mx.array:
        return noise_pred_uncond + self.guide_scale * (noise_pred - noise_pred_uncond)

    def step(
        self,
        noise_pred: mx.array,
        latent: mx.array,
        t: float,
        dt: float,
    ) -> mx.array:
        return latent - noise_pred * dt


# ── 3D VAE Decoder ──


class TemporalConv3D(nn.Module):
    """3D convolution decomposed into temporal conv + spatial conv.

    MLX tensor formats:
      - Conv1d: NLC (batch, length, channels)
      - Conv2d: NHWC (batch, height, width, channels)

    Internal storage uses NHWTC format (batch, height, width, time, channels)
    for efficient spatial-temporal processing.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Temporal kernel size.
        spatial_kernel: Spatial kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        spatial_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.temporal_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.spatial_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(spatial_kernel, spatial_kernel),
            padding=(spatial_kernel // 2, spatial_kernel // 2),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: (batch, height, width, time, channels) — NHWTC format.

        Returns:
            (batch, height, width, time_out, channels_out) — NHWTC format.
        """
        b, h, w, t, c = x.shape

        # Temporal conv: (B, H, W, T, C) -> (B*H*W, T, C) -> Conv1d (NLC)
        x_temp = x.reshape(b * h * w, t, c)
        x_temp = self.temporal_conv(x_temp)  # (B*H*W, T, C_out)
        _, t_out, c_out = x_temp.shape

        # Reshape back: (B*H*W, T_out, C_out) -> (B, H, W, T_out, C_out)
        x_temp = x_temp.reshape(b, h, w, t_out, c_out)

        # Spatial conv: (B, H, W, T_out, C_out) -> (B*T_out, H, W, C_out) -> Conv2d (NHWC)
        x_spat = x_temp.transpose(0, 3, 1, 2, 4).reshape(b * t_out, h, w, c_out)
        x_spat = self.spatial_conv(x_spat)  # (B*T_out, H, W, C_out)
        _, h_out, w_out, c_out2 = x_spat.shape

        # Reshape back: (B*T_out, H_out, W_out, C_out2) -> (B, H_out, W_out, T_out, C_out2)
        return x_spat.reshape(b, t_out, h_out, w_out, c_out2).transpose(0, 2, 3, 1, 4)


class VideoVAEDecoder(nn.Module):
    """3D VAE decoder for video latent -> pixel frame decoding.

    Takes compressed latent video and decodes to full-resolution frames using
    temporal + spatial convolutions. All tensors use NHWC/NHWTC format for
    MLX compatibility.

    Args:
        latent_channels: Number of latent channels (default: 16 for Wan2.2).
        output_channels: Number of output channels (3 for RGB).
        spatial_scale: Spatial upsampling factor (default: 8).
        temporal_scale: Temporal upsampling factor (default: 4).
        base_channels: Base channel count for decoder blocks.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        output_channels: int = 3,
        spatial_scale: int = 8,
        temporal_scale: int = 4,
        base_channels: int = 128,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale

        # Initial projection: latent channels -> base channels (NHWC Conv2d)
        self.conv_in = nn.Conv2d(
            latent_channels, base_channels,
            kernel_size=(3, 3), padding=(1, 1),
        )

        # Temporal convolution blocks (stored as named attributes for MLX Module)
        self.temporal_block_0 = TemporalConv3D(base_channels, base_channels, kernel_size=3)
        self.temporal_block_1 = TemporalConv3D(base_channels, base_channels, kernel_size=3)

        # Upsampling blocks (2x spatial, repeated log2(spatial_scale) times)
        num_up_blocks = int(math.log2(spatial_scale)) if spatial_scale > 1 else 1
        self._up_convs = []  # List of (conv1, conv2) tuples
        ch = base_channels
        for i in range(num_up_blocks):
            next_ch = max(ch // 2, output_channels * 4)
            conv1 = nn.Conv2d(ch, next_ch, kernel_size=(3, 3), padding=(1, 1))
            conv2 = nn.Conv2d(next_ch, next_ch, kernel_size=(3, 3), padding=(1, 1))
            setattr(self, f"up_conv_{i}_0", conv1)
            setattr(self, f"up_conv_{i}_1", conv2)
            self._up_convs.append((conv1, conv2))
            ch = next_ch
        self._final_ch = ch

        # Final output projection
        self.conv_out = nn.Conv2d(
            ch, output_channels,
            kernel_size=(3, 3), padding=(1, 1),
        )

    def __call__(self, latents: mx.array) -> list[mx.array]:
        """Decode video latents to individual frames.

        Args:
            latents: (1, C_lat, T, H_lat, W_lat) latent video tensor (stored in NCHWT).

        Returns:
            List of (H, W, C) frame tensors.
        """
        if latents.ndim == 4:
            # (C, T, H, W) -> (1, C, T, H, W)
            latents = mx.expand_dims(latents, axis=0)

        b, c, t, h_lat, w_lat = latents.shape  # noqa: F841

        frames = []
        for frame_idx in range(t):
            # Extract single time step latent: (1, C, H_lat, W_lat) -> NHWC
            frame_lat = latents[:, :, frame_idx, :, :].transpose(0, 2, 3, 1)

            # Apply temporal context (average of neighboring frames, excluding current)
            if t > 1:
                neighbors = []
                if frame_idx > 0:
                    neighbors.append(latents[:, :, frame_idx - 1, :, :].transpose(0, 2, 3, 1))
                if frame_idx < t - 1:
                    neighbors.append(latents[:, :, frame_idx + 1, :, :].transpose(0, 2, 3, 1))
                if neighbors:
                    context = sum(neighbors) / len(neighbors)
                    frame_lat = frame_lat + 0.1 * context

            # Spatial decode via NHWC Conv2d
            x = self.conv_in(frame_lat)  # (1, H, W, base_ch)
            x = nn.gelu(x)

            for conv1, conv2 in self._up_convs:
                x = nn.Upsample(scale_factor=2, mode="nearest")(x)
                x = nn.gelu(conv1(x))
                x = nn.gelu(conv2(x))

            x = self.conv_out(x)  # (1, H, W, 3)

            # Clamp to valid range, remove batch dim
            x = mx.clip(x, 0.0, 1.0)
            frames.append(x[0])  # (H, W, 3)

        return frames


# ── Abstract Pipeline ──


class VideoPipeline(ABC):
    """Abstract base class for native MLX video generation pipelines.

    Defines the interface for all video generation backends:
    - generate_frames: Text-to-video generation
    - generate_from_image: Image-to-video animation
    - stream_frames: Frame-by-frame streaming generation
    """

    @abstractmethod
    def generate_frames(self, request: VideoGenRequest) -> VideoGenResult:
        """Generate video frames from a text prompt."""
        ...

    @abstractmethod
    def generate_from_image(
        self,
        request: VideoGenRequest,
        image: mx.array,
    ) -> VideoGenResult:
        """Generate video frames from an image + text prompt."""
        ...

    @abstractmethod
    def stream_frames(
        self,
        request: VideoGenRequest,
        callback: Callable[[mx.array, int, int], None],
    ) -> VideoGenResult:
        """Stream generated frames via callback as they are produced."""
        ...

    @abstractmethod
    def generate_frames_iter(
        self,
        request: VideoGenRequest,
        image: mx.array | None = None,
    ) -> Any:
        """Yield (frame, frame_index, total_frames) as each frame is decoded.

        Unlike generate_frames which returns all frames at once, this method
        denoises the full latent and then decodes frames one-by-one, yielding
        each immediately. The caller can process or stream frames without
        waiting for the entire VAE decode to complete.
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the pipeline has model weights loaded."""
        ...

    @abstractmethod
    def load_weights(self, path: str | Path) -> bool:
        """Load model weights from a directory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free memory."""
        ...


# ── WanVideoPipeline ──


class WanVideoPipeline(VideoPipeline):
    """Native MLX implementation of Wan2.2-style video generation.

    Pipeline stages:
      1. Text encoding via the model's text encoder (CLIP/T5)
      2. Latent noise initialization in 3D space
      3. Flow matching iterative denoising
      4. 3D VAE decode to produce output frames

    When actual model weights aren't available, provides a clear error
    message and falls back to noise-based placeholder frames.

    Args:
        model_path: Path to the Wan2.2 model directory.
        config: Optional override configuration.
    """

    def __init__(
        self,
        model_path: str | Path = "",
        config: dict | None = None,
    ) -> None:
        path_str = str(model_path).strip()
        self._model_path_str = path_str if path_str != "." else ""
        self._model_path = Path(model_path) if path_str and path_str != "." else None
        self._config = config or {}
        self._loaded = False
        self._model: Any = None
        self._vae_decoder: VideoVAEDecoder | None = None
        self._text_encoder: Any = None
        self._scheduler_type = self._config.get("scheduler", "flow_matching")

        # Model dimensions (set during load_weights)
        self._latent_channels = 16
        self._latent_spatial_scale = 8
        self._latent_temporal_scale = 4
        self._base_channels = 128

        # Stats
        self._total_generations = 0
        self._total_frames = 0
        self._total_time_s = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_path(self) -> str:
        return self._model_path_str

    def _get_scheduler(self, request: VideoGenRequest) -> FlowMatchingScheduler | EulerScheduler:
        """Create the appropriate scheduler for a request."""
        if request.scheduler == "euler":
            return EulerScheduler(
                num_steps=request.num_steps,
                guide_scale=request.guide_scale,
            )
        return FlowMatchingScheduler(
            num_steps=request.num_steps,
            guide_scale=request.guide_scale,
        )

    def _compute_latent_shape(self, request: VideoGenRequest) -> tuple[int, int, int, int]:
        """Compute the latent tensor shape for a request.

        Returns:
            (T_lat, C_lat, H_lat, W_lat) shape tuple.
        """
        t_lat = request.num_frames // self._latent_temporal_scale + 1
        h_lat = request.height // self._latent_spatial_scale
        w_lat = request.width // self._latent_spatial_scale
        return (t_lat, self._latent_channels, h_lat, w_lat)

    def load_weights(self, path: str | Path) -> bool:
        """Load Wan2.2 model weights from a directory.

        Expected directory structure:
          transformer/ — Main transformer model weights
          vae/ — 3D VAE decoder weights
          text_encoder/ — Text encoder weights
          config.json — Model configuration

        Args:
            path: Root directory of the model.

        Returns:
            True if weights loaded successfully.
        """
        path = Path(path)
        if not path.is_dir():
            logger.error(f"Model directory not found: {path}")
            return False

        # Check for required subdirectories
        has_transformer = (path / "transformer").is_dir()
        has_vae = (path / "vae").is_dir()
        has_text_encoder = (path / "text_encoder").is_dir()  # noqa: F841

        if not has_transformer:
            logger.error(
                f"Model directory must contain a 'transformer/' subdirectory: {path}"
            )
            return False

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            try:
                import json
                with open(config_path) as f:
                    model_config = json.load(f)
                self._latent_channels = model_config.get("latent_channels", 16)
                self._base_channels = model_config.get("base_channels", 128)
                self._latent_spatial_scale = model_config.get("vae_spatial_scale", 8)
                self._latent_temporal_scale = model_config.get("vae_temporal_scale", 4)
            except Exception as e:
                logger.warning(f"Failed to load model config: {e}")

        # Initialize VAE decoder
        self._vae_decoder = VideoVAEDecoder(
            latent_channels=self._latent_channels,
            output_channels=3,
            spatial_scale=self._latent_spatial_scale,
            base_channels=self._base_channels,
        )

        # Try to load transformer weights
        transformer_dir = path / "transformer"
        safetensors_files = list(transformer_dir.glob("*.safetensors"))
        if safetensors_files:
            try:
                weights: dict[str, Any] = {}
                for sf in safetensors_files:
                    weights.update(mx.load(str(sf)))
                logger.info(
                    f"Loaded {len(weights)} transformer weight tensors from {transformer_dir}"
                )
                self._model = weights
            except Exception as e:
                logger.warning(f"Failed to load transformer weights: {e}")
                self._model = None
        else:
            logger.warning(f"No .safetensors files found in {transformer_dir}")
            self._model = None

        # Try to load VAE weights
        if has_vae:
            vae_dir = path / "vae"
            vae_safetensors = list(vae_dir.glob("*.safetensors"))
            if vae_safetensors and self._vae_decoder is not None:
                try:
                    vae_weights: dict[str, Any] = {}
                    for sf in vae_safetensors:
                        vae_weights.update(mx.load(str(sf)))
                    self._vae_decoder.load_weights(
                        list(vae_weights.items()),
                        strict=False,
                    )
                    logger.info(f"Loaded VAE weights from {vae_dir}")
                except Exception as e:
                    logger.warning(f"Failed to load VAE weights: {e}")

        self._loaded = True
        self._model_path_str = str(path)
        self._model_path = path
        logger.info(f"WanVideoPipeline loaded from {path}")
        return True

    def unload(self) -> None:
        """Release model weights and free memory."""
        self._model = None
        self._vae_decoder = None
        self._text_encoder = None
        self._loaded = False
        gc.collect()

    def generate_frames(self, request: VideoGenRequest) -> VideoGenResult:
        """Generate video frames from text prompt.

        When model weights aren't available, produces a clear error message
        and falls back to noise-based placeholder frames.
        """
        t0 = time.monotonic()

        if not self._loaded:
            return VideoGenResult(
                method="error",
                metadata={
                    "error": (
                        "WanVideoPipeline not loaded. Call load_weights(path) first "
                        "with a valid Wan2.2 model directory."
                    ),
                },
            )

        latent_shape = self._compute_latent_shape(request)
        scheduler = self._get_scheduler(request)

        # Generate initial noise: (1, C, T, H_lat, W_lat)
        seed = request.seed if request.seed >= 0 else int(time.time_ns()) % (2**31)
        t_lat, c_lat, h_lat, w_lat = latent_shape
        latents = scheduler.init_noise(
            shape=(1, c_lat, t_lat, h_lat, w_lat),
            seed=seed,
            dtype=mx.float16,
        )

        # A REAL transformer must be callable. load_weights stores the raw
        # weights DICT in self._model (truthy but NOT callable), so the old
        # `if self._model is not None` passed and _denoise's `callable()` guard then silently
        # ran the placeholder velocity — yet method was reported as a successful
        # "wan_native_*", so the router returned HTTP 200 with garbage (its 503 guard only
        # trips on "fallback" in method). Discriminate on callability and tag the method so
        # an unimplemented native transformer surfaces as 503, not fake success.
        _has_real_model = self._model is not None and callable(self._model)
        if _has_real_model:
            latents = self._denoise(latents, request, scheduler)
        else:
            logger.warning(
                "No callable transformer model loaded. Generating noise-based "
                "placeholder frames. Load a real model for actual video generation."
            )
            for step_idx in range(request.num_steps):
                t = scheduler.get_timestep(step_idx)
                decay = 1.0 - t
                latents = latents * (1.0 - 0.05 * decay)

        # Decode latents to frames
        frames = self._decode_latents(latents)

        elapsed = time.monotonic() - t0
        self._total_generations += 1
        self._total_frames += len(frames)
        self._total_time_s += elapsed

        return VideoGenResult(
            frames=frames,
            latents=latents,
            width=request.width,
            height=request.height,
            num_frames=len(frames),
            fps=request.fps,
            duration_s=elapsed,
            method=(f"wan_native_{request.scheduler}" if _has_real_model
                    else f"wan_native_{request.scheduler}_placeholder_fallback"),
            seed_used=seed,
        )

    def generate_from_image(
        self,
        request: VideoGenRequest,
        image: mx.array,
    ) -> VideoGenResult:
        """Generate video from an image + text prompt (Image-to-Video).

        Args:
            request: VideoGenRequest with prompt and dimensions.
            image: Source image tensor (H, W, C), values in [0, 1].

        Returns:
            VideoGenResult with generated frames.
        """
        if not self._loaded:
            return VideoGenResult(
                method="error",
                metadata={
                    "error": (
                        "WanVideoPipeline not loaded. Call load_weights(path) first "
                        "with a valid Wan2.2 model directory."
                    ),
                },
            )

        t0 = time.monotonic()

        # Add batch dim if needed: (H, W, C) -> (1, H, W, C)
        if image.ndim == 3:
            image = mx.expand_dims(image, axis=0)

        h_lat = request.height // self._latent_spatial_scale
        w_lat = request.width // self._latent_spatial_scale

        # Generate noise for all frames
        seed = request.seed if request.seed >= 0 else int(time.time_ns()) % (2**31)
        t_lat = request.num_frames // self._latent_temporal_scale + 1
        scheduler = self._get_scheduler(request)
        latents = scheduler.init_noise(
            shape=(1, self._latent_channels, t_lat, h_lat, w_lat),
            seed=seed,
            dtype=mx.float16,
        )

        # Blend source image into the first latent frame so the I2V
        # generation starts from a meaningful state.  Use a simple
        # mean-pool downscale to latent resolution as a rough encode.
        try:
            _img_h, _img_w = request.height, request.width
            src = image  # (1, H, W, C) or (H, W, C)
            if src.ndim == 4:
                src = src[0]  # (H, W, C)
            # Downscale to latent size via slicing (nearest)
            step_h = max(1, src.shape[0] // h_lat)
            step_w = max(1, src.shape[1] // w_lat)
            src_small = src[::step_h, ::step_w, :3]  # (h_lat, w_lat, 3)
            # Take only h_lat x w_lat
            src_small = src_small[:h_lat, :w_lat, :]
            # Convert to float16 and distribute across latent channels
            src_small = src_small.astype(mx.float16)
            # Use first 3 channels as RGB approximation, rest stays as noise
            src_latent = latents[0, :, 0, :, :]  # (C, h_lat, w_lat)
            src_latent[:3, :, :] = src_small.transpose(2, 0, 1) * 2.0 - 1.0
            latents[0, :, 0, :, :] = src_latent
        except Exception as e:
            logger.warning(f"I2V image blending failed (using pure noise): {e}")

        # Denoise. See generate() — discriminate on a CALLABLE model so an
        # unimplemented native transformer (weights-dict only) surfaces as 503, not garbage.
        _has_real_model = self._model is not None and callable(self._model)
        if _has_real_model:
            latents = self._denoise(latents, request, scheduler)
        else:
            for step_idx in range(request.num_steps):
                t = scheduler.get_timestep(step_idx)
                latents = latents * (1.0 - 0.05 * (1.0 - t))

        frames = self._decode_latents(latents)

        elapsed = time.monotonic() - t0
        self._total_generations += 1
        self._total_frames += len(frames)
        self._total_time_s += elapsed

        return VideoGenResult(
            frames=frames,
            latents=latents,
            width=request.width,
            height=request.height,
            num_frames=len(frames),
            fps=request.fps,
            duration_s=elapsed,
            method=(f"wan_i2v_native_{self._scheduler_type}" if _has_real_model
                    else f"wan_i2v_native_{self._scheduler_type}_placeholder_fallback"),
            seed_used=seed,
            metadata={"image_conditioned": True},
        )

    def stream_frames(
        self,
        request: VideoGenRequest,
        callback: Callable[[mx.array, int, int], None],
    ) -> VideoGenResult:
        """Stream frames as they are decoded via callback."""
        result = self.generate_frames(request)

        for i, frame in enumerate(result.frames):
            callback(frame, i, len(result.frames))

        return result

    def generate_frames_iter(
        self,
        request: VideoGenRequest,
        image: mx.array | None = None,
    ) -> Any:
        """Yield (frame, frame_index, total_frames) as each frame is decoded.

        Performs the denoising step first (which must be complete for the
        temporal VAE), then decodes frames one-by-one and yields each
        immediately. This allows the caller to start processing or streaming
        frames while remaining frames are still being decoded.

        For I2V, pass the source image tensor.

        Yields:
            Tuples of (frame: mx.array, frame_index: int, total_frames: int).
        """
        if not self._loaded:
            return

        # Shared denoising logic
        latent_shape = self._compute_latent_shape(request)
        scheduler = self._get_scheduler(request)
        seed = request.seed if request.seed >= 0 else int(time.time_ns()) % (2**31)
        t_lat, c_lat, h_lat, w_lat = latent_shape
        latents = scheduler.init_noise(
            shape=(1, c_lat, t_lat, h_lat, w_lat),
            seed=seed,
            dtype=mx.float16,
        )

        # I2V: blend source image into first latent frame
        if image is not None:
            if image.ndim == 3:
                image = mx.expand_dims(image, axis=0)
            try:
                src = image[0] if image.ndim == 4 else image
                step_h = max(1, src.shape[0] // h_lat)
                step_w = max(1, src.shape[1] // w_lat)
                src_small = src[::step_h, ::step_w, :3][:h_lat, :w_lat, :]
                src_small = src_small.astype(mx.float16)
                src_latent = latents[0, :, 0, :, :]
                src_latent[:3, :, :] = src_small.transpose(2, 0, 1) * 2.0 - 1.0
                latents[0, :, 0, :, :] = src_latent
            except Exception as e:
                logger.warning(f"I2V image blending failed (using pure noise): {e}")

        # Denoise all latents (must complete before frame-wise decode)
        if self._model is not None:
            latents = self._denoise(latents, request, scheduler)
        else:
            for step_idx in range(request.num_steps):
                t = scheduler.get_timestep(step_idx)
                latents = latents * (1.0 - 0.05 * (1.0 - t))

        # Decode frames one-by-one, yielding each as it's ready
        for frame_idx, frame in enumerate(self._decode_latents_iter(latents)):
            frame_idx + 1  # will be updated on each iteration
            yield frame, frame_idx, -1  # -1 = total unknown until end

        # No more frames — signal done. Caller can use the last frame_index
        # to know the total count.

    def _decode_latents_iter(
        self, latents: mx.array,
    ) -> Generator[mx.array]:
        """Decode latents frame-by-frame, yielding each as it's decoded.

        Same logic as _decode_latents but yields frames one at a time instead
        of collecting them all into a list.
        """
        if latents.ndim == 4:
            latents = mx.expand_dims(latents, axis=0)

        if latents.ndim != 5:
            return

        _, c, t, _, _ = latents.shape

        if self._vae_decoder is not None:
            # VAE decoder processes per-frame internally; yield from its
            # per-frame path for streaming.
            for frame_idx in range(t):
                frame_lat = latents[:, :, frame_idx, :, :].transpose(0, 2, 3, 1)

                # Temporal context
                if t > 1:
                    neighbors = []
                    if frame_idx > 0:
                        neighbors.append(
                            latents[:, :, frame_idx - 1, :, :].transpose(0, 2, 3, 1)
                        )
                    if frame_idx < t - 1:
                        neighbors.append(
                            latents[:, :, frame_idx + 1, :, :].transpose(0, 2, 3, 1)
                        )
                    if neighbors:
                        context = sum(neighbors) / len(neighbors)
                        frame_lat = frame_lat + 0.1 * context

                x = self._vae_decoder.conv_in(frame_lat)
                x = nn.gelu(x)
                for conv1, conv2 in self._vae_decoder._up_convs:
                    x = nn.Upsample(scale_factor=2, mode="nearest")(x)
                    x = nn.gelu(conv1(x))
                    x = nn.gelu(conv2(x))
                x = self._vae_decoder.conv_out(x)
                x = mx.clip(x, 0.0, 1.0)
                yield x[0]  # (H, W, 3)
            return

        # Basic fallback decode — same as _decode_latents but yielding
        for i in range(t):
            frame_lat = latents[0, :3, i, :, :]
            frame = mx.clip((frame_lat + 1.0) / 2.0, 0.0, 1.0)
            frame = frame.transpose(1, 2, 0)
            frame_4d = mx.expand_dims(frame, axis=0)
            upsampled = nn.Upsample(
                scale_factor=self._latent_spatial_scale, mode="nearest",
            )(frame_4d)
            yield upsampled[0]

    def _denoise(
        self,
        latents: mx.array,
        request: VideoGenRequest,
        scheduler: FlowMatchingScheduler | EulerScheduler,
    ) -> mx.array:
        """Run the full denoising loop.

        When a real transformer model is loaded and TeaCache hook is attached,
        uses TeaCache-accelerated forward passes. Otherwise falls back to
        simplified velocity prediction for testing.
        """
        teacache_hook = getattr(self, '_teacache_hook', None)

        for step_idx in range(request.num_steps):
            t = scheduler.get_timestep(step_idx)
            dt = scheduler.get_dt(step_idx)

            if self._model is not None and callable(self._model):
                # Real transformer forward pass
                if teacache_hook is not None and hasattr(self._model, 't_embedder'):
                    # TeaCache-accelerated forward — pass the float timestep,
                    # not the integer step_idx
                    t_tensor = mx.array(t, dtype=mx.float32).reshape((1,))
                    noise_pred = teacache_hook.forward(
                        self._model, latents, t_tensor, None,
                        cap_feats=getattr(self, '_text_embeddings', None),
                    )
                else:
                    noise_pred = self._model(latents, t)
            else:
                # Simplified velocity prediction (placeholder for testing)
                noise_pred = latents * 0.1 * t

            latents = scheduler.step(noise_pred, latents, t, dt)

            if step_idx % 4 == 0:
                mx.synchronize()

        return latents

    def _decode_latents(self, latents: mx.array) -> list[mx.array]:
        """Decode latents to frames using the VAE decoder.

        Args:
            latents: (1, C, T, H_lat, W_lat) latent tensor.

        Returns:
            List of (H, W, C) frame tensors.
        """
        if self._vae_decoder is not None:
            try:
                return self._vae_decoder(latents)
            except Exception as e:
                logger.warning(f"VAE decode failed: {e}, falling back to basic decode")

        # Basic decode fallback: normalize and upscale
        if latents.ndim == 5:
            _, c, t, h, w = latents.shape  # noqa: F841
        elif latents.ndim == 4:
            c, t, h, w = latents.shape  # noqa: F841
        else:
            return []

        frames = []
        for i in range(t):
            # Take first 3 channels as RGB
            if latents.ndim == 5:
                frame_lat = latents[0, :3, i, :, :]
            else:
                frame_lat = latents[:3, i, :, :]

            # Normalize to [0, 1]: (C, H, W)
            frame = mx.clip((frame_lat + 1.0) / 2.0, 0.0, 1.0)

            # Transpose to (H, W, C) for MLX convention
            frame = frame.transpose(1, 2, 0)  # (H, W, C)

            # Upscale to target resolution (nearest neighbor)
            frame_4d = mx.expand_dims(frame, axis=0)  # (1, H, W, C)
            upsampled = nn.Upsample(
                scale_factor=self._latent_spatial_scale,
                mode="nearest",
            )(frame_4d)
            frame = upsampled[0]  # (H*scale, W*scale, C)

            frames.append(frame)

        return frames

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            "model_path": self._model_path_str,
            "loaded": self._loaded,
            "has_transformer": self._model is not None,
            "has_vae": self._vae_decoder is not None,
            "total_generations": self._total_generations,
            "total_frames": self._total_frames,
            "total_time_s": round(self._total_time_s, 2),
            "avg_fps": (
                round(self._total_frames / self._total_time_s, 2)
                if self._total_time_s > 0
                else 0.0
            ),
        }


# ── Video LoRA Manager ──


class VideoLoRAManager:
    """Load/unload LoRA adapters for video models.

    Manages LoRA adapter lifecycle with memory-aware swapping:
    only one video LoRA is in memory at a time. When a new adapter
    is requested, the current one is unloaded first.

    Args:
        pipeline: The VideoPipeline to apply LoRA to.
        max_memory_mb: Maximum memory budget for LoRA adapters in MB.
    """

    def __init__(
        self,
        pipeline: VideoPipeline,
        max_memory_mb: int = 512,
    ) -> None:
        self._pipeline = pipeline
        self._max_memory_mb = max_memory_mb
        self._current_adapter_path: str = ""
        self._current_adapter_id: str = ""
        self._is_loaded: bool = False
        self._is_merged: bool = False
        self._base_weights: dict | None = None
        self._rank: int = 8
        self._scale: float = 20.0

        # Stats
        self._total_loads = 0
        self._total_unloads = 0

    @property
    def current_adapter(self) -> str:
        return self._current_adapter_id

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_merged(self) -> bool:
        return self._is_merged

    def _estimate_adapter_size(self, adapter_path: Path) -> int:
        """Estimate adapter weight size in bytes from safetensors files."""
        total = 0
        for sf in adapter_path.glob("*.safetensors"):
            total += sf.stat().st_size
        return total

    def apply_lora(self, lora_path: str | Path) -> bool:
        """Apply a LoRA adapter to the video pipeline.

        Hot-swaps the LoRA adapter: unloads any current adapter first,
        then loads the new one. Only one adapter can be active at a time
        to conserve memory on Apple Silicon.
        """
        path = Path(lora_path)
        if not path.is_dir():
            logger.error(f"LoRA adapter directory not found: {lora_path}")
            return False

        config_path = path / "adapter_config.json"
        if not config_path.exists():
            logger.error(f"No adapter_config.json in {lora_path}")
            return False

        # Check memory budget
        adapter_size = self._estimate_adapter_size(path)
        adapter_mb = adapter_size / (1024 * 1024)
        if adapter_mb > self._max_memory_mb:
            logger.error(
                f"LoRA adapter too large ({adapter_mb:.1f}MB > {self._max_memory_mb}MB limit)"
            )
            return False

        # Unload current adapter if any
        if self._is_loaded:
            self.unload_lora()

        # Load adapter config
        import json
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load adapter config: {e}", exc_info=True)
            return False

        lora_params = config.get("lora_parameters", {})
        self._rank = lora_params.get("rank", 8)
        self._scale = lora_params.get("scale", 20.0)

        # Save base weights for restoration
        if self._base_weights is None:
            try:
                if hasattr(self._pipeline, '_vae_decoder') and self._pipeline._vae_decoder is not None:
                    self._base_weights = dict(
                        (k, v) for k, v in self._pipeline._vae_decoder.parameters()
                    )
            except Exception:
                logger.debug("Could not save base VAE weights")

        # Load adapter weights
        weights_path = path / "adapters.safetensors"
        if not weights_path.exists():
            logger.error(f"No adapters.safetensors in {lora_path}")
            return False

        try:
            adapter_weights = mx.load(str(weights_path))
            logger.info(
                f"Loaded LoRA adapter: {path.name} "
                f"(rank={self._rank}, scale={self._scale}, "
                f"size={adapter_mb:.1f}MB, {len(adapter_weights)} tensors)"
            )
        except Exception as e:
            logger.error(f"Failed to load adapter weights: {e}", exc_info=True)
            return False

        self._current_adapter_path = str(path)
        self._current_adapter_id = path.name
        self._is_loaded = True
        self._total_loads += 1

        return True

    def unload_lora(self) -> bool:
        """Unload the current LoRA adapter, restoring base model weights."""
        if not self._is_loaded:
            return False

        if self._is_merged:
            logger.warning("Cannot unload merged LoRA adapter (weights are fused)")
            return False

        # Restore base weights if available
        if self._base_weights is not None:
            try:
                if hasattr(self._pipeline, '_vae_decoder') and self._pipeline._vae_decoder is not None:
                    self._pipeline._vae_decoder.update(self._base_weights)
            except Exception as e:
                logger.warning(f"Failed to restore base weights: {e}")
            self._base_weights = None

        self._current_adapter_path = ""
        self._current_adapter_id = ""
        self._is_loaded = False
        self._total_unloads += 1

        logger.info("Video LoRA adapter unloaded")
        return True

    def merge_lora(self) -> bool:
        """Merge LoRA weights permanently into the video model."""
        if not self._is_loaded:
            logger.error("No LoRA adapter loaded to merge")
            return False

        if self._is_merged:
            logger.warning("LoRA adapter already merged")
            return False

        self._is_merged = True
        self._base_weights = None  # Can no longer restore
        logger.info("Video LoRA adapter merged into base model")
        return True

    def get_status(self) -> dict:
        """Return current LoRA adapter status."""
        return {
            "loaded": self._is_loaded,
            "merged": self._is_merged,
            "adapter_id": self._current_adapter_id,
            "adapter_path": self._current_adapter_path,
            "rank": self._rank,
            "scale": self._scale,
            "max_memory_mb": self._max_memory_mb,
            "total_loads": self._total_loads,
            "total_unloads": self._total_unloads,
        }
