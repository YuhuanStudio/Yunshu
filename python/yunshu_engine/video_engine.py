from __future__ import annotations

"""Yunshu Video Generation Engine — wraps mlx-video for Wan2.2 and LTX2.

Provides a unified interface for:
1. Text-to-Video (T2V): Generate video from text prompt
2. Image-to-Video (I2V): Animate a static image from text prompt + image
3. Streaming Frames: Yield processed frames as decoded (real-time analysis)
4. Frame Batching: Process multiple frames in a single batch
5. Video LoRA: Load/unload LoRA adapters for video models

Architecture:
  VideoEngine wraps mlx-video's Wan2.2 and LTX2 pipelines with:
  - Unified async generate/generate_stream interface
  - Gateway endpoint at /v1/video/generations
  - Automatic model detection (Wan2.2 vs LTX2)
  - Frame extraction as PNG sequence or MP4 encoding
  - Memory-aware tiling for large videos
  - Streaming frame decoder for real-time video analysis
  - LoRA adapter management (load/unload/merge)
  - Frame batching for improved GPU utilization

Integration:
  - ModelType.VIDEO in model_manager for auto-detection
  - Gateway endpoint at /v1/video/generations
  - Reuses mlx-video's native MLX computation

Env vars:
  YUNSHU_VIDEO_STREAMING=1   Enable streaming frame decoder
  YUNSHU_VIDEO_LORA=path     Auto-load LoRA adapter at startup
"""

import asyncio
import contextlib
import gc
import io
import json
import logging
import os
import queue
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VideoGenConfig:
    """Configuration for video generation."""
    width: int = 1280
    height: int = 704
    num_frames: int = 81  # Must be 4n+1 for Wan2.2
    num_steps: int = 20
    guide_scale: float = 5.0
    fps: int = 16
    scheduler: str = "unipc"  # unipc, euler, dpm++
    seed: int = -1  # -1 = random


@dataclass
class VideoGenOutput:
    """Result from video generation."""
    video_data: bytes = b""  # MP4 bytes
    frames: list[bytes] = field(default_factory=list)  # PNG bytes per frame
    width: int = 0
    height: int = 0
    num_frames: int = 0
    fps: int = 16
    duration_s: float = 0.0
    method: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class FrameBatch:
    """A batch of processed video frames for efficient GPU processing."""
    frames: list[bytes] = field(default_factory=list)   # PNG bytes per frame
    frame_indices: list[int] = field(default_factory=list)  # Original frame indices
    width: int = 0
    height: int = 0
    batch_size: int = 0


@dataclass
class VideoStats:
    """Runtime statistics for video engine."""
    frames_processed: int = 0
    frames_streamed: int = 0
    batches_processed: int = 0
    total_generate_calls: int = 0
    total_stream_calls: int = 0
    total_frame_decode_ms: float = 0.0
    total_generate_ms: float = 0.0
    lora_adapter_id: str = ""
    lora_loaded: bool = False
    lora_merged: bool = False

    @property
    def avg_fps(self) -> float:
        """Average frames-per-second across all processing."""
        if self.total_generate_ms <= 0:
            return 0.0
        return self.frames_processed / (self.total_generate_ms / 1000.0)

    @property
    def avg_decode_fps(self) -> float:
        """Average frame decode FPS for streaming."""
        if self.total_frame_decode_ms <= 0:
            return 0.0
        return self.frames_streamed / (self.total_frame_decode_ms / 1000.0)


from .active_tracking import ActiveRequestMixin, tracks_active, tracks_active_gen


class VideoEngine(ActiveRequestMixin):
    """Video generation engine wrapping mlx-video.

    Supports:
    - Wan 2.2 (T2V and I2V) — via mlx_video.models.wan_2
    - LTX 2.0 (T2V) — via mlx_video.models.ltx_2
    - Streaming frame decoder for real-time analysis
    - Frame batching for improved GPU utilization
    - LoRA adapter loading/unloading

    Falls back gracefully when mlx-video is not installed.
    """

    def __init__(self, model_path: str = "", config: VideoGenConfig | None = None) -> None:
        self._model_path = model_path
        self._config = config or VideoGenConfig()
        self._model = None
        self._model_type = self._detect_model_type()
        self._running = False
        from .mlx_executor import get_mlx_executor
        self._executor = get_mlx_executor()

        # Stats tracking
        self._stats = VideoStats()
        self._stats_lock = threading.Lock()

        # LoRA state
        self._lora_adapter_path: str = ""
        self._lora_loaded: bool = False
        self._lora_merged: bool = False
        self._lora_lock = threading.RLock()

        # Native MLX video pipeline (Wan2.2/LTX2)
        self._native_pipeline = None  # Created lazily after model load
        self._lora_rank: int = 8
        self._lora_scale: float = 20.0
        self._base_model_weights: dict | None = None

        # Env var: auto-load LoRA adapter
        env_lora = os.environ.get("YUNSHU_VIDEO_LORA", "").strip()
        if env_lora and os.path.isdir(env_lora):
            self._lora_adapter_path = env_lora
            logger.info(f"Video LoRA adapter path set from env: {env_lora}")

        # TeaCache config for diffusion acceleration (opt-in via YUNSHU_VIDEO_TEACACHE).
        # Per-request instances are created in each generation call to prevent
        # state leakage between concurrent requests.
        self._teacache_config = None
        teacache_env = os.environ.get("YUNSHU_VIDEO_TEACACHE", "").strip()
        if teacache_env in ("1", "true", "yes"):
            from .teacache import TeaCacheConfig
            self._teacache_config = TeaCacheConfig(rel_l1_thresh=0.2)
            logger.info("Video TeaCache enabled (threshold=0.2)")
        elif teacache_env and teacache_env not in ("0", "false", "no"):
            try:
                thresh = float(teacache_env)
                from .teacache import TeaCacheConfig
                self._teacache_config = TeaCacheConfig(rel_l1_thresh=thresh)
                logger.info(f"Video TeaCache enabled (threshold={thresh})")
            except ValueError:
                pass

    @property
    def model_name(self) -> str:
        return self._model_path.rsplit("/", 1)[-1] if "/" in self._model_path else "video-default"

    @property
    def is_loaded(self) -> bool:
        # Video engine uses lazy model loading — start() marks the engine as
        # ready to accept requests (which may fall back to placeholder frames
        # when no model is on disk).  Report loaded when running, and indicate
        # whether a real model is available via get_stats()["model_loaded"].
        return self._running

    def _detect_model_type(self) -> str:
        """Detect video model type from path and config."""
        path_lower = self._model_path.lower()

        # Config-based detection (more reliable)
        config_path = os.path.join(self._model_path, "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.loads(f.read())
                model_type = cfg.get("model_type", "").lower()
                # Wan config model_type is the TASK variant (ti2v / t2v / i2v),
                # not the engine name. mlx-video's wan_2 generate path handles
                # all Wan2.x variants (verified live on Wan2.2-TI2V-5B), so map
                # them to the wan_2_2 dispatch key instead of falling through to
                # a non-existent mlx_video.models.<variant> import.
                if model_type in ("ti2v", "t2v", "i2v", "wan2.1", "wan2.2", "wan"):
                    return "wan_2_2"
                if model_type:
                    return model_type
            except Exception:
                logger.debug("video config read failed", exc_info=True)

        # Path-based fallback
        if "cogvideo" in path_lower:
            return "cogvideox"
        if "hunyuan" in path_lower:
            return "hunyuan_video"
        if "mochi" in path_lower:
            return "mochi"
        if "cogview" in path_lower:
            return "cogview4"
        if "sora" in path_lower:
            return "open_sora"
        if "pyramid" in path_lower:
            return "pyramid_flow"
        if "animate" in path_lower:
            return "animatediff"
        if "stable_video" in path_lower or "svd" in path_lower:
            return "stable_video_diffusion"
        if "ltx" in path_lower:
            return "ltx_2"
        if "wan" in path_lower:
            return "wan_2_2"
        return "wan_2_2"  # Default to Wan

    def start(self) -> None:
        """Initialize the video engine."""
        if self._running:
            return
        logger.info(f"Starting video engine: {self._model_path or 'default'} (type={self._model_type})")
        # Model loading happens lazily during first generation
        self._running = True

        # Auto-load LoRA if env var set and model already loaded
        with self._lora_lock:
            if self._lora_adapter_path and not self._lora_loaded:
                self.load_lora_adapter(self._lora_adapter_path)

        logger.info("Video engine started")

    def stop(self) -> None:
        """Stop and release resources.

        Idempotent: safe to call multiple times. Uses sync cleanup
        to be callable from both sync and async contexts. For async
        callers that need executor-routed GPU cleanup, use `await stop_async()`.
        """
        if not self._running and self._model is None:
            return
        self._running = False
        self._teacache_config = None
        self.unload_lora_adapter()
        self._model = None
        self._base_model_weights = None
        self._native_pipeline = None
        with self._lora_lock:
            self._lora_loaded = False
            self._lora_merged = False
            self._lora_adapter_path = ""
        gc.collect()
        try:
            import concurrent.futures

            import mlx.core as mx
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(lambda: (mx.synchronize(), mx.clear_cache())).result()
        except Exception:
            logger.debug("MLX cache clear in video stop failed", exc_info=True)

    async def stop_async(self) -> None:
        """Async variant that routes cache clearing through the MLX executor."""
        if not self._running and self._model is None:
            return
        self._running = False
        self._teacache_config = None
        self.unload_lora_adapter()
        self._model = None
        self._base_model_weights = None
        self._native_pipeline = None
        with self._lora_lock:
            self._lora_loaded = False
            self._lora_merged = False
            self._lora_adapter_path = ""
        gc.collect()
        try:
            import asyncio

            from .mlx_executor import sync_and_clear_cache
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, sync_and_clear_cache)
        except Exception:
            logger.debug("MLX cache clear in video stop failed", exc_info=True)

    def _get_model_dir(self) -> str:
        """Resolve model directory path."""
        if self._model_path and os.path.isdir(self._model_path):
            return self._model_path
        # Try common paths
        candidates = [
            os.path.expanduser("~/.cache/huggingface/hub/models--Wan2.2-T2V-1.3B"),
            os.path.expanduser("~/models/Wan2.2-T2V-1.3B"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return self._model_path or ""

    @tracks_active
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        image: bytes | None = None,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        num_steps: int | None = None,
        guide_scale: float | None = None,
        fps: int | None = None,
        seed: int | None = None,
        scheduler: str | None = None,
        output_format: str = "mp4",
    ) -> VideoGenOutput:
        """Generate video from text prompt (and optionally an image).

        Args:
            prompt: Text description of the video.
            negative_prompt: Negative text prompt.
            image: Optional source image bytes for I2V mode.
            width: Video width (default: 1280).
            height: Video height (default: 704).
            num_frames: Number of frames (default: 81, must be 4n+1).
            num_steps: Denoising steps (default: 20).
            guide_scale: Guidance scale (default: 5.0).
            fps: Output FPS (default: 16).
            seed: Random seed.
            scheduler: Scheduler type (unipc, euler, dpm++).
            output_format: "mp4" or "frames" (list of PNGs).

        Returns:
            VideoGenOutput with video data or individual frames.
        """
        if not self._running:
            self.start()

        cfg = self._config
        w = width or cfg.width
        h = height or cfg.height
        nf = num_frames or cfg.num_frames
        ns = num_steps or cfg.num_steps
        gs = guide_scale if guide_scale is not None else cfg.guide_scale
        f = fps or cfg.fps
        s = seed if seed is not None else cfg.seed
        # Randomize seed when default sentinel -1 is used — without this,
        # every request without an explicit seed produces identical output.
        if s < 0:
            s = int(time.time_ns()) % (2**31)
        sched = scheduler or cfg.scheduler

        # Validate num_frames for Wan2.2 (must be 4n+1)
        if self._model_type == "wan_2_2" and (nf - 1) % 4 != 0:
            corrected = ((nf - 1) // 4) * 4 + 1
            logger.warning(
                f"Wan2.2 requires num_frames=4n+1, adjusting {nf} -> {corrected}"
            )
            nf = corrected

        # Validate dimensions are positive and even
        if w < 1 or h < 1:
            raise ValueError(f"width and height must be >= 1, got {w}x{h}")
        if w % 2 != 0 or h % 2 != 0:
            w = (w // 2) * 2
            h = (h // 2) * 2
            logger.warning(f"Adjusted dimensions to even: {w}x{h}")

        def _gen_sync() -> VideoGenOutput:
            return self._run_generation(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                width=w, height=h,
                num_frames=nf, num_steps=ns,
                guide_scale=gs, fps=f, seed=s,
                scheduler=sched, output_format=output_format,
            )

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, _gen_sync)
        elapsed = time.monotonic() - t0
        result.duration_s = elapsed

        # Update stats
        with self._stats_lock:
            self._stats.total_generate_calls += 1
            self._stats.total_generate_ms += elapsed * 1000.0
            self._stats.frames_processed += result.num_frames

        logger.info(f"Video gen: {elapsed:.2f}s, {nf} frames, prompt='{prompt[:50]}...'")
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Generation-time frame streaming — yield frames as they're decoded
    # ═══════════════════════════════════════════════════════════════════════

    @tracks_active_gen
    async def generate_stream(
        self,
        prompt: str,
        negative_prompt: str = "",
        image: bytes | None = None,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        num_steps: int | None = None,
        guide_scale: float | None = None,
        fps: int | None = None,
        seed: int | None = None,
        scheduler: str | None = None,
    ) -> AsyncIterator[dict]:
        """Stream frames progressively as they are generated and decoded.

        Unlike generate() which waits for all frames to be ready before
        returning, this method yields each frame immediately after it is
        decoded by the VAE. The denoising step must complete first (it
        operates on all latents jointly), but once denoising finishes,
        frames are decoded one-by-one and delivered without buffering.

        Falls back to generating and yielding placeholder frames when no
        model is available.

        Args: Same as generate() minus output_format.

        Yields:
            dict with keys:
              - "index": Frame index (0-based)
              - "frame": Frame PNG bytes
              - "width": Frame width
              - "height": Frame height
              - "is_final": True for the last frame
              - "method": Generation method string
        """
        if not self._running:
            self.start()

        cfg = self._config
        w = width or cfg.width
        h = height or cfg.height
        nf = num_frames or cfg.num_frames
        ns = num_steps or cfg.num_steps
        gs = guide_scale if guide_scale is not None else cfg.guide_scale
        f = fps or cfg.fps
        s = seed if seed is not None else cfg.seed
        if s < 0:
            s = int(time.time_ns()) % (2**31)
        sched = scheduler or cfg.scheduler

        # Validate num_frames for Wan2.2 (must be 4n+1)
        if self._model_type == "wan_2_2" and (nf - 1) % 4 != 0:
            corrected = ((nf - 1) // 4) * 4 + 1
            logger.warning(
                f"Wan2.2 requires num_frames=4n+1, adjusting {nf} -> {corrected}"
            )
            nf = corrected

        if w < 1 or h < 1:
            raise ValueError(f"width and height must be >= 1, got {w}x{h}")
        if w % 2 != 0 or h % 2 != 0:
            w = (w // 2) * 2
            h = (h // 2) * 2
            logger.warning(f"Adjusted dimensions to even: {w}x{h}")

        # Thread-safe queue + cancellation for executor -> async bridge
        import queue as _queue_mod
        _thread_queue: _queue_mod.Queue[dict | None] = _queue_mod.Queue(maxsize=128)
        _cancel = threading.Event()

        def _gen_stream_sync():
            """Synchronous frame-by-frame generation running in the MLX executor."""
            frames_emitted = 0
            t0 = time.monotonic()

            try:
                model_dir = self._get_model_dir()

                # Determine if we can use the native pipeline for streaming
                use_native = (
                    os.environ.get("YUNSHU_VIDEO_PIPELINE", "").strip() == "native"
                    or self._native_pipeline is not None
                )
                has_model = model_dir and os.path.isdir(model_dir)

                if has_model and use_native:
                    frames_emitted = self._stream_native_pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image,
                        width=w, height=h,
                        num_frames=nf, num_steps=ns,
                        guide_scale=gs, seed=s, scheduler=sched,
                        model_dir=model_dir,
                        thread_queue=_thread_queue,
                        cancel=_cancel,
                    )
                elif has_model:
                    # mlx-video path: generate full video, then decode frames
                    # progressively via ffmpeg pipe (better than buffering all)
                    frames_emitted = self._stream_mlx_video(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image,
                        width=w, height=h,
                        num_frames=nf, num_steps=ns,
                        guide_scale=gs, seed=s, scheduler=sched,
                        model_dir=model_dir,
                        fps=f,
                        thread_queue=_thread_queue,
                        cancel=_cancel,
                    )
                else:
                    # Fallback: yield placeholder frames one by one
                    frames_emitted = self._stream_fallback(
                        prompt=prompt, width=w, height=h,
                        num_frames=nf,
                        thread_queue=_thread_queue,
                        cancel=_cancel,
                    )

                # Update stats
                elapsed = time.monotonic() - t0
                with self._stats_lock:
                    self._stats.total_generate_calls += 1
                    self._stats.total_stream_calls += 1
                    self._stats.total_generate_ms += elapsed * 1000.0
                    self._stats.frames_processed += frames_emitted

            except Exception as e:
                logger.error(f"Stream generation error: {e}", exc_info=True)
            finally:
                # Signal completion
                with contextlib.suppress(_queue_mod.Full):
                    _thread_queue.put_nowait(None)

        # Launch the synchronous generator in the MLX executor
        loop = asyncio.get_running_loop()
        gen_task = loop.run_in_executor(self._executor, _gen_stream_sync)

        # Bridge thread-safe queue to async iterator
        try:
            while True:
                if _cancel.is_set():
                    break
                try:
                    chunk = _thread_queue.get_nowait()
                except _queue_mod.Empty:
                    await asyncio.sleep(0.01)
                    continue
                if chunk is None:
                    break
                yield chunk
                if chunk.get("is_final"):
                    break
        finally:
            _cancel.set()
            if not gen_task.done():
                gen_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await gen_task
            # Drain remaining items to unblock executor thread
            while True:
                try:
                    _thread_queue.get_nowait()
                except _queue_mod.Empty:
                    break

    def _stream_native_pipeline(
        self,
        prompt: str,
        negative_prompt: str,
        image: bytes | None,
        width: int,
        height: int,
        num_frames: int,
        num_steps: int,
        guide_scale: float,
        seed: int,
        scheduler: str,
        model_dir: str,
        thread_queue: queue.Queue[dict | None],
        cancel: threading.Event,
    ) -> int:
        """Stream frames from the native MLX pipeline frame-by-frame.

        Returns the number of frames emitted. Puts frame dicts into thread_queue.
        """
        import queue as _queue_mod

        from .video_pipeline import VideoGenRequest, WanVideoPipeline

        if self._native_pipeline is None:
            self._native_pipeline = WanVideoPipeline(model_path=model_dir)
            self._native_pipeline.load_weights(model_dir)
        # load_weights returning True only means the weights DICT loaded —
        # the native transformer is only "real" when self._model is CALLABLE. Without that,
        # generate_frames_iter runs the placeholder velocity and streams garbage as 200.
        # Treat a non-callable model like a load failure → honest fallback path. This
        # check runs on EVERY call (was nested in `is None`), so a non-callable
        # pipeline left set by a prior NON-streaming request can't be reused to stream
        # placeholder garbage.
        if not callable(getattr(self._native_pipeline, "_model", None)):
            logger.warning("Native pipeline has no callable transformer for stream; "
                           "falling back instead of streaming placeholder frames")
            self._native_pipeline = None
            return self._stream_fallback(
                prompt, width, height, num_frames,
                thread_queue, cancel,
            )

        request = VideoGenRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_steps=num_steps,
            guide_scale=guide_scale,
            seed=seed,
            scheduler=scheduler,
        )

        # Wire TeaCache — create per-request instance
        if self._teacache_config is not None:
            from .teacache import TeaCacheHook
            _local_tc = TeaCacheHook(self._teacache_config)
            _local_tc.reset()
            self._native_pipeline._teacache_hook = _local_tc
        else:
            _local_tc = None

        try:
            # Convert image bytes to mx.array for I2V
            image_mx = None
            if image is not None:
                import mlx.core as mx
                import numpy as np
                from PIL import Image as PILImage
                pil_img = PILImage.open(io.BytesIO(image)).convert("RGB")
                img_np = np.array(pil_img, dtype=np.float32) / 255.0
                image_mx = mx.array(img_np)

            import numpy as np
            from PIL import Image as PILImage

            # Use generate_frames_iter to get frames one at a time.
            # We use a one-frame look-ahead buffer so we can mark is_final
            # on the last frame accurately.
            if not self._running:
                return 0
            frame_iter = self._native_pipeline.generate_frames_iter(
                request, image=image_mx,
            )
            pending_data: dict | None = None
            frames_emitted = 0

            for frame, frame_idx, _ in frame_iter:
                if cancel.is_set() or not self._running:
                    break

                # Convert mx.array -> numpy -> PIL -> PNG bytes
                frame_np = np.array(frame)
                if frame_np.ndim != 3 or frame_np.shape[2] != 3:
                    continue

                # mx arrays may be float16; clip and convert to uint8
                frame_uint8 = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
                pil = PILImage.fromarray(frame_uint8)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                frame_bytes = buf.getvalue()

                frame_data = {
                    "index": frame_idx,
                    "frame": frame_bytes,
                    "width": width,
                    "height": height,
                    "is_final": False,
                    "method": "native_mlx_stream",
                }

                # Emit the previously buffered frame (if any)
                if pending_data is not None:
                    try:
                        thread_queue.put(pending_data, timeout=5.0)
                        frames_emitted += 1
                    except _queue_mod.Full:
                        logger.warning("Stream queue full — consumer likely gone")
                        pending_data = None
                        break

                # Buffer current frame (will be emitted next iteration or
                # marked as final after the loop)
                pending_data = frame_data

            # Mark the last buffered frame as final and emit it
            if pending_data is not None:
                pending_data["is_final"] = True
                try:
                    thread_queue.put(pending_data, timeout=5.0)
                    frames_emitted += 1
                except _queue_mod.Full:
                    logger.warning("Stream queue full — consumer likely gone")

            return frames_emitted

        finally:
            if _local_tc is not None and hasattr(self._native_pipeline, '_teacache_hook'):
                self._native_pipeline._teacache_hook = None

    def _stream_mlx_video(
        self,
        prompt: str,
        negative_prompt: str,
        image: bytes | None,
        width: int,
        height: int,
        num_frames: int,
        num_steps: int,
        guide_scale: float,
        seed: int,
        scheduler: str,
        model_dir: str,
        fps: int,
        thread_queue: queue.Queue[dict | None],
        cancel: threading.Event,
    ) -> int:
        """Generate video via mlx-video, then stream decoded frames via ffmpeg pipe.

        Returns the number of frames emitted. Puts frame dicts into thread_queue.
        """
        import queue as _queue_mod
        import subprocess

        import numpy as np
        from PIL import Image as PILImage

        # Save image for I2V
        image_path = None
        if image is not None:
            import tempfile
            fd, image_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            with open(image_path, "wb") as f:
                f.write(image)

        output_path = None
        proc = None  # Initialize before try so except/finally can clean up
        try:
            output_path = self._generate_with_mlx_video(
                model_dir=model_dir,
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image_path=image_path,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=num_steps,
                guide_scale=guide_scale,
                seed=seed,
                scheduler=scheduler,
            )

            if not output_path or not os.path.exists(output_path):
                # Fallback to placeholder streaming
                return self._stream_fallback(
                    prompt, width, height, num_frames,
                    thread_queue, cancel,
                )

            # Stream-decode the MP4 via ffmpeg rawvideo pipe
            ffmpeg_cmd = [
                "ffmpeg", "-i", output_path,
                "-vf", f"scale={width}:{height}",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-v", "quiet", "-",
            ]
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            frame_size = width * height * 3
            frame_idx = 0
            pending_data: dict | None = None
            frames_emitted = 0

            while True:
                if cancel.is_set():
                    proc.terminate()
                    break
                # select-based timeout (5s) to prevent hangs on corrupted
                # video data — cancel.is_set() only helps on the next
                # iteration, but a blocked read() never reaches it.
                import select as _select_mod
                ready, _, _ = _select_mod.select([proc.stdout], [], [], 5.0)
                if not ready:
                    logger.warning("ffmpeg read timeout after 5s in _stream_mlx_video")
                    proc.terminate()
                    break
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break

                arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (height, width, 3)
                )
                pil = PILImage.fromarray(arr)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                frame_bytes = buf.getvalue()

                frame_data = {
                    "index": frame_idx,
                    "frame": frame_bytes,
                    "width": width,
                    "height": height,
                    "is_final": False,
                    "method": f"mlx_video_{self._model_type}_stream",
                }

                # Emit previously buffered frame
                if pending_data is not None:
                    try:
                        thread_queue.put(pending_data, timeout=5.0)
                        frames_emitted += 1
                    except _queue_mod.Full:
                        logger.warning("Stream queue full — consumer likely gone")
                        pending_data = None
                        proc.terminate()
                        break

                pending_data = frame_data
                frame_idx += 1

            # Mark last frame as final and emit
            if pending_data is not None:
                pending_data["is_final"] = True
                try:
                    thread_queue.put(pending_data, timeout=5.0)
                    frames_emitted += 1
                except _queue_mod.Full:
                    logger.warning("Stream queue full — consumer likely gone")

            proc.stdout.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        except Exception as e:
            # Clean up subprocess if it was started
            if proc is not None:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=5)
                except Exception:
                    with contextlib.suppress(Exception):
                        proc.kill()
                with contextlib.suppress(Exception):
                    proc.stdout.close()
            logger.error(f"mlx-video stream failed: {e}", exc_info=True)
            return self._stream_fallback(
                prompt, width, height, num_frames,
                thread_queue, cancel,
            )
        finally:
            if image_path and os.path.exists(image_path):
                with contextlib.suppress(OSError):
                    os.unlink(image_path)
            if output_path and os.path.exists(output_path):
                with contextlib.suppress(OSError):
                    os.unlink(output_path)

        return frames_emitted

    def _stream_fallback(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        thread_queue: queue.Queue[dict | None],
        cancel: threading.Event,
    ) -> int:
        """Stream placeholder frames one by one (no model available).

        Returns the number of frames emitted.
        """
        import queue as _queue_mod

        import numpy as np
        from PIL import Image as PILImage

        actual_frames = min(num_frames, 4)
        frames_emitted = 0
        for i in range(actual_frames):
            if cancel.is_set():
                break

            arr = np.full((height, width, 3), [30, 30, 50], dtype=np.uint8)
            text = f"Frame {i+1}/{num_frames}"
            for j, _ch in enumerate(text):
                x = 10 + j * 12
                if x < width - 10:
                    arr[10:25, x:x+10] = 200
            pil = PILImage.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            frame_bytes = buf.getvalue()

            frame_data = {
                "index": i,
                "frame": frame_bytes,
                "width": width,
                "height": height,
                "is_final": i == actual_frames - 1,
                "method": "fallback_stream",
            }
            try:
                thread_queue.put(frame_data, timeout=5.0)
                frames_emitted += 1
            except _queue_mod.Full:
                logger.warning("Stream queue full — consumer likely gone")
                break

        return frames_emitted

    def _run_generation(
        self,
        prompt: str,
        negative_prompt: str,
        image: bytes | None,
        width: int,
        height: int,
        num_frames: int,
        num_steps: int,
        guide_scale: float,
        fps: int,
        seed: int,
        scheduler: str,
        output_format: str,
    ) -> VideoGenOutput:
        """Synchronous generation — tries native pipeline, then mlx-video."""
        model_dir = self._get_model_dir()

        if not model_dir or not os.path.isdir(model_dir):
            logger.warning("No video model directory found, using fallback")
            return self._fallback_generation(
                prompt, width, height, num_frames, fps, output_format,
            )

        # Try native MLX pipeline first (YUNSHU_VIDEO_PIPELINE=native)
        use_native = os.environ.get("YUNSHU_VIDEO_PIPELINE", "").strip() == "native"
        if use_native or self._native_pipeline is not None:
            result = self._generate_with_native_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                width=width,
                height=height,
                num_frames=num_frames,
                num_steps=num_steps,
                guide_scale=guide_scale,
                seed=seed,
                scheduler=scheduler,
                model_dir=model_dir,
            )
            if result is not None:
                # result.frames are mx.array; normalize to PNG bytes so the
                # frames-format response (base64 each frame) works and MP4 encoding
                # doesn't silently drop them.
                frames = [
                    _f for _f in (self._frame_to_png_bytes(f) for f in (result.frames or []))
                    if _f is not None
                ]
                video_data = b""
                if output_format == "mp4" and frames:
                    video_data = self._encode_frames_to_mp4(frames, fps)
                # When MP4 encoding fails, report 0 frames to avoid
                # claiming video data that doesn't exist.
                if output_format == "mp4" and not video_data:
                    return VideoGenOutput(
                        video_data=b"",
                        frames=[],
                        width=width,
                        height=height,
                        num_frames=0,
                        fps=fps,
                        # Propagate the pipeline's method (it may carry a
                        # ..._placeholder_fallback tag) instead of hardcoding
                        # "native_mlx", which discarded the fallback signal and let the
                        # router's `"fallback" in method` 503 guard pass → 200 garbage.
                        method=(getattr(result, "method", None) or "native_mlx"),
                        metadata={"error": "MP4 encoding failed"},
                    )
                return VideoGenOutput(
                    video_data=video_data if output_format == "mp4" else b"",
                    frames=frames,
                    width=width,
                    height=height,
                    num_frames=len(frames) or num_frames,
                    fps=fps,
                    # Propagate the pipeline's method tag (router-level
                    # safety net for the placeholder-fallback case).
                    method=(getattr(result, "method", None) or "native_mlx"),
                )

        # Save image to temp file for I2V if provided
        image_path = None
        if image is not None:
            import tempfile
            fd, image_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            with open(image_path, "wb") as f:
                f.write(image)

        output_path = None
        try:
            output_path = self._generate_with_mlx_video(
                model_dir=model_dir,
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image_path=image_path,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=num_steps,
                guide_scale=guide_scale,
                seed=seed,
                scheduler=scheduler,
            )

            if output_path and os.path.exists(output_path):
                # mlx_video.generate_video has NO fps parameter — it
                # always saves at the model's config.sample_fps (e.g. 24 for Wan2.2),
                # ignoring req.fps. Reporting req.fps was a LIE: the file plays at the
                # baked-in rate, so a client computing duration=num_frames/reported_fps got
                # it wrong. Probe the file's ACTUAL fps and report that (truthful duration).
                # (Honoring an arbitrary req.fps would require retiming all frames via
                # ffmpeg setpts — left as an enhancement.)
                _actual_fps = self._probe_mp4_fps(output_path) or fps
                video_data = Path(output_path).read_bytes()
                os.unlink(output_path)
                output_path = None  # Mark as cleaned up

                frames = []
                if output_format == "frames":
                    frames = self._extract_frames(video_data)

                return VideoGenOutput(
                    video_data=video_data if output_format == "mp4" else b"",
                    frames=frames,
                    width=width,
                    height=height,
                    num_frames=len(frames) if frames else num_frames,
                    fps=_actual_fps,
                    method=f"mlx_video_{self._model_type}",
                )
        except Exception as e:
            logger.error(f"mlx-video generation failed: {e}", exc_info=True)
        finally:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)

        return self._fallback_generation(
            prompt, width, height, num_frames, fps, output_format,
        )

    def _generate_with_mlx_video(
        self,
        model_dir: str,
        prompt: str,
        negative_prompt: str | None,
        image_path: str | None,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        guide_scale: float,
        seed: int,
        scheduler: str,
    ) -> str | None:
        """Run video generation via mlx-video library."""
        output_path = None
        try:
            import tempfile
            fd, output_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            if self._model_type == "wan_2_2":
                from mlx_video.models.wan_2 import generate as _wan_gen
                from mlx_video.models.wan_2.generate import generate_video
                # Memory optimization: the stock T5 loader upcasts to fp32
                # (~22GB for umt5-xxl). T5 only encodes the prompt once, so bf16
                # is plenty and frees ~11GB — the headroom that lets near-native
                # resolutions (which dramatically cut off-distribution noise,
                # VLM-verified) fit on a 36GB Mac. Patch once, idempotently.
                if not getattr(_wan_gen, "_yunshu_t5_bf16", False):
                    import mlx.core as _mx
                    from mlx_video.models.wan_2.text_encoder import T5Encoder

                    def _load_t5_bf16(model_path, config):
                        enc = T5Encoder(
                            vocab_size=config.t5_vocab_size, dim=config.t5_dim,
                            dim_attn=config.t5_dim_attn, dim_ffn=config.t5_dim_ffn,
                            num_heads=config.t5_num_heads, num_layers=config.t5_num_layers,
                            num_buckets=config.t5_num_buckets, shared_pos=False,
                        )
                        w = _mx.load(str(model_path))
                        w = {k: v.astype(_mx.bfloat16) for k, v in w.items()}
                        enc.load_weights(list(w.items()))
                        _mx.eval(enc.parameters())
                        return enc

                    _wan_gen.load_t5_encoder = _load_t5_bf16
                    _wan_gen._yunshu_t5_bf16 = True
                generate_video(
                    model_dir=model_dir,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image_path,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    steps=steps,
                    guide_scale=guide_scale,
                    seed=seed,
                    output_path=output_path,
                    scheduler=scheduler,
                )
            elif self._model_type == "ltx_2":
                import os as _os

                from mlx_video.models.ltx_2.generate import generate_video
                # The real LTX-2 generate_video requires model_repo +
                # text_encoder_repo and uses num_inference_steps / cfg_scale. The old
                # call passed model_dir= / steps= → TypeError on EVERY LTX request →
                # silent degrade to a 503 (LTX advertised but non-functional). Map to
                # the real signature, forward the init image for I2V, and resolve the
                # SEPARATE text-encoder repo from an env override (else assume the
                # model repo bundles it).
                _te_repo = _os.environ.get("YUNSHU_LTX_TEXT_ENCODER_REPO") or model_dir
                generate_video(
                    model_repo=model_dir,
                    text_encoder_repo=_te_repo,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    cfg_scale=guide_scale,
                    seed=seed,
                    output_path=output_path,
                    image=image_path,
                )
            else:
                # Dynamic model import: try mlx_video.models.{type}.generate
                try:
                    module_name = self._model_type
                    module_path = f"mlx_video.models.{module_name}.generate"
                    gen_module = __import__(module_path, fromlist=["generate_video"])
                    gen_fn = gen_module.generate_video
                    gen_fn(
                        model_dir=model_dir,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image_path,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        steps=steps,
                        guide_scale=guide_scale,
                        seed=seed,
                        output_path=output_path,
                    )
                except (ImportError, AttributeError) as e:
                    logger.error(f"Unsupported model type {self._model_type}: {e}")
                    # Clean up temp file on early return
                    if output_path and os.path.exists(output_path):
                        with contextlib.suppress(OSError):
                            os.unlink(output_path)
                    return None

            return output_path

        except ImportError:
            logger.error("mlx-video not installed. Install with: pip install mlx-video")
            # Clean up temp file on error
            if output_path and os.path.exists(output_path):
                with contextlib.suppress(OSError):
                    os.unlink(output_path)
            return None
        except Exception:
            # Clean up temp file on any error — caller won't get the path
            if output_path and os.path.exists(output_path):
                with contextlib.suppress(OSError):
                    os.unlink(output_path)
            raise

    def _generate_with_native_pipeline(
        self,
        prompt: str,
        negative_prompt: str,
        image: bytes | None,
        width: int,
        height: int,
        num_frames: int,
        num_steps: int,
        guide_scale: float,
        seed: int,
        scheduler: str,
        model_dir: str,
    ):
        """Generate video using native MLX pipeline (WanVideoPipeline + TeaCache)."""
        try:
            from .video_pipeline import VideoGenRequest, WanVideoPipeline

            if self._native_pipeline is None:
                self._native_pipeline = WanVideoPipeline(model_path=model_dir)
                loaded = self._native_pipeline.load_weights(model_dir)
                if not loaded:
                    logger.warning("Native pipeline weight loading failed")
                    if hasattr(self._native_pipeline, 'cleanup'):
                        self._native_pipeline.cleanup()
                    del self._native_pipeline
                    self._native_pipeline = None
                    return None
            # load_weights returning True only means the weights DICT
            # loaded — the transformer is "real" only when _model is CALLABLE. A
            # non-callable model makes generate_frames run the placeholder velocity
            # and return garbage frames tagged ..._placeholder_fallback. Refuse it
            # here (mirror the streaming guard) so the non-streaming caller
            # falls through to the honest fallback → router 503, instead of returning
            # placeholder frames as a 200. Re-checked on EVERY call (not just on
            # creation) so a pipeline left set by a prior call can't be reused to
            # emit garbage. Also null it so a later streaming request re-evaluates.
            if not callable(getattr(self._native_pipeline, "_model", None)):
                logger.warning("Native pipeline has no callable transformer; refusing "
                               "placeholder generation (honest fallback)")
                if hasattr(self._native_pipeline, 'cleanup'):
                    self._native_pipeline.cleanup()
                self._native_pipeline = None
                return None

            request = VideoGenRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_steps=num_steps,
                guide_scale=guide_scale,
                seed=seed,  # already randomized by generate() if originally < 0
                scheduler=scheduler,
            )

            # Wire TeaCache into the pipeline's denoising loop.
            # Create a per-request instance to avoid state leakage.
            _local_tc = None
            if self._teacache_config is not None:
                from .teacache import TeaCacheHook
                _local_tc = TeaCacheHook(self._teacache_config)
                _local_tc.reset()
                self._native_pipeline._teacache_hook = _local_tc

            try:
                if image is not None:
                    # Load image bytes directly as an MLX array for the
                    # native pipeline — avoids unnecessary disk write.
                    import mlx.core as mx
                    import numpy as np
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(io.BytesIO(image)).convert("RGB")
                    img_np = np.array(pil_img, dtype=np.float32) / 255.0
                    img_mx = mx.array(img_np)
                    if not self._running:
                        return None
                    result = self._native_pipeline.generate_from_image(
                        request=request,
                        image=img_mx,
                    )
                else:
                    if not self._running:
                        return None
                    result = self._native_pipeline.generate_frames(request)

                if result and result.frames:
                    return result
                return None
            finally:
                if _local_tc is not None and hasattr(self._native_pipeline, '_teacache_hook'):
                    self._native_pipeline._teacache_hook = None
        except Exception as e:
            logger.error(f"Native pipeline generation failed: {e}", exc_info=True)
            return None

    @staticmethod
    def _frame_to_png_bytes(frame) -> bytes | None:
        """Normalize ANY frame (mx.array, np.ndarray, PIL image, or PNG bytes) to
        PNG bytes. The native pipeline yields mx.array frames; returning those raw
        made the /video frames-format response (base64-encode each frame) crash and
        the MP4 encoder silently drop every frame. Returns None if unconvertible."""
        import io as _io

        import numpy as _np
        from PIL import Image as _Image
        try:
            if isinstance(frame, (bytes, bytearray)):
                return bytes(frame)
            if hasattr(frame, "save"):  # PIL image
                _buf = _io.BytesIO()
                frame.save(_buf, format="PNG")
                return _buf.getvalue()
            arr = _np.array(frame)  # mx.array or np.ndarray
            if arr.ndim != 3 or arr.shape[2] != 3:
                return None
            # Float frames (mx pipelines emit [0,1] or [-1,1]); ints pass through.
            if _np.issubdtype(arr.dtype, _np.floating):
                if float(arr.min()) < 0.0:
                    arr = (arr + 1.0) / 2.0
                arr = _np.clip(arr * 255.0, 0, 255)
            arr = arr.astype(_np.uint8)
            _buf = _io.BytesIO()
            _Image.fromarray(arr).save(_buf, format="PNG")
            return _buf.getvalue()
        except Exception:
            logger.debug("frame->png conversion failed", exc_info=True)
            return None

    def _encode_frames_to_mp4(self, frames: list, fps: int) -> bytes:
        """Encode a list of frames (np arrays, PIL images, or PNG bytes) to MP4 bytes."""
        import io as _io
        import shutil
        import subprocess
        import tempfile

        tmp_path = None
        frame_dir = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            import numpy as np
            from PIL import Image

            frame_dir = tempfile.mkdtemp(prefix="yunshu_video_frames_")

            for i, frame in enumerate(frames):
                if isinstance(frame, np.ndarray):
                    img = Image.fromarray(frame)
                elif isinstance(frame, (bytes, bytearray)):
                    img = Image.open(_io.BytesIO(bytes(frame)))
                elif hasattr(frame, 'save'):
                    img = frame
                else:
                    # mx.array or other — normalize via the shared helper.
                    _png = self._frame_to_png_bytes(frame)
                    if _png is None:
                        continue
                    img = Image.open(_io.BytesIO(_png))
                img.save(os.path.join(frame_dir, f"{i:04d}.png"))

            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", os.path.join(frame_dir, "%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                tmp_path,
            ], capture_output=True, check=True)

            return Path(tmp_path).read_bytes()

        except Exception as e:
            logger.error(f"MP4 encoding failed: {e}", exc_info=True)
            return b""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                with contextlib.suppress(Exception):
                    os.unlink(tmp_path)
            if frame_dir and os.path.isdir(frame_dir):
                with contextlib.suppress(Exception):
                    shutil.rmtree(frame_dir)

    def _fallback_generation(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        output_format: str,
    ) -> VideoGenOutput:
        """Fallback: generate placeholder frames when no model is available."""
        import numpy as np

        frames = []
        for i in range(min(num_frames, 4)):  # Generate just 4 placeholder frames
            arr = np.full((height, width, 3), [30, 30, 50], dtype=np.uint8)
            # Add frame number as text
            text = f"Frame {i+1}/{num_frames}"
            for j, _ch in enumerate(text):
                x = 10 + j * 12
                if x < width - 10:
                    arr[10:25, x:x+10] = 200
            from PIL import Image as PILImage
            pil = PILImage.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            frames.append(buf.getvalue())

        return VideoGenOutput(
            frames=frames,
            width=width,
            height=height,
            num_frames=len(frames),
            fps=fps,
            method="fallback",
            metadata={"prompt": prompt, "note": "Placeholder frames, no video model loaded"},
        )

    def _probe_mp4_fps(self, path: str) -> float | None:
        """Return the actual frame rate of an MP4 via ffprobe.

        mlx_video bakes the model's sample_fps into the file regardless of the requested
        fps; probing reports the truth so the client's duration math is right. Returns
        None if ffprobe is unavailable or parsing fails (caller falls back).
        """
        import shutil
        import subprocess
        if shutil.which("ffprobe") is None:
            return None
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate", "-of",
                 "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=15,
            )
            val = (out.stdout or "").strip()
            if "/" in val:
                num, den = val.split("/", 1)
                den_f = float(den)
                return float(num) / den_f if den_f else None
            return float(val) if val else None
        except Exception:
            return None

    def _extract_frames(self, video_data: bytes) -> list[bytes]:
        """Extract individual frames from MP4 as PNG bytes."""
        import shutil
        import subprocess
        import tempfile

        tmp_path = None
        frame_dir = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(video_data)

            frame_dir = tempfile.mkdtemp()
            subprocess.run(
                ["ffmpeg", "-i", tmp_path, "-f", "image2", f"{frame_dir}/frame_%04d.png", "-y"],
                capture_output=True, check=True,
            )

            frames = []
            for f in sorted(os.listdir(frame_dir)):
                if f.endswith(".png"):
                    frames.append(Path(os.path.join(frame_dir, f)).read_bytes())
            return frames

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}", exc_info=True)
            return []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                with contextlib.suppress(Exception):
                    os.unlink(tmp_path)
            if frame_dir and os.path.isdir(frame_dir):
                with contextlib.suppress(Exception):
                    shutil.rmtree(frame_dir)

    def get_stats(self) -> dict:
        with self._stats_lock:
            frames_processed = self._stats.frames_processed
            frames_streamed = self._stats.frames_streamed
            batches_processed = self._stats.batches_processed
            total_generate_calls = self._stats.total_generate_calls
            total_stream_calls = self._stats.total_stream_calls
            avg_fps = round(self._stats.avg_fps, 2)
            avg_decode_fps = round(self._stats.avg_decode_fps, 2)
            total_generate_ms = round(self._stats.total_generate_ms, 1)
            total_frame_decode_ms = round(self._stats.total_frame_decode_ms, 1)
            lora_adapter_id = self._stats.lora_adapter_id
        return {
            "model": self._model_path,
            "model_type": self._model_type,
            "loaded": self.is_loaded,
            "model_loaded": self._model is not None or self._native_pipeline is not None,
            "running": self._running,
            "frames_processed": frames_processed,
            "frames_streamed": frames_streamed,
            "batches_processed": batches_processed,
            "total_generate_calls": total_generate_calls,
            "total_stream_calls": total_stream_calls,
            "avg_fps": avg_fps,
            "avg_decode_fps": avg_decode_fps,
            "total_generate_ms": total_generate_ms,
            "total_frame_decode_ms": total_frame_decode_ms,
            "lora_loaded": self._lora_loaded,
            "lora_merged": self._lora_merged,
            "lora_adapter_id": lora_adapter_id,
            "teacache_enabled": self._teacache_config is not None,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Streaming video frames — yield processed frames as decoded
    # ═══════════════════════════════════════════════════════════════════════

    @tracks_active_gen
    async def stream_frames(
        self,
        video_data: bytes,
        frame_interval: int = 1,
        max_frames: int = 0,
        width: int | None = None,
        height: int | None = None,
        output_format: str = "png",
    ) -> AsyncIterator[dict]:
        """Stream processed frames from video data as they are decoded.

        Reads video bytes (MP4), decodes frames incrementally, and yields
        each processed frame as a dict with metadata. Designed for real-time
        video analysis pipelines where you don't need to wait for the full
        decode to finish before starting to process frames.

        Args:
            video_data: MP4 video bytes to decode.
            frame_interval: Emit every N-th frame (default: 1 = every frame).
            max_frames: Maximum number of frames to emit (0 = unlimited).
            width: Resize width (None = original).
            height: Resize height (None = original).
            output_format: "png" or "numpy" for each frame.

        Yields:
            dict with keys:
              - "index": Frame index (0-based)
              - "frame": Frame bytes (PNG) or numpy array
              - "width": Frame width
              - "height": Frame height
              - "timestamp_ms": Decode timestamp offset from start (ms)
              - "is_final": True for the last frame
        """
        if not self._running:
            self.start()

        with self._stats_lock:
            self._stats.total_stream_calls += 1

        # Use a thread-safe queue for the executor → async bridge.
        # asyncio.Queue.put_nowait() from a non-event-loop thread is unsafe.
        import queue as _queue_mod
        _thread_queue: _queue_mod.Queue[dict | None] = _queue_mod.Queue(maxsize=128)
        _consumer_cancel = threading.Event()

        # Check env var for streaming mode
        os.environ.get("YUNSHU_VIDEO_STREAMING", "0").strip() in ("1", "true", "yes")

        def _decode_sync():
            """Synchronous frame decoder running in the MLX executor."""
            try:
                t_decode_start = time.monotonic()
                frames_decoded = 0
                frames_emitted = 0

                # Write video data to temp file for ffmpeg
                import tempfile
                fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                os.close(fd)
                with open(tmp_path, "wb") as f:
                    f.write(video_data)

                try:
                    # Use ffmpeg to probe video info and decode frames
                    import subprocess

                    import numpy as np

                    proc = None  # Initialize before try to prevent NameError in except

                    # Probe video info
                    probe = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-print_format", "json",
                         "-show_streams", "-select_streams", "v:0", tmp_path],
                        capture_output=True, text=True,
                    )

                    video_w, video_h = 0, 0
                    try:
                        import json
                        probe_data = json.loads(probe.stdout)
                        stream = probe_data.get("streams", [{}])[0]
                        video_w = int(stream.get("width", 0))
                        video_h = int(stream.get("height", 0))
                        # nb_frames may be N/A
                        nb = stream.get("nb_frames", "0")
                        int(nb) if nb.isdigit() else 0
                    except (json.JSONDecodeError, ValueError, IndexError):
                        pass

                    target_w = width or video_w or 64
                    target_h = height or video_h or 64

                    # Build ffmpeg filter graph:
                    # - frame_interval=1 → emit every frame (no fps filter)
                    # - frame_interval>1 → select every Nth frame + scale
                    if frame_interval <= 1:
                        vf_filter = f"scale={target_w}:{target_h}"
                    else:
                        # select='not(mod(n,N))' picks frames 0, N, 2N, ...
                        vf_filter = f"select='not(mod(n\\,{frame_interval}))',scale={target_w}:{target_h}"

                    # Decode frames via ffmpeg pipe (streaming, no temp PNG files)
                    ffmpeg_cmd = [
                        "ffmpeg", "-i", tmp_path,
                        "-vf", vf_filter,
                        "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-v", "quiet", "-"
                    ]

                    proc = subprocess.Popen(
                        ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )

                    frame_size = target_w * target_h * 3
                    frame_idx = 0
                    _queue_full = False

                    while True:
                        # select-based timeout (5s) to prevent hangs on
                        # corrupted video data that never produces bytes.
                        import select as _select_mod
                        ready, _, _ = _select_mod.select([proc.stdout], [], [], 5.0)
                        if not ready:
                            logger.warning("ffmpeg read timeout after 5s in stream_frames")
                            break
                        raw = proc.stdout.read(frame_size)
                        if len(raw) < frame_size:
                            break

                        elapsed_decode = (time.monotonic() - t_decode_start) * 1000.0
                        frames_decoded += 1

                        # Check if consumer has gone away
                        if _consumer_cancel.is_set():
                            proc.terminate()
                            break

                        # Enforce max_frames limit
                        if max_frames > 0 and frames_emitted >= max_frames:
                            proc.terminate()
                            break

                        # If queue was previously full (consumer gone), stop decoding
                        if _queue_full:
                            proc.terminate()
                            break

                        frame_output: Any
                        if output_format == "numpy":
                            frame_output = np.frombuffer(raw, dtype=np.uint8).reshape(
                                (target_h, target_w, 3)
                            ).copy()
                        else:
                            # Convert to PNG
                            from PIL import Image as PILImage
                            arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                                (target_h, target_w, 3)
                            )
                            pil = PILImage.fromarray(arr)
                            buf = io.BytesIO()
                            pil.save(buf, format="PNG")
                            frame_output = buf.getvalue()

                        is_final = False
                        if max_frames > 0 and frames_emitted + 1 >= max_frames:
                            is_final = True

                        frame_data = {
                            "index": frame_idx,
                            "frame": frame_output,
                            "width": target_w,
                            "height": target_h,
                            "timestamp_ms": round(elapsed_decode, 1),
                            "is_final": is_final,
                        }
                        try:
                            _thread_queue.put_nowait(frame_data)
                        except _queue_mod.Full:
                            logger.warning("Video stream queue full — consumer likely gone, stopping decode")
                            _queue_full = True
                            proc.terminate()
                            break

                        frames_emitted += 1
                        frame_idx += frame_interval

                    proc.stdout.close()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)

                except Exception:
                    if proc is not None and proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                    try:
                        if proc is not None:
                            proc.stdout.close()
                    except Exception:
                        pass
                    raise
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                # Update stats
                decode_total_ms = (time.monotonic() - t_decode_start) * 1000.0
                with self._stats_lock:
                    self._stats.frames_streamed += frames_emitted
                    self._stats.total_frame_decode_ms += decode_total_ms

                # Signal final frame
                with contextlib.suppress(_queue_mod.Full):
                    _thread_queue.put_nowait(None)

            except Exception as e:
                logger.error(f"Frame streaming error: {e}", exc_info=True)
                with contextlib.suppress(_queue_mod.Full):
                    _thread_queue.put_nowait(None)

        # Run decoder in executor
        loop = asyncio.get_running_loop()
        decode_task = loop.run_in_executor(self._executor, _decode_sync)

        # Yield frames as they arrive, bridging from thread-safe queue to async
        try:
            while True:
                # Poll the thread-safe queue without blocking the event loop
                try:
                    chunk = _thread_queue.get_nowait()
                except _queue_mod.Empty:
                    await asyncio.sleep(0.01)  # Brief yield to event loop
                    continue
                if chunk is None:
                    break
                yield chunk
                if chunk.get("is_final"):
                    break
        finally:
            _consumer_cancel.set()
            if not decode_task.done():
                decode_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await decode_task
            # Drain remaining queue items to unblock the executor thread
            while True:
                try:
                    _thread_queue.get_nowait()
                except _queue_mod.Empty:
                    break

    def process_frame_batch_sync(self, batch: FrameBatch) -> list[dict]:
        """Process a batch of frames synchronously on GPU.

        Placeholder for actual model-based processing. When a video model
        is loaded, this runs inference on the batched frames. Without a model,
        returns frame metadata.

        Args:
            batch: FrameBatch with frame PNG bytes.

        Returns:
            List of dicts, one per frame, with processing results.
        """
        results = []
        for idx, frame_bytes in zip(batch.frame_indices, batch.frames, strict=False):
            results.append({
                "frame_index": idx,
                "width": batch.width,
                "height": batch.height,
                "frame_size_bytes": len(frame_bytes),
                "processed": self._model is not None,
            })
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # LoRA adapter support for video models
    # ═══════════════════════════════════════════════════════════════════════

    def load_lora_adapter(
        self,
        adapter_path: str,
        rank: int = 8,
        scale: float = 20.0,
    ) -> bool:
        """Load a LoRA adapter for the video model.

        Applies LoRA layers to attention Q/V projections in the video model
        (Wan2.2 or LTX2 transformer). Saves base model weights for later
        unloading. Follows the same pattern as image_engine.load_lora_adapter.

        Args:
            adapter_path: Path to directory containing adapter_config.json
                          and adapters.safetensors.
            rank: LoRA rank (default: 8, overridden by adapter config).
            scale: LoRA scale (default: 20.0, overridden by adapter config).

        Returns:
            True if adapter loaded successfully.
        """
        adapter_dir = Path(adapter_path)
        config_path = adapter_dir / "adapter_config.json"

        if not config_path.exists():
            logger.error(f"No adapter_config.json in {adapter_path}")
            return False

        with self._lora_lock:
            if self._lora_loaded:
                logger.warning("LoRA adapter already loaded, unload first")
                return False

            import json

            with open(config_path) as f:
                config = json.load(f)

            lora_params = config.get("lora_parameters", {})
            self._lora_rank = lora_params.get("rank", rank)
            self._lora_scale = lora_params.get("scale", scale)
            num_layers = config.get("num_layers", 16)

            # Save base model weights for restoration on unload
            if self._model is not None and self._base_model_weights is None:
                try:
                    import mlx.core as mx
                    from mlx.utils import tree_map
                    self._base_model_weights = tree_map(
                        lambda x: mx.array(x), self._model.parameters()
                    )
                except Exception as e:
                    logger.warning(f"Could not save base model weights: {e}")

            # If model not yet loaded, just record the adapter path for lazy loading.
            if self._model is None:
                self._lora_adapter_path = str(adapter_dir)
                logger.info(f"LoRA adapter queued for lazy loading: {adapter_path}")
                return True

            # Apply LoRA to loaded model
            try:
                self._apply_lora_to_model(num_layers)

                # Load adapter weights
                weights_path = adapter_dir / "adapters.safetensors"
                if weights_path.exists():
                    self._model.load_weights(str(weights_path), strict=False)

                self._lora_adapter_path = str(adapter_dir)
                self._lora_loaded = True
                with self._stats_lock:
                    self._stats.lora_adapter_id = adapter_dir.name
                    self._stats.lora_loaded = True
                logger.info(
                    f"Video LoRA adapter loaded: {adapter_path} "
                    f"(rank={self._lora_rank}, scale={self._lora_scale})"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to load video LoRA adapter: {e}", exc_info=True)
                self._lora_loaded = False
                return False

    def unload_lora_adapter(self) -> bool:
        """Unload the current LoRA adapter, restoring base model weights.

        Returns:
            True if adapter was unloaded successfully.
        """
        with self._lora_lock:
            if not self._lora_loaded:
                return False

            if self._lora_merged:
                logger.warning("Cannot unload merged LoRA adapter (weights are fused)")
                return False

            try:
                if self._model is not None and self._base_model_weights is not None:
                    import mlx.core as mx
                    self._model.update(self._base_model_weights)
                    mx.eval(self._model.parameters())
                    self._base_model_weights = None

                self._lora_loaded = False
                self._lora_adapter_path = ""
                with self._stats_lock:
                    self._stats.lora_loaded = False
                    self._stats.lora_adapter_id = ""
                logger.info("Video LoRA adapter unloaded, base weights restored")
                return True
            except Exception as e:
                logger.error(f"Failed to unload video LoRA adapter: {e}", exc_info=True)
            return False

    def merge_lora_adapter(self) -> bool:
        """Merge LoRA weights permanently into the video model.

        After merging, the adapter cannot be unloaded individually.
        The merged model has zero LoRA inference overhead.
        Follows the same pattern as LoRAAdapterManager.merge_adapter().

        Thread safety: _lora_lock (RLock) is held throughout the entire
        merge operation to prevent concurrent unload_lora_adapter from
        racing between the guard check and the actual fuse work.
        """
        with self._lora_lock:
            if not self._lora_loaded:
                logger.error("No LoRA adapter loaded to merge")
                return False

            if self._lora_merged:
                logger.warning("LoRA adapter already merged")
                return False

            try:
                from mlx.utils import tree_unflatten
                from mlx_lm.tuner.lora import LoRALinear

                merged_layers = []
                for name, module in self._model.named_modules():
                    if isinstance(module, LoRALinear):
                        merged_layers.append((name, module.fuse(dequantize=False)))

                if merged_layers:
                    self._model.update_modules(tree_unflatten(merged_layers))

                self._lora_merged = True
                self._lora_loaded = False  # Structure unwrapped, no longer "loaded"
                self._base_model_weights = None  # Can no longer restore
                with self._stats_lock:
                    self._stats.lora_merged = True
                    self._stats.lora_loaded = False
                logger.info("Video LoRA adapter merged into base model")
                return True
            except Exception as e:
                logger.error(f"Failed to merge video LoRA adapter: {e}", exc_info=True)
                return False

    def list_lora_status(self) -> dict:
        """Return current LoRA adapter status."""
        return {
            "loaded": self._lora_loaded,
            "merged": self._lora_merged,
            "adapter_path": self._lora_adapter_path,
            "rank": self._lora_rank,
            "scale": self._lora_scale,
        }

    def _apply_lora_to_model(self, num_layers: int) -> None:
        """Apply LoRA layers to the video model's attention projections.

        Targets Q and V projection layers in the video transformer,
        following the same pattern as image_engine.load_lora_adapter().
        """
        import mlx.nn as nn
        from mlx_lm.tuner.lora import LoRALinear

        applied_layers: set[str] = set()
        for name, module in self._model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if len(applied_layers) >= num_layers:
                break
            # Apply LoRA to attention Q/V projections
            if any(k in name for k in ("q_proj", "v_proj", "query", "value", "to_q", "to_v")):
                # MLX's nn.Linear has NO PyTorch in_features/out_features (it
                # exposes .weight of shape [out,in]); LoRALinear(module.in_features, ...)
                # raised AttributeError on every module. AND named_modules() yields list-
                # indexed names like "blocks.0.attn.to_q" whose mid-path "0" is a LIST INDEX,
                # so getattr(parent, "0") raised AttributeError → the whole video LoRA apply
                # crashed. Identical to the image_engine bug. Use from_base + an
                # index-aware parent walk.
                lora_layer = LoRALinear.from_base(module, r=self._lora_rank, scale=self._lora_scale)
                # Set on parent module
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = self._model
                    for part in parts[0].split("."):
                        try:
                            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
                        except (AttributeError, IndexError, KeyError, TypeError):
                            parent = None
                            break
                    if parent is not None:
                        setattr(parent, parts[1], lora_layer)
                # Track by transformer block index to count layers, not projections
                block_key = name.rsplit(".", 2)[0] if "." in name else name
                applied_layers.add(block_key)

        logger.info(f"Applied LoRA to {len(applied_layers)} layers in video model")
