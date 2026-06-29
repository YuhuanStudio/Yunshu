"""Tests for native MLX video pipeline.

Covers:
  - VideoGenRequest defaults and custom values
  - VideoGenResult dataclass
  - FlowMatchingScheduler: timesteps, noise init, guidance, step
  - EulerScheduler: step and guidance
  - TemporalConv3D forward pass
  - VideoVAEDecoder: init and decode
  - WanVideoPipeline: load, unload, generate, stream
  - WanVideoPipeline: generate_from_image (I2V)
  - WanVideoPipeline: error handling (not loaded)
  - VideoLoRAManager: apply, unload, merge, status
  - VideoLoRAManager: memory budget enforcement
  - VideoLoRAManager: merged state cannot unload
  - Edge cases: empty request, invalid path, large dimensions
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import mlx.core as mx

from yunshu_engine.video_pipeline import (
    EulerScheduler,
    FlowMatchingScheduler,
    TemporalConv3D,
    VideoGenRequest,
    VideoGenResult,
    VideoLoRAManager,
    VideoVAEDecoder,
    WanVideoPipeline,
)

# ── VideoGenRequest Tests ──


class TestVideoGenRequest:
    def test_defaults(self):
        req = VideoGenRequest()
        assert req.prompt == ""
        assert req.negative_prompt == ""
        assert req.image is None
        assert req.num_frames == 81
        assert req.height == 704
        assert req.width == 1280
        assert req.fps == 16
        assert req.num_steps == 20
        assert req.guide_scale == 5.0
        assert req.seed == -1
        assert req.scheduler == "flow_matching"

    def test_custom_values(self):
        req = VideoGenRequest(
            prompt="a cat playing piano",
            num_frames=33,
            width=640,
            height=480,
            num_steps=10,
            guide_scale=7.5,
            seed=42,
            scheduler="euler",
        )
        assert req.prompt == "a cat playing piano"
        assert req.num_frames == 33
        assert req.seed == 42
        assert req.scheduler == "euler"


# ── VideoGenResult Tests ──


class TestVideoGenResult:
    def test_defaults(self):
        result = VideoGenResult()
        assert result.frames == []
        assert result.latents is None
        assert result.width == 0
        assert result.height == 0
        assert result.num_frames == 0
        assert result.fps == 16
        assert result.method == ""


# ── FlowMatchingScheduler Tests ──


class TestFlowMatchingScheduler:
    def test_init(self):
        sched = FlowMatchingScheduler(num_steps=20, guide_scale=5.0)
        assert sched.num_steps == 20
        assert sched.guide_scale == 5.0

    def test_timesteps(self):
        sched = FlowMatchingScheduler(num_steps=10)
        ts = sched.timesteps
        assert len(ts) == 11
        assert abs(ts[0] - 1.0) < 1e-6
        assert abs(ts[-1] - 0.0) < 1e-6

    def test_get_timestep(self):
        sched = FlowMatchingScheduler(num_steps=10)
        assert abs(sched.get_timestep(0) - 1.0) < 1e-6
        assert abs(sched.get_timestep(5) - 0.5) < 1e-6
        assert abs(sched.get_timestep(10) - 0.0) < 1e-6

    def test_get_timestep_out_of_range(self):
        sched = FlowMatchingScheduler(num_steps=5)
        assert sched.get_timestep(999) == 0.0

    def test_get_dt(self):
        sched = FlowMatchingScheduler(num_steps=20)
        assert abs(sched.get_dt(0) - 0.05) < 1e-6

    def test_init_noise_shape(self):
        sched = FlowMatchingScheduler()
        noise = sched.init_noise(shape=(1, 16, 5, 88, 160), seed=42, dtype=mx.float16)
        assert noise.shape == (1, 16, 5, 88, 160)
        assert noise.dtype == mx.float16

    def test_init_noise_random_seed(self):
        sched = FlowMatchingScheduler()
        n1 = sched.init_noise(shape=(1, 4, 4, 4), seed=123)
        n2 = sched.init_noise(shape=(1, 4, 4, 4), seed=123)
        assert mx.allclose(n1, n2).item()

    def test_apply_guidance(self):
        sched = FlowMatchingScheduler(guide_scale=5.0)
        cond = mx.ones((1, 16, 4, 4)) * 2.0
        uncond = mx.ones((1, 16, 4, 4))
        guided = sched.apply_guidance(cond, uncond)
        expected = 1.0 + 5.0 * (2.0 - 1.0)  # 6.0
        assert mx.allclose(guided, mx.full(guided.shape, expected)).item()

    def test_step(self):
        sched = FlowMatchingScheduler(num_steps=10)
        latent = mx.ones((1, 16, 4, 4))
        noise_pred = mx.ones((1, 16, 4, 4)) * 0.5
        result = sched.step(noise_pred, latent, t=0.5, dt=0.1)
        expected = 1.0 + 0.5 * 0.1  # 1.05
        assert mx.allclose(result, mx.full(result.shape, expected)).item()


# ── EulerScheduler Tests ──


class TestEulerScheduler:
    def test_init(self):
        sched = EulerScheduler(num_steps=20, guide_scale=3.0)
        assert sched.num_steps == 20
        assert sched.guide_scale == 3.0

    def test_get_timestep(self):
        sched = EulerScheduler(num_steps=10)
        assert abs(sched.get_timestep(0) - 1.0) < 1e-6
        assert abs(sched.get_timestep(5) - 0.5) < 1e-6

    def test_step(self):
        sched = EulerScheduler()
        latent = mx.ones((1, 16, 4, 4))
        noise_pred = mx.ones((1, 16, 4, 4)) * 0.5
        result = sched.step(noise_pred, latent, t=0.5, dt=0.1)
        expected = 1.0 - 0.5 * 0.1  # 0.95
        assert mx.allclose(result, mx.full(result.shape, expected)).item()

    def test_guidance(self):
        sched = EulerScheduler(guide_scale=2.0)
        cond = mx.ones((1, 4, 4, 4)) * 3.0
        uncond = mx.ones((1, 4, 4, 4))
        guided = sched.apply_guidance(cond, uncond)
        expected = 1.0 + 2.0 * (3.0 - 1.0)  # 5.0
        assert mx.allclose(guided, mx.full(guided.shape, expected)).item()


# ── TemporalConv3D Tests ──


class TestTemporalConv3D:
    def test_forward_shape(self):
        conv = TemporalConv3D(in_channels=16, out_channels=32, kernel_size=3)
        # NHWTC format: (batch, height, width, time, channels)
        x = mx.random.normal((1, 8, 8, 4, 16))
        out = conv(x)
        assert out.shape[0] == 1  # batch preserved
        assert out.shape[1] == 8  # height preserved
        assert out.shape[2] == 8  # width preserved
        assert out.shape[3] == 4  # temporal preserved
        assert out.shape[4] == 32  # channels changed

    def test_single_frame(self):
        conv = TemporalConv3D(in_channels=8, out_channels=16)
        x = mx.random.normal((1, 4, 4, 1, 8))
        out = conv(x)
        assert out.shape[0] == 1
        assert out.shape[4] == 16
        assert out.shape[3] == 1


# ── VideoVAEDecoder Tests ──


class TestVideoVAEDecoder:
    def test_init_defaults(self):
        decoder = VideoVAEDecoder()
        assert decoder.latent_channels == 16
        assert decoder.output_channels == 3
        assert decoder.spatial_scale == 8

    def test_init_custom(self):
        decoder = VideoVAEDecoder(
            latent_channels=8,
            output_channels=3,
            spatial_scale=4,
            base_channels=64,
        )
        assert decoder.latent_channels == 8
        assert decoder.spatial_scale == 4

    def test_decode_returns_frames(self):
        # Use small dimensions for speed
        decoder = VideoVAEDecoder(
            latent_channels=4,
            output_channels=3,
            spatial_scale=2,
            base_channels=16,
        )
        # (1, C, T, H_lat, W_lat) with spatial_scale=2
        latents = mx.random.normal((1, 4, 2, 4, 4))
        frames = decoder(latents)
        assert len(frames) == 2
        for frame in frames:
            # NHWC output: (H_out, W_out, C) with spatial_scale=2
            assert frame.shape == (8, 8, 3)

    def test_decode_single_frame(self):
        decoder = VideoVAEDecoder(
            latent_channels=4,
            spatial_scale=2,
            base_channels=8,
        )
        latents = mx.random.normal((1, 4, 1, 2, 2))
        frames = decoder(latents)
        assert len(frames) == 1
        assert frames[0].shape[-1] == 3  # 3 output channels


# ── WanVideoPipeline Tests ──


class TestWanVideoPipeline:
    def test_init(self):
        pipeline = WanVideoPipeline()
        assert not pipeline.is_loaded
        assert pipeline.model_path == ""

    def test_init_with_path(self):
        pipeline = WanVideoPipeline(model_path="/some/path")
        assert pipeline.model_path == "/some/path"

    def test_load_nonexistent_path(self):
        pipeline = WanVideoPipeline()
        result = pipeline.load_weights("/nonexistent/path")
        assert result is False
        assert not pipeline.is_loaded

    def test_generate_without_load(self):
        pipeline = WanVideoPipeline()
        req = VideoGenRequest(prompt="test")
        result = pipeline.generate_frames(req)
        assert result.method == "error"
        assert "not loaded" in result.metadata.get("error", "").lower()

    def test_load_from_temp_dir(self):
        """Test loading with a temp directory that has the right structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create required subdirectories
            os.makedirs(os.path.join(tmpdir, "transformer"))
            os.makedirs(os.path.join(tmpdir, "vae"))
            os.makedirs(os.path.join(tmpdir, "text_encoder"))

            pipeline = WanVideoPipeline()
            result = pipeline.load_weights(tmpdir)
            assert result is True
            assert pipeline.is_loaded

            # Generate with minimal params
            req = VideoGenRequest(
                prompt="test video",
                num_frames=5,
                height=64,
                width=64,
                num_steps=2,
                seed=42,
            )
            gen_result = pipeline.generate_frames(req)
            assert gen_result.method.startswith("wan_native")
            assert gen_result.seed_used == 42
            assert gen_result.duration_s > 0
            assert len(gen_result.frames) > 0

    def test_generate_from_image_without_load(self):
        pipeline = WanVideoPipeline()
        image = mx.random.uniform(0, 1, (3, 64, 64))
        req = VideoGenRequest(prompt="animate this", num_frames=5)
        result = pipeline.generate_from_image(req, image)
        assert result.method == "error"

    def test_generate_from_image_with_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))
            os.makedirs(os.path.join(tmpdir, "vae"))

            pipeline = WanVideoPipeline()
            pipeline.load_weights(tmpdir)

            # Image in NHWC format: (H, W, C)
            image = mx.random.uniform(0, 1, (64, 64, 3))
            req = VideoGenRequest(
                prompt="animate",
                num_frames=5,
                height=64,
                width=64,
                num_steps=2,
                seed=42,
            )
            result = pipeline.generate_from_image(req, image)
            assert "wan_i2v" in result.method
            assert len(result.frames) > 0

    def test_stream_frames_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))
            os.makedirs(os.path.join(tmpdir, "vae"))

            pipeline = WanVideoPipeline()
            pipeline.load_weights(tmpdir)

            received = []

            def callback(frame, idx, total):
                received.append((idx, frame.shape))

            req = VideoGenRequest(
                prompt="stream test",
                num_frames=5,
                height=64,
                width=64,
                num_steps=2,
                seed=42,
            )
            result = pipeline.stream_frames(req, callback)
            assert len(result.frames) > 0
            assert len(received) == len(result.frames)
            # Verify indices are sequential
            for i, (idx, _shape) in enumerate(received):
                assert idx == i

    def test_unload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))

            pipeline = WanVideoPipeline()
            pipeline.load_weights(tmpdir)
            assert pipeline.is_loaded

            pipeline.unload()
            assert not pipeline.is_loaded

    def test_get_stats(self):
        pipeline = WanVideoPipeline()
        stats = pipeline.get_stats()
        assert "loaded" in stats
        assert "total_generations" in stats
        assert "total_frames" in stats
        assert "avg_fps" in stats

    def test_euler_scheduler_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))
            os.makedirs(os.path.join(tmpdir, "vae"))

            pipeline = WanVideoPipeline()
            pipeline.load_weights(tmpdir)

            req = VideoGenRequest(
                prompt="euler test",
                num_frames=5,
                height=64,
                width=64,
                num_steps=2,
                scheduler="euler",
                seed=42,
            )
            result = pipeline.generate_frames(req)
            assert "euler" in result.method


# ── VideoLoRAManager Tests ──


class TestVideoLoRAManager:
    def _make_pipeline(self) -> WanVideoPipeline:
        """Create a loaded pipeline for LoRA testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))
            os.makedirs(os.path.join(tmpdir, "vae"))
            pipeline = WanVideoPipeline()
            pipeline.load_weights(tmpdir)
            return pipeline

    def test_init(self):
        pipeline = WanVideoPipeline()
        mgr = VideoLoRAManager(pipeline)
        assert not mgr.is_loaded
        assert not mgr.is_merged
        assert mgr.current_adapter == ""

    def test_apply_nonexistent(self):
        pipeline = WanVideoPipeline()
        mgr = VideoLoRAManager(pipeline)
        result = mgr.apply_lora("/nonexistent/path")
        assert result is False

    def test_apply_and_unload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))
            os.makedirs(os.path.join(tmpdir, "vae"))
            pipeline = WanVideoPipeline()
            pipeline.load_weights(tmpdir)

            # Create fake LoRA adapter
            adapter_dir = os.path.join(tmpdir, "my_lora")
            os.makedirs(adapter_dir)
            config = {"lora_parameters": {"rank": 4, "scale": 10.0}, "num_layers": 2}
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                json.dump(config, f)

            # Create small safetensors file (empty dict is valid)
            safetensors_path = os.path.join(adapter_dir, "adapters.safetensors")
            mx.save_safetensors(safetensors_path, {})

            mgr = VideoLoRAManager(pipeline, max_memory_mb=1024)
            result = mgr.apply_lora(adapter_dir)
            assert result is True
            assert mgr.is_loaded
            assert "my_lora" in mgr.current_adapter

            # Unload
            result = mgr.unload_lora()
            assert result is True
            assert not mgr.is_loaded

    def test_apply_exceeds_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))
            os.makedirs(os.path.join(tmpdir, "vae"))
            pipeline = WanVideoPipeline()
            pipeline.load_weights(tmpdir)

            adapter_dir = os.path.join(tmpdir, "big_lora")
            os.makedirs(adapter_dir)
            config = {"lora_parameters": {"rank": 8, "scale": 20.0}}
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                json.dump(config, f)
            # Create a file larger than memory budget (1KB = 1024 bytes)
            safetensors_path = os.path.join(adapter_dir, "adapters.safetensors")
            # Write 2048 bytes of dummy data
            with open(safetensors_path, "wb") as f:
                f.write(b"\x00" * 2048)

            mgr = VideoLoRAManager(pipeline, max_memory_mb=0)  # 0MB budget
            result = mgr.apply_lora(adapter_dir)
            assert result is False

    def test_merge_and_cannot_unload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))
            os.makedirs(os.path.join(tmpdir, "vae"))
            pipeline = WanVideoPipeline()
            pipeline.load_weights(tmpdir)

            adapter_dir = os.path.join(tmpdir, "merge_lora")
            os.makedirs(adapter_dir)
            config = {"lora_parameters": {"rank": 4, "scale": 10.0}}
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                json.dump(config, f)
            mx.save_safetensors(os.path.join(adapter_dir, "adapters.safetensors"), {})

            mgr = VideoLoRAManager(pipeline)
            mgr.apply_lora(adapter_dir)

            # Merge
            result = mgr.merge_lora()
            assert result is True
            assert mgr.is_merged

            # Cannot unload merged
            result = mgr.unload_lora()
            assert result is False

    def test_merge_without_load(self):
        pipeline = WanVideoPipeline()
        mgr = VideoLoRAManager(pipeline)
        result = mgr.merge_lora()
        assert result is False

    def test_unload_without_load(self):
        pipeline = WanVideoPipeline()
        mgr = VideoLoRAManager(pipeline)
        result = mgr.unload_lora()
        assert result is False

    def test_hot_swap(self):
        """Applying a new LoRA automatically unloads the previous one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))
            os.makedirs(os.path.join(tmpdir, "vae"))
            pipeline = WanVideoPipeline()
            pipeline.load_weights(tmpdir)

            # Create two fake adapters
            for name in ("lora_a", "lora_b"):
                adapter_dir = os.path.join(tmpdir, name)
                os.makedirs(adapter_dir)
                config = {"lora_parameters": {"rank": 4, "scale": 10.0}}
                with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                    json.dump(config, f)
                mx.save_safetensors(
                    os.path.join(adapter_dir, "adapters.safetensors"), {}
                )

            mgr = VideoLoRAManager(pipeline)

            # Load first adapter
            mgr.apply_lora(os.path.join(tmpdir, "lora_a"))
            assert "lora_a" in mgr.current_adapter

            # Hot-swap to second adapter
            mgr.apply_lora(os.path.join(tmpdir, "lora_b"))
            assert "lora_b" in mgr.current_adapter
            assert mgr.is_loaded

    def test_get_status(self):
        pipeline = WanVideoPipeline()
        mgr = VideoLoRAManager(pipeline)
        status = mgr.get_status()
        assert "loaded" in status
        assert "merged" in status
        assert "adapter_id" in status
        assert "rank" in status
        assert "scale" in status
        assert "max_memory_mb" in status
        assert "total_loads" in status
        assert "total_unloads" in status

    def test_no_config_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))
            os.makedirs(os.path.join(tmpdir, "vae"))
            pipeline = WanVideoPipeline()
            pipeline.load_weights(tmpdir)

            adapter_dir = os.path.join(tmpdir, "no_config")
            os.makedirs(adapter_dir)
            # Only safetensors, no config.json
            mx.save_safetensors(os.path.join(adapter_dir, "adapters.safetensors"), {})

            mgr = VideoLoRAManager(pipeline)
            result = mgr.apply_lora(adapter_dir)
            assert result is False
