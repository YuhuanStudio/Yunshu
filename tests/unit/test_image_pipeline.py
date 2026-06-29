"""Tests for image pipeline registry — multi-model diffusion support."""

import json
import os
import tempfile


class TestPipelineType:
    def test_enum_values(self):
        from yunshu_engine.image_pipeline import PipelineType

        assert PipelineType.Z_IMAGE.value == "z_image"
        assert PipelineType.FLUX.value == "flux"
        assert PipelineType.FLUX2.value == "flux2"
        assert PipelineType.QWEN_IMAGE.value == "qwen_image"


class TestPipelineInfo:
    def test_creation(self):
        from yunshu_engine.image_pipeline import PipelineInfo, PipelineType

        info = PipelineInfo(
            pipeline_type=PipelineType.Z_IMAGE,
            name="Test",
            description="Test pipeline",
            default_steps=4,
            latent_channels=16,
        )
        assert info.pipeline_type == PipelineType.Z_IMAGE
        assert info.default_steps == 4

    def test_features_list(self):
        from yunshu_engine.image_pipeline import PipelineInfo, PipelineType

        info = PipelineInfo(
            pipeline_type=PipelineType.Z_IMAGE,
            name="Test",
            description="Test",
            supported_features=["text2img", "inpaint"],
        )
        assert "inpaint" in info.supported_features


class TestRegistry:
    def test_register_and_get(self):
        from yunshu_engine.image_pipeline import (
            PipelineInfo,
            PipelineType,
            get_pipeline_info,
            register_pipeline,
        )

        register_pipeline(
            "test-pipeline-x",
            PipelineInfo(
                pipeline_type=PipelineType.Z_IMAGE,
                name="Test Pipeline X",
                description="Test",
            ),
        )
        info = get_pipeline_info("Test-Pipeline-X")
        assert info is not None
        assert info.name == "Test Pipeline X"

    def test_get_nonexistent(self):
        from yunshu_engine.image_pipeline import get_pipeline_info

        assert get_pipeline_info("nonexistent-pipeline-xyz") is None

    def test_list_pipelines(self):
        from yunshu_engine.image_pipeline import list_pipelines

        pipelines = list_pipelines()
        assert len(pipelines) > 0
        assert any("z-image" in name for name in pipelines)

    def test_zimage_turbo_registered(self):
        from yunshu_engine.image_pipeline import get_pipeline_info

        info = get_pipeline_info("z-image-turbo-mlx-4bit")
        assert info is not None
        assert "text2img" in info.supported_features
        assert "inpaint" in info.supported_features
        assert info.default_steps == 4

    def test_flux_registered(self):
        from yunshu_engine.image_pipeline import get_pipeline_info

        info = get_pipeline_info("flux-dev")
        assert info is not None

    def test_qwen_image_registered(self):
        from yunshu_engine.image_pipeline import get_pipeline_info

        info = get_pipeline_info("qwen-image")
        assert info is not None


class TestDetectPipelineType:
    def test_zimage_path(self):
        from yunshu_engine.image_pipeline import PipelineType, detect_pipeline_type

        assert (
            detect_pipeline_type("/models/Z-Image-Turbo-MLX-4bit")
            == PipelineType.Z_IMAGE
        )

    def test_flux_path(self):
        from yunshu_engine.image_pipeline import PipelineType, detect_pipeline_type

        assert detect_pipeline_type("/models/flux-dev") == PipelineType.FLUX
        assert detect_pipeline_type("/models/FLUX.1-schnell") == PipelineType.FLUX

    def test_flux2_path(self):
        from yunshu_engine.image_pipeline import PipelineType, detect_pipeline_type

        assert detect_pipeline_type("/models/flux2-klein") == PipelineType.FLUX2

    def test_qwen_path(self):
        from yunshu_engine.image_pipeline import PipelineType, detect_pipeline_type

        assert detect_pipeline_type("/models/qwen-image-2.5") == PipelineType.QWEN_IMAGE

    def test_unknown_path(self):
        from yunshu_engine.image_pipeline import PipelineType, detect_pipeline_type

        assert detect_pipeline_type("/random/path") == PipelineType.UNKNOWN

    def test_directory_structure_detection(self):
        """Test detection from directory structure when name is ambiguous."""
        from yunshu_engine.image_pipeline import PipelineType, detect_pipeline_type

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Z-Image directory structure
            for d in ("transformer", "vae", "text_encoder"):
                os.makedirs(os.path.join(tmpdir, d))

            result = detect_pipeline_type(tmpdir)
            assert result == PipelineType.Z_IMAGE

    def test_config_based_detection(self):
        """Test detection from transformer config.json."""
        from yunshu_engine.image_pipeline import PipelineType, detect_pipeline_type

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "transformer"))
            config = {"_class_name": "FluxTransformer2DModel"}
            with open(os.path.join(tmpdir, "transformer", "config.json"), "w") as f:
                json.dump(config, f)

            result = detect_pipeline_type(tmpdir)
            assert result == PipelineType.FLUX


class TestGetPipelineInfoForPath:
    def test_known_path(self):
        from yunshu_engine.image_pipeline import get_pipeline_info_for_path

        info = get_pipeline_info_for_path("/models/Z-Image-Turbo-MLX-4bit")
        assert info.name == "Z-Image-Turbo-MLX-4bit"

    def test_unknown_path_fallback(self):
        from yunshu_engine.image_pipeline import get_pipeline_info_for_path

        info = get_pipeline_info_for_path("/some/random/model")
        assert info is not None  # Should return a default info


class TestCreatePipeline:
    def test_create_zimage(self):
        from yunshu_engine.image_engine import ImageGenEngine
        from yunshu_engine.image_pipeline import create_pipeline_for_path

        pipeline = create_pipeline_for_path("/models/Z-Image-Turbo-MLX-4bit")
        assert isinstance(pipeline, ImageGenEngine)

    def test_create_flux_fallback(self):
        """Flux should fall back to Z-Image engine (not yet supported)."""
        from yunshu_engine.image_engine import ImageGenEngine
        from yunshu_engine.image_pipeline import create_pipeline_for_path

        pipeline = create_pipeline_for_path("/models/flux-dev")
        assert isinstance(pipeline, ImageGenEngine)
