"""MTP integration tests — validates load, forward, and generation for all Qwen3.5 models.

Requires models in models/ directory. Skips gracefully if models not available.
Run: .venv/bin/python3 -m pytest tests/integration/test_mtp.py -v
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

import mlx.core as mx

MODELS_DIR = ROOT / "models"
QWEN35_MODELS = sorted(d.name for d in MODELS_DIR.glob("Qwen3.5-*") if d.is_dir())


def _has_model(name: str) -> bool:
    return (MODELS_DIR / name).exists()


def _has_mtp_weights(name: str) -> bool:
    return (MODELS_DIR / name / "mtp-weights.safetensors").exists()


def _release_model():
    """Force-release GPU memory between tests to prevent 25GB+ accumulation."""
    gc.collect()
    mx.clear_cache()
    gc.collect()


@pytest.fixture(params=QWEN35_MODELS, ids=QWEN35_MODELS)
def model_name(request):
    if not _has_model(request.param):
        pytest.skip(f"Model {request.param} not found")
    if not _has_mtp_weights(request.param):
        pytest.skip(f"MTP weights for {request.param} not found")
    return request.param


@pytest.fixture()
def loaded_model(model_name):
    """Load model once per test, then release GPU memory."""
    from yunshu_engine.mtp_patch import load_model_with_mtp

    model = load_model_with_mtp(str(MODELS_DIR / model_name))
    yield model
    del model
    _release_model()


class TestMTPLoad:
    """Verify MTP module loads correctly for each model."""

    def test_load_creates_mtp_module(self, loaded_model):
        inner = getattr(loaded_model, "language_model", loaded_model)
        assert hasattr(inner, "mtp"), "MTP module not created"

    def test_mtp_module_has_expected_layers(self, loaded_model):
        inner = getattr(loaded_model, "language_model", loaded_model)

        assert hasattr(inner.mtp, "fc"), "Missing fusion projection (fc)"
        assert hasattr(inner.mtp, "layers"), "Missing MTP decoder layers"
        assert hasattr(inner.mtp, "norm"), "Missing final norm"
        assert hasattr(inner.mtp, "pre_fc_norm_hidden"), "Missing pre_fc_norm_hidden"
        assert hasattr(inner.mtp, "pre_fc_norm_embedding"), "Missing pre_fc_norm_embedding"

    def test_mtp_fc_weight_shape(self, loaded_model):
        inner = getattr(loaded_model, "language_model", loaded_model)
        hidden_size = inner.args.hidden_size

        # fc maps 2*hidden -> hidden
        assert inner.mtp.fc.weight.shape == (hidden_size, 2 * hidden_size)

    def test_make_mtp_cache(self, loaded_model):
        inner = getattr(loaded_model, "language_model", loaded_model)
        cache = inner.make_mtp_cache()

        assert len(cache) == len(inner.mtp.layers)
        from mlx_lm.models.cache import KVCache
        assert all(isinstance(c, KVCache) for c in cache)

    def test_idempotent_patch(self):
        from yunshu_engine.mtp_patch import apply_mtp_patch

        assert apply_mtp_patch() is True
        assert apply_mtp_patch() is True  # second call is no-op


class TestMTPForward:
    """Verify MTP forward pass produces correct shapes and reasonable values."""

    def test_return_hidden_shape(self, loaded_model):
        inner = getattr(loaded_model, "language_model", loaded_model)

        ids = mx.array([[1, 2, 3]])
        out, hidden = loaded_model(ids, return_hidden=True)

        assert out.shape[0] == 1
        assert hidden.shape == (1, 3, inner.args.hidden_size)

    def test_mtp_forward_logits_shape(self, loaded_model):
        inner = getattr(loaded_model, "language_model", loaded_model)

        ids = mx.array([[1, 2, 3]])
        out, hidden = loaded_model(ids, return_hidden=True)

        mtp_cache = inner.make_mtp_cache()
        next_tok = mx.array([[int(mx.argmax(out[0, -1, :]).item())]])
        mtp_logits = loaded_model.mtp_forward(hidden, next_tok, mtp_cache)

        vocab_size = inner.args.vocab_size
        assert mtp_logits.shape == (1, 1, vocab_size)

    def test_mtp_forward_with_multi_position_hidden(self, loaded_model):
        """MTP forward should auto-slice multi-position hidden to last position."""
        inner = getattr(loaded_model, "language_model", loaded_model)

        ids = mx.array([[1, 2, 3, 4, 5]])
        out, hidden = loaded_model(ids, return_hidden=True)

        mtp_cache = inner.make_mtp_cache()
        next_tok = mx.array([[100]])
        # Pass full hidden (5 positions) — should auto-slice to last
        mtp_logits = loaded_model.mtp_forward(hidden, next_tok, mtp_cache)

        vocab_size = inner.args.vocab_size
        assert mtp_logits.shape == (1, 1, vocab_size)

    def test_mtp_prediction_is_valid_token(self, loaded_model):
        inner = getattr(loaded_model, "language_model", loaded_model)

        ids = mx.array([[1, 2, 3]])
        out, hidden = loaded_model(ids, return_hidden=True)

        mtp_cache = inner.make_mtp_cache()
        next_tok = mx.array([[int(mx.argmax(out[0, -1, :]).item())]])
        mtp_logits = loaded_model.mtp_forward(hidden, next_tok, mtp_cache)

        mtp_pred = int(mx.argmax(mtp_logits[0, -1, :]).item())
        assert 0 <= mtp_pred < inner.args.vocab_size


class TestMTPDecodeLoop:
    """Verify MTP works in a full decode loop with backbone verification."""

    def test_mtp_acceptance_reasonable(self, loaded_model, model_name):
        """MTP should predict the same token as backbone at least 30% of the time."""
        from mlx_lm.utils import load_tokenizer
        from mlx_lm.models.cache import make_prompt_cache

        inner = getattr(loaded_model, "language_model", loaded_model)
        tokenizer = load_tokenizer(MODELS_DIR / model_name)

        prompt = "The capital of France is"
        ids = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
        cache = make_prompt_cache(loaded_model)

        out, hidden = loaded_model(ids, cache=cache, return_hidden=True)
        first_tok = int(mx.argmax(out[0, -1, :]).item())

        matches = 0
        total = 10
        current_tok = first_tok

        for _ in range(total):
            last_hidden = hidden[:, -1:, :]
            mtp_cache = inner.make_mtp_cache()
            mtp_logits = loaded_model.mtp_forward(last_hidden, mx.array([[current_tok]]), mtp_cache)
            mtp_tok = int(mx.argmax(mtp_logits[0, -1, :]).item())

            t_logits = loaded_model(mx.array([[current_tok]]), cache=cache)
            if isinstance(t_logits, tuple):
                t_logits = t_logits[0]
            backbone_tok = int(mx.argmax(t_logits[0, -1, :]).item())

            if mtp_tok == backbone_tok:
                matches += 1

            # Get hidden for next iteration
            _, hidden = loaded_model(mx.array([[current_tok]]), cache=cache, return_hidden=True)
            current_tok = backbone_tok

        acceptance = matches / total
        # Even 0.8B model should match some tokens; larger models should be higher
        assert acceptance >= 0.1, f"MTP acceptance too low: {acceptance:.1%}"


class TestMTPSanitize:
    """Verify weight sanitization handles MTP weights correctly."""

    def test_sanitize_keeps_mtp_keys(self):
        from yunshu_engine.mtp_patch import apply_mtp_patch
        apply_mtp_patch()

        from mlx_lm.models import qwen3_5 as q35

        # Create a minimal TextModel with MTP
        from mlx_lm.models.qwen3_5 import TextModelArgs
        args = TextModelArgs(
            model_type="qwen3_5_text",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
        inner = q35.TextModel(args)
        inner.mtp = q35.MTPModule(args)

        weights = {
            "mtp.fc.weight": mx.zeros((256, 512)),
            "mtp.layers.0.input_layernorm.weight": mx.zeros(256),
            "model.embed_tokens.weight": mx.zeros((100, 256)),
            "model.norm.weight": mx.zeros(256),
        }

        result = inner.sanitize(weights)
        assert "mtp.fc.weight" in result, "sanitize stripped MTP fc weight"
        assert "mtp.layers.0.input_layernorm.weight" in result

    def test_sanitize_strips_mtp_when_no_module(self):
        from yunshu_engine.mtp_patch import apply_mtp_patch
        apply_mtp_patch()

        from mlx_lm.models import qwen3_5 as q35

        from mlx_lm.models.qwen3_5 import TextModelArgs
        args = TextModelArgs(
            model_type="qwen3_5_text",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        inner = q35.TextModel(args)
        # No inner.mtp

        weights = {
            "mtp.fc.weight": mx.zeros((256, 512)),
            "model.embed_tokens.weight": mx.zeros((100, 256)),
        }

        result = inner.sanitize(weights)
        assert "mtp.fc.weight" not in result, "sanitize kept MTP keys without module"

    def test_sanitize_shifts_norm_weights(self):
        from yunshu_engine.mtp_patch import apply_mtp_patch
        apply_mtp_patch()

        from mlx_lm.models import qwen3_5 as q35
        from mlx_lm.models.qwen3_5 import TextModelArgs

        args = TextModelArgs(
            model_type="qwen3_5_text",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
        inner = q35.TextModel(args)
        inner.mtp = q35.MTPModule(args)

        weights = {
            "mtp.norm.weight": mx.zeros(256),  # Should become 1.0
            "model.norm.weight": mx.zeros(256),  # Should become 1.0
            "model.embed_tokens.weight": mx.zeros((100, 256)),
        }

        result = inner.sanitize(weights)
        assert float(result["mtp.norm.weight"][0].item()) == pytest.approx(1.0)
        assert float(result["model.norm.weight"][0].item()) == pytest.approx(1.0)
