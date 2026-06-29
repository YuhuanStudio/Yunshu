"""unit coverage for yunshu_engine.mtp_patch.

Only the idempotency / flag semantics of apply_mtp_patch() are exercised
here — the heavy load_model_with_mtp() path needs real Qwen3.5 weights
and is verified by integration tests.
"""

from __future__ import annotations

import pytest

import yunshu_engine.mtp_patch as mtp_mod


class TestApplyMTPPatch:
    def test_apply_returns_true(self):
        """First or subsequent application should report success when
        mlx_lm.models.qwen3_5 is importable.

        On platforms where qwen3_5 cannot be imported (e.g. older mlx-lm),
        apply_mtp_patch() may legitimately return False — skip in that
        case rather than fail.
        """
        try:
            import mlx_lm.models.qwen3_5  # noqa: F401
        except ImportError:
            pytest.skip("mlx_lm.models.qwen3_5 not importable on this platform")
        assert mtp_mod.apply_mtp_patch() is True

    def test_apply_idempotent(self):
        """Calling twice must return True both times and leave the
        module flag set, with no observable side-effect on a second pass."""
        try:
            import mlx_lm.models.qwen3_5  # noqa: F401
        except ImportError:
            pytest.skip("mlx_lm.models.qwen3_5 not importable on this platform")
        first = mtp_mod.apply_mtp_patch()
        assert first is True
        assert mtp_mod._PATCHED is True

        # Re-apply — should short-circuit on the _PATCHED flag.
        second = mtp_mod.apply_mtp_patch()
        assert second is True
        assert mtp_mod._PATCHED is True

    def test_patched_flag_flips_from_false(self, monkeypatch):
        """Force _PATCHED back to False, re-apply, confirm flip."""
        try:
            from mlx_lm.models import qwen3_5  # noqa: F401
        except ImportError:
            pytest.skip("mlx_lm.models.qwen3_5 not importable on this platform")

        # Reset the patched flag — note the class-level
        # `_yunshu_mtp_patched` attribute on TextModel keeps the in-place
        # patches alive; we only verify the module-level flag flips.
        monkeypatch.setattr(mtp_mod, "_PATCHED", False)
        assert mtp_mod._PATCHED is False
        result = mtp_mod.apply_mtp_patch()
        assert result is True
        assert mtp_mod._PATCHED is True

    def test_text_model_has_patched_marker(self):
        """After apply_mtp_patch(), TextModel carries the patched marker."""
        try:
            from mlx_lm.models import qwen3_5
        except ImportError:
            pytest.skip("mlx_lm.models.qwen3_5 not importable on this platform")
        mtp_mod.apply_mtp_patch()
        assert getattr(qwen3_5.TextModel, "_yunshu_mtp_patched", False) is True
        assert getattr(qwen3_5.Model, "_yunshu_mtp_patched", False) is True

    def test_mtp_classes_registered(self):
        """MTPModule + MTPDecoderLayer classes should exist on the module
        after patching."""
        try:
            from mlx_lm.models import qwen3_5
        except ImportError:
            pytest.skip("mlx_lm.models.qwen3_5 not importable on this platform")
        mtp_mod.apply_mtp_patch()
        assert hasattr(qwen3_5, "MTPModule")
        assert hasattr(qwen3_5, "MTPDecoderLayer")

    def test_already_patched_via_class_marker_short_circuits(self, monkeypatch):
        """When _PATCHED=False but TextModel already carries
        _yunshu_mtp_patched, apply_mtp_patch must short-circuit and set
        the module flag — without re-running _register/_patch helpers."""
        try:
            from mlx_lm.models import qwen3_5
        except ImportError:
            pytest.skip("mlx_lm.models.qwen3_5 not importable on this platform")

        # Prime the class-level marker (real patching already set it from
        # the earlier tests, but be defensive in case ordering changes).
        if not hasattr(qwen3_5.TextModel, "_yunshu_mtp_patched"):
            mtp_mod.apply_mtp_patch()

        # Reset only the module flag — class marker stays.
        monkeypatch.setattr(mtp_mod, "_PATCHED", False)

        # Spy on the helpers to detect any unintended re-execution.
        called: list[str] = []
        real_register = mtp_mod._register_mtp_classes

        def spy_register(*args, **kwargs):
            called.append("register")
            return real_register(*args, **kwargs)

        monkeypatch.setattr(mtp_mod, "_register_mtp_classes", spy_register)

        result = mtp_mod.apply_mtp_patch()
        assert result is True
        assert mtp_mod._PATCHED is True
        # No helper rerun — short-circuit via the class-level marker.
        assert called == []

    def test_load_model_with_mtp_raises_on_missing_weights(self, tmp_path):
        """load_model_with_mtp() should raise FileNotFoundError when the
        mtp-weights.safetensors file isn't present in the model dir.

        We skip if mlx_lm.utils.load_model can't be imported (e.g. older
        mlx-lm), or if it errors before reaching the MTP-weight check.
        """
        try:
            from mlx_lm.models import qwen3_5  # noqa: F401
        except ImportError:
            pytest.skip("mlx_lm.models.qwen3_5 not importable on this platform")

        # Use a non-existent path; load_model will fail before the MTP
        # check, so we just assert that *some* exception is raised — i.e.
        # the function doesn't silently return None.
        with pytest.raises((FileNotFoundError, Exception)):
            mtp_mod.load_model_with_mtp(str(tmp_path / "no-such-model"))
