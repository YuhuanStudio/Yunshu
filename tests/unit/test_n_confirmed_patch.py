"""Tests for n_confirmed MTP support patch."""

from unittest.mock import MagicMock


class TestClearRollback:
    """Test clear_rollback clears all rollback states."""

    def test_clears_ssm_rollback_state(self):
        from yunshu_engine.n_confirmed_patch import clear_rollback
        cache = [MagicMock()]
        cache[0].rollback_state = ("conv_snap", "ssm_snap")
        clear_rollback(cache)
        assert cache[0].rollback_state is None

    def test_clears_multiple_layers(self):
        from yunshu_engine.n_confirmed_patch import clear_rollback
        cache = [MagicMock(), MagicMock(), MagicMock()]
        cache[0].rollback_state = ("conv1", "ssm1")
        cache[1].rollback_state = ("conv2", "ssm2")
        # cache[2] has no rollback_state
        clear_rollback(cache)
        assert cache[0].rollback_state is None
        assert cache[1].rollback_state is None

    def test_no_rollback_state_does_nothing(self):
        from yunshu_engine.n_confirmed_patch import clear_rollback
        cache = [MagicMock(spec=["advance"])]
        # No crash
        clear_rollback(cache)


class TestRestoreRollback:
    """Test restore_rollback after rejected draft."""

    def test_restores_ssm_state(self):
        from yunshu_engine.n_confirmed_patch import restore_rollback
        cache = [MagicMock()]
        conv_snap = MagicMock()
        ssm_snap = MagicMock()
        cache[0].rollback_state = (conv_snap, ssm_snap)
        cache[0].lengths = 5
        # Make cache[0][0] / cache[0][1] work via __getitem__
        cache[0].__getitem__ = MagicMock(side_effect=lambda i: (conv_snap, ssm_snap)[i])

        result = restore_rollback(cache)
        assert result is True
        assert cache[0].rollback_state is None
        assert cache[0].lengths == 4  # decremented by 1

    def test_trims_kv_layers(self):
        from yunshu_engine.n_confirmed_patch import restore_rollback
        cache = [MagicMock()]
        cache[0].is_trimmable.return_value = True
        cache[0].rollback_state = None  # not an SSM layer

        result = restore_rollback(cache)
        assert result is True
        cache[0].trim.assert_called_once_with(1)

    def test_mixed_layers(self):
        from yunshu_engine.n_confirmed_patch import restore_rollback
        ssm_layer = MagicMock()
        ssm_layer.rollback_state = ("conv", "ssm")
        ssm_layer.lengths = 10

        kv_layer = MagicMock()
        kv_layer.is_trimmable.return_value = True
        kv_layer.rollback_state = None

        cache = [ssm_layer, kv_layer]
        result = restore_rollback(cache)
        assert result is True
        assert ssm_layer.lengths == 9
        kv_layer.trim.assert_called_once_with(1)

    def test_returns_false_for_unsupported_layer(self):
        from yunshu_engine.n_confirmed_patch import restore_rollback
        cache = [MagicMock()]
        cache[0].rollback_state = None
        cache[0].is_trimmable.return_value = False
        # Still processes but returns False
        result = restore_rollback(cache)
        assert result is False


class TestApplyNPConfirmedPatch:
    """Test patch application is idempotent."""

    def test_patch_is_idempotent(self):
        from yunshu_engine.n_confirmed_patch import apply_n_confirmed_patch
        # First call may or may not succeed depending on mlx_lm availability
        result1 = apply_n_confirmed_patch()
        result2 = apply_n_confirmed_patch()
        assert result1 == result2
