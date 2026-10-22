"""A complete stream alone cannot certify lossless APC restoration."""

import pytest
from scripts.research.apc_restore_identity import assert_identity


def results():
    return {
        mode: dict(cached=128, tokens_equal=True, logprobs_equal=True)
        for mode in ("partial", "full")
    }


@pytest.mark.parametrize("mode", ["partial", "full"])
@pytest.mark.parametrize("field", ["cached", "tokens_equal", "logprobs_equal"])
def test_identity_probe_rejects_miss_and_bit_drift(mode, field):
    out = results()
    out[mode][field] = False
    with pytest.raises(RuntimeError, match="not bit-equal"):
        assert_identity(out)


def test_identity_probe_accepts_both_restores():
    assert_identity(results())


def test_spec_off_reference_does_not_require_a_cache_hit():
    out = results()
    out["spec_off"] = dict(cached=0, tokens_equal=True, logprobs_equal=True)
    assert_identity(out)
    out["spec_off"]["logprobs_equal"] = False
    with pytest.raises(RuntimeError, match="spec_off restore is not bit-equal"):
        assert_identity(out)
