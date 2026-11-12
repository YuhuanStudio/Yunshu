"""Silent-fallback gates count their misses (row 317): each gate that answers no records one."""

from yunshu_engine import dflash_fast, fallbacks, mtp_lane, tree_verify


def _kinds():
    return fallbacks.counts()


def test_gates_count_their_misses():
    fallbacks.reset()
    assert tree_verify.supported(object()) is False
    assert dflash_fast.supported(object(), object()) is False
    assert mtp_lane.verify_max_rows(False) == 8
    k = _kinds()
    assert k["tree_verify_unsupported"]["count"] == 1
    assert k["dflash_fast_unsupported"]["count"] == 1
    assert k["verify_max_rows_narrow"]["count"] == 1
