"""CPU checks for the context cap; copy proposals keep their own allowance."""

import pytest

from yunshu_engine.round_driver.allocate import dflash_room, fair_nodes


@pytest.mark.parametrize("position", [0, 32768, 65536, 131071])
def test_short_context_retains_fifteen_model_nodes(position):
    assert dflash_room(position, 100, 0) == 15


@pytest.mark.parametrize("position", [131072, 131073, 262144])
def test_long_context_model_tree_fits_eight_verify_rows(position):
    room = dflash_room(position, 100, 0)
    assert fair_nodes([room], budget=64) == [7]


@pytest.mark.parametrize("remaining", [0, 1, 6, 7])
def test_length_limit_still_bounds_model_tree(remaining):
    assert dflash_room(131072, remaining, 0) == remaining


@pytest.mark.parametrize("copied", [1, 15, 31, 63])
def test_long_context_keeps_copy_proposals(copied):
    assert dflash_room(131072, 100, copied) == copied


def test_mixed_context_pack_shares_unused_long_context_nodes():
    rooms = [dflash_room(pos, 100, 0) for pos in [131072, 32768]]
    assert fair_nodes(rooms, budget=32) == [7, 15]
