"""Import compatibility of the moved blocks (M3): every old ``kernels/`` name resolves to the same module
object as its new ``blocks/`` name, so old imports, monkeypatches and attribute reads keep working."""

import importlib
import sys

import pytest

MOVED = (
    # (old name, new name)
    ("yunshu_engine.kernels.ragged_kv", "yunshu_engine.blocks.kv"),  # M3a
    ("yunshu_engine.kernels.lane_linear", "yunshu_engine.blocks.linear"),  # M3b
    ("yunshu_engine.kernels.gdn_rows", "yunshu_engine.blocks.recurrent"),  # M3c
    ("yunshu_engine.text_logprobs", "yunshu_engine.blocks.head"),  # M3d
)
IDS = [new.rsplit(".", 1)[1] for _, new in MOVED]


@pytest.mark.parametrize("old,new", MOVED, ids=IDS)
def test_old_name_is_the_same_module_object(old, new):
    a = importlib.import_module(old)
    b = importlib.import_module(new)
    assert a is b
    assert sys.modules[old] is b and sys.modules[new] is b


@pytest.mark.parametrize("old,new", MOVED, ids=IDS)
def test_every_public_name_is_the_same_object(old, new):
    old_mod = importlib.import_module(old)
    new_mod = importlib.import_module(new)
    names = [n for n in vars(new_mod) if not n.startswith("__")]
    assert names, new
    for name in names:
        assert getattr(old_mod, name) is getattr(new_mod, name), name
