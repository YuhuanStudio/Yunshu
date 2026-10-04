import mlx.core as mx

from yunshu_engine.memory_census import census


class _Holder:
    def __init__(self):
        self.keys = mx.zeros((1024, 1024), dtype=mx.float32)  # 4 MiB
        self.layers = [mx.zeros((512, 1024), dtype=mx.float32)]  # 2 MiB


def test_census_names_the_holding_attribute():
    h = _Holder()
    mx.eval(h.keys, h.layers[0])
    out = census(min_mib=1.0)
    holders = {g["holder"]: g for g in out["holders"]}
    assert holders["_Holder.keys"]["mib"] == 4.0
    assert any(k.startswith("_Holder.layers") for k in holders)
    assert out["counted_mib"] >= 6.0
    del h


def test_census_skips_small_arrays():
    small = mx.zeros((8,))
    mx.eval(small)
    assert all(g["mib"] >= 1.0 for g in census(min_mib=1.0)["holders"])
