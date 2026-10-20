"""A stopped engine's Flash-Next plans (which hold its expert weights) are dropped."""

from yunshu_engine.kernels import tf_hc_forward


class _Module:
    def __init__(self, children=()):
        self.children = list(children)

    def modules(self):
        out = [self]
        for c in self.children:
            out.extend(c.modules())
        return out


def test_forget_drops_only_that_models_plans(monkeypatch):
    mlp, hc = _Module(), _Module()
    model = _Module([_Module([mlp, hc])])
    other = _Module()
    plans = {
        id(hc): "hc plan",
        ("moe", id(mlp)): {"sw": object()},
        ("model", id(model)): True,
        ("moe", id(other)): {"sw": object()},
    }
    monkeypatch.setitem(tf_hc_forward._STATE, "modules", plans)
    tf_hc_forward.forget(model)
    assert list(plans) == [("moe", id(other))]
    tf_hc_forward.forget(None)
    assert len(plans) == 1
