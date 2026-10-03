"""A forward profile must not prune the final MLP from the lazy graph."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_profile():
    path = Path(__file__).parents[2] / "scripts/research/prefill_profile.py"
    spec = importlib.util.spec_from_file_location("prefill_profile_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_materializes_hidden_output_and_cache(monkeypatch):
    module = load_profile()
    hidden, state = object(), object()
    evaluated = []

    class Inputs:
        def __getitem__(self, _):
            return self

    monkeypatch.setattr(module.mx, "array", lambda _: Inputs())
    monkeypatch.setattr(module.mx, "eval", lambda *args: evaluated.append(args))
    owner = SimpleNamespace(make_cache=lambda: [SimpleNamespace(state=state)])
    times, _ = module.run_chunks(
        lambda *args, **kwargs: hidden, owner, list(range(4)), 2, clear=False
    )
    assert len(times) == 2
    assert evaluated == [(hidden, [state]), (hidden, [state])]


def test_nested_layout_and_gdn_timers_are_not_double_counted():
    timers = load_profile().Timers()
    timers.t.update(
        {
            "mlp.down": 1.0,
            "mlp.down.layout": 0.2,
            "gdn.prepare+core": 0.5,
            "gdn.native_core": 0.4,
            "norm": 0.1,
        }
    )
    timers.n["gdn.prepare+core"] = 48
    result = timers.exclusive()
    assert result["mlp.down"] == 0.8
    assert result["gdn.prepare"] == 0.5 - 0.4
    assert timers.n["gdn.prepare"] == 48
    assert abs(sum(result.values()) - 1.6) < 1e-12
