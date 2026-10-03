from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("abort", [False, True])
def test_conv_arm_engages_its_mode_and_restores_upstream(enabled, abort, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts/research")
    )
    from scripts.research.bench_dflash_precision import run_with_conv

    original, raw, compiled = object(), object(), object()
    module = SimpleNamespace(_grouped_dynamic_convolve=original)

    def run():
        assert module._grouped_dynamic_convolve is (compiled if enabled else raw)
        if abort:
            raise RuntimeError("generation aborted")
        return [17, 42]

    if abort:
        with pytest.raises(RuntimeError, match="aborted"):
            run_with_conv(run, module, raw, compiled, enabled=enabled)
    else:
        assert run_with_conv(run, module, raw, compiled, enabled=enabled) == [17, 42]
    assert module._grouped_dynamic_convolve is original
