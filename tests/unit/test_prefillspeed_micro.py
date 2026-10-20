import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "prefillspeed_micro",
    Path(__file__).parents[2] / "scripts/research/prefillspeed_micro.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_flops_causal_full_prefix():
    # one chunk at depth 0 is a triangle; a chunk behind a prefix adds q*off full pairs
    assert m.sdpa_flops(2048, 2048) == 4.0 * (2048 * 2049 / 2) * 24 * 256
    assert m.sdpa_flops(2048, 4096) > m.sdpa_flops(2048, 2048)


def test_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "o.jsonl"
    assert m.main(["--mode", "sdpa", "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert "flops" in capsys.readouterr().out
