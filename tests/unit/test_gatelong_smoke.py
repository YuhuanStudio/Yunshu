import importlib.util
import pathlib

p = pathlib.Path(__file__).resolve().parents[2] / "scripts/research/gatelong_smoke.py"
spec = importlib.util.spec_from_file_location("gatelong_smoke", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_parse_and_check():
    t = (
        'yunshu_process_footprint_bytes{type="current"} 100\n'
        'yunshu_process_footprint_bytes{type="peak"} 200\n'
        "yunshu_process_footprint_samples_total 50\n"
    )
    fp = m.parse_footprint(t)
    assert fp == {"current": 100, "peak": 200, "samples": 50}
    assert m.check_footprint(fp) == []
    assert m.check_footprint(m.parse_footprint("")) != []
