"""CPU schedule/sources guards for the measured native M5 paths."""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[2]
KERNELS = ROOT / "python/yunshu_engine/kernels/tensorfold"


def schedule_namespace():
    tree = ast.parse((KERNELS / "lane_qmm.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_lane_schedule"
    )
    namespace = {"_VARIANT": "m5"}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "schedule", "exec"),
        namespace,
    )
    return namespace


def test_schedule_is_limited_to_measured_shapes_and_rows():
    namespace = schedule_namespace()
    select = namespace["_lane_schedule"]
    for rows in range(1, 65):
        assert select(5, 5120, 17408, rows, 64) == "default"
        assert select(4, 5120, 17408, rows, 64) == "default"
        assert select(8, 34816, 5120, rows, 64) == (
            "direct8" if rows <= 16 else "default"
        )
        assert select(8, 5120, 25600, rows, 64) == (
            "direct8" if rows <= 32 else "default"
        )
        for bits, n, k in [
            (5, 48, 5120),
            (5, 1024, 5120),
            (4, 17408, 5120),
            (8, 248320, 5120),
        ]:
            assert select(bits, n, k, rows, 64) == "default"
        assert select(5, 5120, 17408, rows, 32) == "default"
        assert select(5, 5120, 17408, rows, 64, 64) == "default"
    namespace["_VARIANT"] = "portable"
    assert select(5, 5120, 17408, 32, 64) == "default"
    assert select(8, 34816, 5120, 16, 64) == "default"


def test_native_sources_match_the_bit_qualified_experiments():
    native, widen = {}, {}
    exec(compile((KERNELS / "lane_m5.py").read_text(), "m5_source", "exec"), native)
    exec(
        compile((KERNELS / "lane_widen_m5.py").read_text(), "widen_source", "exec"),
        widen,
    )
    for file, function, sources in [
        ("lane_direct8.py", "direct_bytes", [widen["BYTES"]])
    ]:
        spec = importlib.util.spec_from_file_location(
            "qualified_source", ROOT / "scripts/research" / file
        )
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
        for source in sources:
            assert native[function](source) == experiment.transform(source)
