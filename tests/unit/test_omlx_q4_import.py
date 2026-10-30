import importlib

import pytest

pytest.importorskip("mlx_lm")
pytest.importorskip("mlx_vlm")

MODULES = (
    "module_cache",
    "moe_verify_gather",
    "row_exact_qmv",
    "qwen35_verify_qmm",
    "qwen35_gdn_verify_fused",
    "qwen35_gdn_prework",
)


@pytest.mark.parametrize("name", MODULES)
def test_vendored_oq4_module_imports(name):
    importlib.import_module(f"yunshu_engine.kernels.omlx_q4.{name}")


def test_qwen4_entry_points_exist():
    prework = importlib.import_module(
        "yunshu_engine.kernels.omlx_q4.qwen35_gdn_prework"
    )
    for attr in (
        "_qwen4_verify_plan",
        "_qwen4_verify",
        "_qwen4_verify_state_eligible",
        "qwen4_verify_step_fused",
    ):
        assert callable(getattr(prework, attr)), attr
