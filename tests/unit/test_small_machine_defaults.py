"""Cache budgets scale with the machine: simulated 8 / 16 / 32 / 64 / 128 GB of RAM and
128 GB - 2 TB of disk, no real hardware read."""

from __future__ import annotations

from collections import namedtuple

import pytest

from yunshu_engine import apc_manager, settings
from yunshu_engine.apc_manager import auto_memory_gb
from yunshu_kv import disk_budget

GIB = 1 << 30
# (RAM in GiB, weights in GiB of a model that machine would plausibly run)
MACHINES = [(8, 2.5), (16, 5.0), (32, 17.0), (64, 17.0), (128, 17.0)]


@pytest.mark.parametrize(("ram", "weights"), MACHINES)
def test_apc_ram_budget_never_eats_the_model_or_the_os(ram, weights):
    gb = auto_memory_gb(ram * GIB, int(weights * GIB))
    assert gb == 0 or gb >= 1.0  # off, or big enough to matter
    assert gb <= 0.25 * ram  # never more than a quarter of the machine
    assert gb <= 32
    # weights + cache leave at least 4 GiB (a quarter of the machine, up to 16) to the OS,
    # activations and live KV
    assert ram - weights - gb >= min(16.0, max(4.0, 0.25 * ram)) - 1e-9


def test_eight_gb_machine_with_a_4b_model_reserves_nothing():
    assert auto_memory_gb(8 * GIB, 4 * GIB) == 0.0
    assert auto_memory_gb(8 * GIB, int(4.5 * GIB)) == 0.0


def test_eight_gb_machine_with_a_tiny_model_gets_at_most_a_gib_or_two():
    gb = auto_memory_gb(8 * GIB, 1 * GIB)
    assert 0 <= gb <= 2.0


def test_sixteen_gb_machine_keeps_a_modest_cache_for_a_small_model():
    gb = auto_memory_gb(16 * GIB, int(4.5 * GIB))
    assert 1.0 <= gb <= 4.0


def test_budget_grows_with_memory_and_shrinks_with_weights():
    sizes = [auto_memory_gb(r * GIB, 17 * GIB) for r in (8, 16, 32, 64, 128, 256)]
    assert sizes == sorted(sizes)
    assert sizes[-2:] == [32.0, 32.0]  # 128 GB and up: the tuned 27B default
    assert auto_memory_gb(64 * GIB, 8 * GIB) >= auto_memory_gb(64 * GIB, 40 * GIB)


def test_model_larger_than_the_machine_is_zero_not_negative():
    assert auto_memory_gb(16 * GIB, 40 * GIB) == 0.0


@pytest.mark.parametrize(("ram", "weights"), MACHINES)
def test_engine_uses_the_auto_budget_and_zero_is_honoured(
    tmp_path, monkeypatch, ram, weights
):
    from yunshu_engine.vlm_engine import VLMEngine

    (tmp_path / "w.safetensors").write_bytes(b"x")
    monkeypatch.setattr(apc_manager, "total_memory_bytes", lambda: ram * GIB)
    monkeypatch.setattr(
        apc_manager,
        "auto_memory_gb",
        lambda total, w: auto_memory_gb(total, int(weights * GIB)),
    )
    eng = VLMEngine.__new__(VLMEngine)
    eng._model_path = str(tmp_path)
    monkeypatch.delenv("YUNSHU_VLM_APC_MEMORY_GB", raising=False)
    auto = eng._apc_memory_gb()
    assert auto == auto_memory_gb(ram * GIB, int(weights * GIB))
    monkeypatch.setenv("YUNSHU_VLM_APC_MEMORY_GB", "0")
    assert eng._apc_memory_gb() == 0.0
    assert settings.get("YUNSHU_VLM_APC_MEMORY_GB") == 0.0


def _usage(total_gb, free_gb):
    u = namedtuple("u", "total used free")
    return lambda path: u(
        int(total_gb * GIB), int((total_gb - free_gb) * GIB), int(free_gb * GIB)
    )


@pytest.mark.parametrize(
    ("disk_gb", "expected"),
    [(128, 32.0), (256, 64.0), (512, 64.0), (2048, 64.0), (64, 16.0)],
)
def test_ssd_cap_is_a_quarter_of_the_volume_up_to_64_gib(
    tmp_path, monkeypatch, disk_gb, expected
):
    monkeypatch.setattr(disk_budget, "disk_usage", _usage(disk_gb, disk_gb / 2))
    assert disk_budget.auto_cap_gb(tmp_path) == expected
    assert disk_budget.resolve_cap_gb(None, tmp_path) == expected
    assert disk_budget.resolve_cap_gb(5.0, tmp_path) == 5.0  # an explicit cap wins
    assert disk_budget.resolve_cap_gb(0.0, tmp_path) == 0.0  # 0 = no configured cap


def test_apc_disk_default_is_auto_not_a_fixed_64(monkeypatch):
    assert settings.get("YUNSHU_VLM_APC_DISK_GB") is None


@pytest.mark.parametrize("disk_gb", [128, 256, 512, 2048])
def test_small_disk_never_spills_into_its_reserve(tmp_path, monkeypatch, disk_gb):
    """With little free space the effective cap follows the reserve, not the 64 GiB ceiling."""
    free = 30.0
    monkeypatch.setattr(disk_budget, "disk_usage", _usage(disk_gb, free))
    b = disk_budget.DiskBudget(
        tmp_path,
        cap_bytes=int(disk_budget.auto_cap_gb(tmp_path) * GIB),
        reserve_pct=10.0,
        reserve_min_bytes=20 * GIB,
    )
    reserve = max(disk_gb * 0.10, 20.0)
    cap = b.effective_cap()
    assert cap is not None
    assert cap / GIB <= max(0.0, free - reserve) + 1e-6
    assert not b.allow_write(int((free - reserve + 1) * GIB))
