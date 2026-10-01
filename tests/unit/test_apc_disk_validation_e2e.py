"""APC SSD reload through the real mlx-vlm BatchGenerator on a tiny hybrid Qwen3.5: a real
checkpoint passes the model check (no false rejection), and a foreign or corrupt file goes
cold with the exact output of a cold run and is deleted (R06 / R07)."""

import os
import random

import pytest

mx = pytest.importorskip("mlx.core")

from tests.unit.test_apc_manager_e2e import (  # noqa: E402
    RunStats,
    VLMBatchRunner,
    YunshuAPCManager,
    _processor,
    _run,
    model,  # noqa: F401  (fixture)
)
from yunshu_engine.apc_manager import SpillDiskStore, check_loaded_cache  # noqa: E402


def _spilled(model, tmp_path, validator):  # noqa: F811
    """Conversation ``a`` pushed out to the SSD by ``b``; a fresh runner on the same dir."""
    rnd = random.Random(9)
    a = [rnd.randrange(3, 800) for _ in range(260)]
    b = [rnd.randrange(3, 800) for _ in range(260)]
    a2 = a + [rnd.randrange(3, 800) for _ in range(40)]

    def runner_for():
        disk = SpillDiskStore(
            tmp_path, namespace="e2e", num_workers=1, max_bytes=1 << 30
        )
        disk.validator = validator
        mgr = YunshuAPCManager(
            num_blocks=64,
            block_size=16,
            disk=disk,
            max_entries=1,
            overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 0},
        )
        return VLMBatchRunner(
            model, processor=_processor(), apc_manager=mgr, apc_semantic_hash=0
        )

    first = runner_for()
    _run(first, a)
    _run(first, b)
    first.apc_manager.disk.flush()
    return runner_for(), a2


def _cold(model, ids):  # noqa: F811
    return _run(VLMBatchRunner(model, processor=_processor()), ids)[0]


def test_validator_accepts_real_states_and_reload_stays_lossless(model, tmp_path):  # noqa: F811
    lm = model.language_model
    seen = []

    def validator(tokens, cache):
        ok = check_loaded_cache(
            tokens,
            cache,
            lm.make_cache(),
            lm.args.num_key_value_heads,
            lm.args.head_dim,
        )
        seen.append(ok)
        return ok

    runner, a2 = _spilled(model, tmp_path, validator)
    out, st = _run(runner, a2)
    assert seen == [True]  # the real hybrid checkpoint passes its own model's check
    assert st.cache_tier == "ssd" and st.cached_tokens > 0
    assert out == _cold(model, a2)


def test_foreign_geometry_on_disk_goes_cold_and_is_deleted(model, tmp_path):  # noqa: F811
    lm = model.language_model

    def validator(tokens, cache):  # as if the file came from a model with 2 kv heads
        return check_loaded_cache(tokens, cache, lm.make_cache(), 2, 256)

    runner, a2 = _spilled(model, tmp_path, validator)
    disk = runner.apc_manager.disk
    files = list(disk.dir.glob("exact_*.safetensors"))
    assert files
    out, st = _run(runner, a2)
    assert st.cache_tier == "none" and st.cached_tokens == 0
    assert out == _cold(model, a2)  # a cold prefill, never the foreign state
    assert disk.invalidated >= 1
    assert not any(f.exists() for f in files)


@pytest.mark.parametrize("damage", ["truncate", "garbage", "empty"])
def test_corrupt_checkpoint_files_fall_back_cold(model, tmp_path, damage):  # noqa: F811
    runner, a2 = _spilled(model, tmp_path, None)
    disk = runner.apc_manager.disk
    files = list(disk.dir.glob("exact_*.safetensors"))
    assert files
    for f in files:
        data = f.read_bytes()
        f.write_bytes(
            {
                "truncate": data[: len(data) * 2 // 3],
                "garbage": os.urandom(len(data)),
                "empty": b"",
            }[damage]
        )
    disk._header_cache.clear()
    out, st = _run(runner, a2)
    assert st.cached_tokens == 0
    assert out == _cold(model, a2)
    assert isinstance(st, RunStats)
