"""Memory admission: load plan, gate decisions, runner hold, APC shedding (simulated memory)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from yunshu_engine import memory_admission as ma

GIB = 1 << 30
CONFIG = {
    "text_config": {
        "num_hidden_layers": 64,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        * 16,
        "num_key_value_heads": 4,
        "num_attention_heads": 24,
        "head_dim": 256,
        "hidden_size": 5120,
    }
}


def host(avail_gib, swap_gib=0.0, level=1, total=128):
    return ma.HostMemory(total * GIB, int(avail_gib * GIB), int(swap_gib * GIB), level)


def make_gate(readings, reserve_gib=6.0):
    state = {"h": readings}
    gate = ma.MemoryGate(
        kv_per_token=ma.kv_bytes_per_token(CONFIG),
        act_per_token=ma.activation_bytes_per_token(CONFIG),
        reserve_bytes=int(reserve_gib * GIB),
        attn_heads=24,
        read=lambda: state["h"],
        ttl_s=0.0,
    )
    gate.state = state
    return gate


def test_kv_bytes_per_token_counts_full_attention_layers_only():
    # 16 full layers * 2 (K,V) * 4 kv heads * 256 * 2 bytes
    assert ma.kv_bytes_per_token(CONFIG) == 16 * 2 * 4 * 256 * 2
    assert ma.kv_bytes_per_token({}) == 0


def test_parse_vm_stat():
    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                               100.\n"
        "Pages inactive:                            50.\n"
        "Pages speculative:                         10.\n"
        "Pages purgeable:                            5.\n"
    )
    assert ma.parse_vm_stat(text) == 165 * 16384
    assert ma.parse_vm_stat("garbage") is None


def test_plan_load_shrinks_apc_for_large_weights():
    total = 128 * GIB
    plan = ma.plan_load(
        total, int(97.3 * GIB), 32.0, kv_per_token=ma.kv_bytes_per_token(CONFIG)
    )
    # 128 - 97.3 - reserve 6.4 - min working set leaves ~21 GiB at most; 32 must shrink
    assert 0 <= plan.apc_gb < 32.0
    assert plan.error is None
    assert plan.notes and "lowered" in plan.notes[0]


def test_plan_load_keeps_budget_when_it_fits():
    plan = ma.plan_load(128 * GIB, 16 * GIB, 32.0, kv_per_token=1 << 16)
    assert plan.apc_gb == 32.0 and not plan.notes


def test_plan_load_refuses_when_minimum_does_not_fit_and_names_view(tmp_path):
    raw = tmp_path / "m"
    raw.mkdir()
    (tmp_path / "m-ple-ssd").mkdir()
    plan = ma.plan_load(64 * GIB, 70 * GIB, 8.0, model_path=raw)
    assert plan.error and "m-ple-ssd" in plan.error and plan.apc_gb == 0.0


def test_estimate_counts_kv_and_chunk_attention_workspace():
    gate = ma.MemoryGate(
        kv_per_token=1000, act_per_token=10, attn_heads=24, reserve_bytes=0
    )
    n = 52_000
    want = (n + 100) * 1000 + 2048 * 10 + 2048 * n * 24 * 2
    assert gate.estimate(n, 100) == want
    # a cached prefix shrinks the chunk, never below zero
    assert gate.estimate(n, 100, cached_tokens=n) == (n + 100) * 1000


def test_gate_idle_server_always_admits():
    gate = make_gate(host(0.5))
    assert gate.decide(50 * GIB, 0, any_active=False).admit


def test_gate_holds_when_no_headroom_and_reports_reason():
    gate = make_gate(host(8.0))
    d = gate.decide(4 * GIB, 0, any_active=True)  # 8 - 4 < 6 reserve
    assert not d.admit and d.reason == "memory" and "GiB" in d.detail
    assert gate.decide(1 * GIB, 0, any_active=True).admit  # 8 - 1 >= 6


def test_gate_counts_growth_of_running_requests():
    gate = make_gate(host(20.0))
    assert gate.decide(2 * GIB, 0, any_active=True).admit
    assert not gate.decide(2 * GIB, 13 * GIB, any_active=True).admit  # 20 - 13 - 2 < 6


def test_gate_holds_on_critical_pressure_and_swap_growth():
    gate = make_gate(host(50.0, level=4))
    assert not gate.decide(GIB, 0, True).admit
    gate = make_gate(host(50.0, swap_gib=1.0))
    assert gate.decide(GIB, 0, True).admit
    gate.state["h"] = host(50.0, swap_gib=2.0)  # +1 GiB over the lowest seen
    d = gate.decide(GIB, 0, True)
    assert not d.admit and "swap" in d.detail


def test_gate_unreadable_host_admits():
    gate = make_gate(None)
    assert gate.decide(100 * GIB, 0, True).admit


def test_ceiling_bounds_available():
    gate = ma.MemoryGate(
        kv_per_token=1,
        act_per_token=1,
        reserve_bytes=GIB,
        read=lambda: host(100.0),
        ceiling_free=lambda: 3 * GIB,
        ttl_s=0.0,
    )
    assert gate.available() == 3 * GIB
    assert not gate.decide(2 * GIB + 1, 0, True).admit


def test_pressure_shortfall():
    gate = make_gate(host(10.0))
    assert gate.pressure_shortfall() == 0
    gate.state["h"] = host(4.0)
    assert gate.pressure_shortfall() == 2 * GIB
    gate.state["h"] = host(10.0, level=2)
    assert gate.pressure_shortfall() == 3 * GIB


# --------------------------------------------------------------- runner integration
def _job(n_ids=1000, max_tokens=100):
    stats = SimpleNamespace(
        queue_reason=None,
        queue_detail=None,
        extra={},
        cached_tokens=0,
        prefill_done=0,
        generated=0,
    )
    return SimpleNamespace(
        ids=[0] * n_ids,
        max_tokens=max_tokens,
        stats=stats,
        mem_est=0,
        mem_held_at=0.0,
        terminal=False,
    )


def _runner(gate):
    from yunshu_engine.vlm_batch_runner import VLMBatchRunner

    r = VLMBatchRunner.__new__(VLMBatchRunner)
    r.memory_gate = gate
    r._last_shed = 0.0
    return r


def test_runner_holds_then_admits_and_reports_wait():
    gate = make_gate(host(7.0))
    r = _runner(gate)
    running = _job(10_000, 2000)
    assert r._memory_wait(running, []) is False  # idle: admitted
    waiting = _job(20_000, 1000)  # ~0.5 GiB KV + 2.3 GiB scores: 7 - 2.8 < 6
    assert r._memory_wait(waiting, [running]) is True
    assert (
        waiting.stats.queue_reason == "memory" and "GiB" in waiting.stats.queue_detail
    )
    assert gate.holds == 1
    r._memory_wait(waiting, [running])
    assert gate.holds == 1  # one request, counted once
    gate.state["h"] = host(40.0)
    assert r._memory_wait(waiting, [running]) is False
    assert waiting.stats.queue_reason is None
    assert waiting.stats.extra["memory_wait_ms"] >= 0


def test_runner_without_gate_never_holds():
    assert _runner(None)._memory_wait(_job(), [_job()]) is False


def test_shed_under_pressure_calls_apc_and_counts():
    gate = make_gate(host(2.0))
    r = _runner(gate)
    freed = []
    r.apc_manager = SimpleNamespace(shed_ram=lambda want: freed.append(want) or want)
    r._shed_under_pressure()
    assert freed == [4 * GIB] and gate.sheds == 1 and gate.shed_bytes == 4 * GIB
    r._shed_under_pressure()  # rate limited to once a second
    assert len(freed) == 1


def test_shed_idle_when_headroom_is_fine():
    gate = make_gate(host(30.0))
    r = _runner(gate)
    r.apc_manager = SimpleNamespace(shed_ram=lambda want: pytest.fail("must not shed"))
    r._shed_under_pressure()
    assert gate.sheds == 0
    time.sleep(0)


def test_apc_shed_ram_drops_oldest_exact_entries_first(monkeypatch):
    import collections
    import threading

    import mlx_vlm.apc as apc

    from yunshu_engine.apc_manager import YunshuAPCManager

    monkeypatch.setattr(apc, "_cache_nbytes", lambda cache: cache["n"])
    m = YunshuAPCManager.__new__(YunshuAPCManager)
    m.lock = threading.RLock()
    m._exact_cache = collections.OrderedDict(
        (k, SimpleNamespace(prompt_cache={"n": GIB})) for k in "abc"
    )
    m._free_head = None
    m.stats = SimpleNamespace(evictions=0, memory_evictions=0)
    assert m.shed_ram(int(1.5 * GIB)) == 2 * GIB
    assert list(m._exact_cache) == ["c"]
    assert m.stats.memory_evictions == 2
    assert m.shed_ram(10 * GIB) == GIB and not m._exact_cache


def test_idle_request_that_exceeds_the_metal_working_set_is_refused():
    gate = ma.MemoryGate(
        kv_per_token=1,
        act_per_token=1,
        reserve_bytes=GIB,
        read=lambda: host(100.0),
        ceiling_free=lambda: 10 * GIB,
        ttl_s=0.0,
    )
    d = gate.decide(11 * GIB, 0, any_active=False)
    assert (
        d.refuse and not d.admit and d.reason == "memory_limit" and "Metal" in d.detail
    )
    assert gate.decide(9 * GIB, 0, any_active=False).admit
    # with other requests running the same request waits instead of being refused
    held = gate.decide(11 * GIB, 0, any_active=True)
    assert not held.admit and not held.refuse and held.reason == "memory"


def test_runner_raises_a_memory_error_for_a_refusal():
    gate = ma.MemoryGate(
        kv_per_token=1000,
        act_per_token=0,
        reserve_bytes=0,
        read=lambda: host(100.0),
        ceiling_free=lambda: GIB,
        ttl_s=0.0,
    )
    r = _runner(gate)
    with pytest.raises(ma.MemoryAdmissionError) as e:
        r._memory_wait(_job(2_000_000, 0), [])
    assert isinstance(e.value, MemoryError) and "Metal" in str(e.value)


def test_chat_reports_the_refusal_reason():
    from yunshu_gateway.routers import chat

    exc = ma.MemoryAdmissionError("needs 20.00 GiB")
    assert "needs 20.00 GiB" in chat._memory_error_message(exc)
    assert chat._memory_error_code(exc) == "memory_limit"
    assert chat._memory_error_message(MemoryError()) == "Out of GPU memory"
    assert chat._memory_error_code(MemoryError()) == "oom"


def test_refusal_probe_judges_the_answer():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "memadmit_refuse", "scripts/research/memadmit_refuse.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memadmit_refuse"] = mod
    spec.loader.exec_module(mod)
    ok = {"error": {"code": "memory_limit"}}
    assert mod.judge(507, ok, True) == []
    assert len(mod.judge(500, {"error": {"code": "oom"}}, False)) == 3
