"""CPU-only fakes: energy units, phase coverage, native ownership and failures."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from yunshu_engine.telemetry import apple, sampler


def reading(seconds=1.0):
    return apple.EnergyReading(
        seconds,
        {"GPU Energy": 10.0, "CPU Energy": 5.0, "ANE": 1.0, "DRAM": 2.0},
        [("OFF", 25), ("P1", 25), ("P2", 50)],
    )


def test_power_frequency_and_snapshot_copy():
    host = sampler.HostSampler()
    host.publish(reading(), 10, {"die_max_c": 70.0}, [300, 600])
    snap = host.snapshot()
    assert snap["watts"]["package"] == 18
    assert snap["gpu"]["frequency_mhz"] == 500
    assert snap["gpu"]["active_ratio"] == 0.75
    snap["watts"]["gpu"] = 100
    assert host.snapshot()["watts"]["gpu"] == 10
    assert sampler.frequency(reading().gpu_states, [300])[0] is None


def test_energy_windows_clipping_tail_and_receipt():
    host = sampler.HostSampler()
    host.publish(reading(), 10, {}, [300, 600])
    host.publish(reading(), 11, {}, [300, 600])
    stats = SimpleNamespace(
        t_admit=9.5,
        t_prefill_end=10,
        t_first=10.1,
        t_last=11.5,
        prompt_tokens=20,
        cached_tokens=10,
        generated=3,
    )
    receipt = host.receipt(stats)
    assert receipt["prefill"]["joules"] == 6
    assert receipt["prefill"]["joules_per_token"] == 0.6
    assert receipt["decode"]["joules"] == 18
    assert receipt["decode"]["joules_per_token"] == 6
    assert receipt["decode"]["extrapolated_s"] == 0.5
    assert receipt["decode"]["gpu_watts_mean"] == 10
    assert host.window(8, 10, 1)["joules"] is None
    assert host.window(10, 15, 1)["joules"] is None
    assert host.window(10, 11, 0)["joules_per_token"] is None
    host.record(receipt)
    assert (
        'yunshu_request_energy_joules_total{phase="decode"} 18.0' in host.prometheus()
    )


def test_unknown_power_and_gaps_never_become_zero():
    host = sampler.HostSampler()
    bad = reading()
    bad.watts.pop("DRAM")
    bad.watts["GPU Energy"] = float("nan")
    host.publish(bad, 10, {}, [])
    assert host.snapshot()["watts"]["gpu"] is None
    assert host.window(9, 10, 2)["joules"] is None
    host.publish(reading(), 12, {}, [300, 600])
    assert host.window(9, 12, 2)["joules"] is None
    assert "yunshu_gpu_watts 10.0" in host.prometheus()


def test_sampler_worker_owns_native_lifecycle_and_degrades():
    thread_names = []
    closed = []

    class Energy:
        def __init__(self):
            thread_names.append(threading.current_thread().name)

        def read(self):
            raise RuntimeError("fake counter failure")

        def close(self):
            closed.append(True)

    host = sampler.HostSampler(
        0.1, Energy, lambda: (_ for _ in ()).throw(RuntimeError("no HID")), lambda: []
    )
    host.start()
    host._stop.wait(0.15)
    host.close()
    assert thread_names == ["yunshu-host-telemetry"]
    assert closed == [True]
    assert host.snapshot()["reason"] == "fake counter failure"
    host.close()


class FakeCF:
    def __init__(self):
        self.released = []

    def CFRelease(self, ref):
        self.released.append(ref)

    def CFArrayGetCount(self, ref):
        return 1

    def CFArrayGetValueAtIndex(self, ref, index):
        return "service"


@pytest.mark.parametrize("fail", [False, True])
def test_hid_releases_service_property_event_on_failure(monkeypatch, fail):
    cf = FakeCF()

    def value(*args):
        if fail:
            raise RuntimeError("fake HID failure")
        return 70

    iok = SimpleNamespace(
        IOHIDEventSystemClientCopyServices=lambda client: "services",
        IOHIDServiceClientCopyProperty=lambda *args: "property",
        IOHIDServiceClientCopyEvent=lambda *args: "event",
        IOHIDEventGetFloatValue=value,
    )
    monkeypatch.setattr(apple, "_LIBS", (cf, iok, None))
    monkeypatch.setattr(apple, "_str", lambda ref: "tdie0")
    temp = apple.TemperatureSampler.__new__(apple.TemperatureSampler)
    temp._lock = threading.Lock()
    temp._client, temp._product = "client", "product"
    if fail:
        with pytest.raises(RuntimeError, match="fake HID"):
            temp.read()
    else:
        assert temp.read() == {"tdie0": 70}
    assert cf.released == ["property", "event", "services"]
    temp.close()
    temp.close()
    assert cf.released[-2:] == ["product", "client"]


def test_failed_delta_advances_baseline_and_releases(monkeypatch):
    cf = FakeCF()
    ior = SimpleNamespace(
        IOReportCreateSamples=lambda *a: 2, IOReportCreateSamplesDelta=lambda *a: None
    )
    monkeypatch.setattr(apple, "_LIBS", (cf, None, ior))
    energy = apple.EnergySampler.__new__(apple.EnergySampler)
    energy._lock = threading.Lock()
    energy._sub, energy._key, energy._prev, energy._subbed, energy._t = (
        3,
        4,
        1,
        apple._vp(5),
        0,
    )
    with pytest.raises(RuntimeError, match="Delta"):
        energy.read()
    assert cf.released == [1]
    energy.close()
    energy.close()
    assert cf.released == [1, 2, 3, 5, 4]


@pytest.mark.parametrize("unit,scale", [("mJ", 1e-3), ("uJ", 1e-6), ("nJ", 1e-9)])
def test_ioreport_unit_conversion(monkeypatch, unit, scale):
    cf = FakeCF()
    cf.CFDictionaryGetValue = lambda *a: "channels"
    ior = SimpleNamespace(
        IOReportChannelGetGroup=lambda ch: "Energy Model",
        IOReportChannelGetChannelName=lambda ch: "GPU Energy",
        IOReportChannelGetUnitLabel=lambda ch: unit,
        IOReportSimpleGetIntegerValue=lambda *a: 2000,
    )
    monkeypatch.setattr(apple, "_LIBS", (cf, None, ior))
    monkeypatch.setattr(apple, "_str", lambda ref: ref)
    energy = apple.EnergySampler.__new__(apple.EnergySampler)
    energy._key = 1
    assert energy._parse(None, 2).gpu_watts == 1000 * scale


def test_cli_unknown_view():
    from yunshu_cli.top import render

    assert render({"telemetry": sampler.unknown("off")}).row_count >= 7


def test_host_cache_keeps_os_fields_but_refreshes_telemetry(monkeypatch):
    from yunshu_gateway import host_state

    host = sampler.HostSampler()
    monkeypatch.setattr(sampler, "_SERVICE", host)
    monkeypatch.setattr(
        host_state, "_CACHED", {"object": "yunshu.host", "thermal": {"state": "normal"}}
    )
    monkeypatch.setattr(host_state, "_EXPIRES", float("inf"))
    host.publish(reading(), 10, {}, [300, 600])
    assert host_state.snapshot()["telemetry"]["watts"]["gpu"] == 10
    changed = reading()
    changed.watts["GPU Energy"] = 20
    host.publish(changed, 11, {}, [300, 600])
    assert host_state.snapshot()["telemetry"]["watts"]["gpu"] == 20
    assert host_state.snapshot()["thermal"] == {"state": "normal"}


def test_gateway_receipt_runstats_recent_and_counter(monkeypatch):
    from yunshu_gateway import x_yunshu as x

    host = sampler.HostSampler()
    host.publish(reading(), 10, {}, [300, 600])
    host.publish(reading(), 11, {}, [300, 600])
    monkeypatch.setattr(sampler, "_SERVICE", host)
    monkeypatch.setattr(x, "registry", type(x.registry)())
    stats = SimpleNamespace(
        t_admit=9,
        t_submit=9,
        t_prefill_end=10,
        t_first=10,
        t_last=11,
        prompt_tokens=20,
        cached_tokens=0,
        generated=2,
        spec_mode=None,
        cache_tier=None,
        cache_reload_ms=None,
        latency_marks={},
    )
    info = x.RequestInfo(
        "energy-fixture",
        "POST",
        "/v1/chat/completions",
        arrived=9,
        gen=SimpleNamespace(stats=stats),
    )
    out = x.build_stats(info)
    assert out["energy"]["decode"]["joules"] == 12
    assert stats.energy == out["energy"]
    x.record_done(info, out)
    assert x.registry.last()["energy"] == out["energy"]
    assert host._total == {"prefill": 12, "decode": 12}
    host.prometheus()
    host.prometheus()
    assert host._total == {"prefill": 12, "decode": 12}


def test_failed_energy_constructor_releases_all_temporaries(monkeypatch):
    cf = FakeCF()
    owned = []

    def create(*args):
        owned.append(len(owned) + 1)
        return owned[-1]

    cf.CFStringCreateWithCString = create

    def channels(group, subgroup, *args):
        return None if subgroup else create()

    ior = SimpleNamespace(IOReportCopyChannelsInGroup=channels)
    monkeypatch.setattr(apple, "_LIBS", (cf, None, ior))
    with pytest.raises(RuntimeError, match="GPU Stats"):
        apple.EnergySampler()
    assert sorted(cf.released) == owned
