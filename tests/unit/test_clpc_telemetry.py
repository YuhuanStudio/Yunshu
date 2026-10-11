"""Native CLPC descriptor ABI and ownership, exercised without hardware."""

from __future__ import annotations

import ctypes as c
from types import SimpleNamespace

import pytest

from yunshu_engine.telemetry import apple, clpc


class CF:
    def __init__(self):
        self.objects = {}
        self.released = []
        self.fail_copy = False

    def create(self, value):
        ref = len(self.objects) + 1
        self.objects[ref] = value
        return ref

    def CFRelease(self, ref):
        assert ref not in self.released, f"double release {ref}"
        self.released.append(ref)

    def CFStringCreateWithCString(self, _, value, encoding):
        return self.create(value.decode())

    def CFStringGetCString(self, ref, buf, size, encoding):
        raw = self.objects[ref].encode() + b"\0"
        c.memmove(buf, raw, len(raw))
        return True

    def CFNumberCreate(self, _, kind, value):
        assert kind == 4
        return self.create(c.cast(value, c.POINTER(c.c_int64)).contents.value)

    def CFArrayCreate(self, _, values, count, callback):
        return self.create([values[i] for i in range(count)])

    def CFDictionaryCreate(self, _, keys, values, count, *callbacks):
        return self.create({self.objects[keys[i]]: values[i] for i in range(count)})

    def CFDictionaryCreateMutableCopy(self, _, capacity, ref):
        return None if self.fail_copy else self.create(dict(self.objects[ref]))

    def CFDictionaryGetValue(self, ref, key):
        return self.objects[ref].get(self.objects[key])

    def CFArrayCreateMutableCopy(self, _, capacity, ref):
        return self.create(list(self.objects[ref]))

    def CFArrayAppendValue(self, array, value):
        self.objects[array].append(value)

    def CFDictionarySetValue(self, ref, key, value):
        self.objects[ref][self.objects[key]] = value


@pytest.fixture
def native(monkeypatch):
    cf = CF()
    monkeypatch.setattr(apple, "_LIBS", (cf, None, None))
    monkeypatch.setattr(apple, "_DICT_CALLBACKS", (None, None))
    monkeypatch.setattr(apple, "_ARRAY_CALLBACKS", None)
    return cf


def test_catalog_full_counter_ids_are_version_and_driver_qualified():
    assert clpc.energy_channels("com.apple.driver.AppleT8142CLPC", "27.0.1") == [
        (0x0000001043708744, "CPU Energy"),
        (0x000000196A010933, "GPU Energy"),
        (0x00000018BC5EFDF0, "ANE"),
    ]
    assert not clpc.energy_channels("unknown", "27.0.1")
    assert not clpc.energy_channels("com.apple.driver.AppleT8142CLPC", "26.0")
    assert not clpc.energy_channels("com.apple.driver.AppleT8142CLPC", "28.0")


def test_descriptor_preserves_uint64_id_unit_and_cf_ownership(native):
    cf = native
    channel = clpc._channel(0x100000428, 0x0000001043708744, "CPU Energy")
    desc = cf.objects[channel]
    assert cf.objects[desc["DriverID"]] == 0x100000428
    assert cf.objects[desc["IOReportGroupName"]] == "CLPC"
    legend = cf.objects[desc["LegendChannel"]]
    assert [cf.objects[ref] for ref in legend] == [
        0x0000001043708744,
        1 | (2 << 16) | (1 << 32),
        "CPU Energy",
    ]
    info = cf.objects[desc["IOReportChannelInfo"]]
    assert cf.objects[info["IOReportChannelUnit"]] == (3 << 56) | (118 << 32)
    cf.CFRelease(channel)
    assert sorted(cf.released) == sorted(cf.objects)


def test_null_descriptor_releases_every_created_cf_object(native):
    native.fail_copy = True
    with pytest.raises(RuntimeError, match="mutable descriptor"):
        clpc._channel(1, 2, "CPU Energy")
    assert sorted(native.released) == sorted(native.objects)


@pytest.mark.parametrize("qualified", [True, False])
def test_augmentation_consumes_matching_and_releases_registry_objects(
    native, monkeypatch, qualified
):
    cf = native
    source = cf.create([])
    desired = cf.create({"IOReportChannels": source})
    before = set(cf.objects)
    entries = iter([10001, 0])
    released_io = []

    def match_services(_, matching, iterator):
        cf.CFRelease(matching)  # IOKit consumes the owned CF dictionary.
        c.cast(iterator, c.POINTER(c.c_uint32)).contents.value = 10000
        return 0

    def registry_id(entry, pointer):
        c.cast(pointer, c.POINTER(c.c_uint64)).contents.value = 0x100000428
        return 0

    iok = SimpleNamespace(
        IOServiceMatching=lambda _: cf.create({}),
        IOServiceGetMatchingServices=match_services,
        IOIteratorNext=lambda _: next(entries),
        IORegistryEntryGetRegistryEntryID=registry_id,
        IORegistryEntryCreateCFProperty=lambda *a: cf.create(
            "com.apple.driver.AppleT8142CLPC" if qualified else "unknown"
        ),
        IOObjectRelease=lambda ref: released_io.append(
            ref.value if hasattr(ref, "value") else ref
        ),
    )
    monkeypatch.setattr(apple, "_LIBS", (cf, iok, None))
    if qualified:
        clpc.augment(desired, "27.0.1")
        assert len(cf.objects[cf.objects[desired]["IOReportChannels"]]) == 3
    else:
        with pytest.raises(RuntimeError, match="no qualified"):
            clpc.augment(desired, "27.0.1")
        assert cf.objects[desired]["IOReportChannels"] == source
    assert sorted(released_io) == [10000, 10001]
    assert set(cf.released) == set(cf.objects) - before


def parse(monkeypatch, channels, untrusted=False):
    cf = SimpleNamespace(
        CFDictionaryGetValue=lambda *a: channels,
        CFArrayGetCount=len,
        CFArrayGetValueAtIndex=lambda rows, i: rows[i],
    )
    ior = SimpleNamespace(
        IOReportChannelGetGroup=lambda ch: ch[0],
        IOReportChannelGetChannelName=lambda ch: ch[1],
        IOReportChannelGetUnitLabel=lambda ch: ch[2],
        IOReportSimpleGetIntegerValue=lambda ch, _: ch[3],
    )
    monkeypatch.setattr(apple, "_LIBS", (cf, None, ior))
    monkeypatch.setattr(apple, "_str", lambda ref: ref)
    energy = apple.EnergySampler.__new__(apple.EnergySampler)
    energy._key = 1
    energy._untrusted_energy_model = untrusted
    return energy._parse(None, 1)


def test_family_channels_and_clpc_preference_without_double_count(monkeypatch):
    reading = parse(
        monkeypatch,
        [
            ("Energy Model", "DIE_0_CPU Energy", "mJ", 1000),
            ("Energy Model", "DIE_1_CPU Energy", "mJ", 2000),
            ("Energy Model", "ANE0", "mJ", 100),
            ("Energy Model", "ANE1", "mJ", 200),
            ("Energy Model", "DRAM0", "mJ", 1000),
            ("Energy Model", "DRAM1", "mJ", 2000),
            ("CLPC", "CPU Energy", "nJ", 5000000000),
            ("CLPC", "ANE", "nJ", 0),
        ],
        True,
    )
    assert reading.watts == {"CPU Energy": 5, "ANE": 0, "DRAM": 3}
    assert not reading.reasons


def test_unqualified_frozen_or_negative_counters_stay_unknown(monkeypatch):
    reading = parse(
        monkeypatch,
        [
            ("Energy Model", "CPU Energy", "mJ", 0),
            ("Energy Model", "GPU Energy", "mJ", 30000),
            ("Energy Model", "ANE0", "mJ", 0),
            ("CLPC", "CPU Energy", "nJ", -1),
        ],
        True,
    )
    assert reading.watts == {"GPU Energy": 30}
    assert "may freeze" in reading.reasons["CPU Energy"]
    assert "may freeze" in reading.reasons["ANE"]
    reading = parse(
        monkeypatch,
        [("Energy Model", "ANE0", "mJ", 100), ("Energy Model", "ANE1", "invalid", 100)],
        False,
    )
    assert "ANE" not in reading.watts


def test_hid_mtr_fallback_does_not_mix_sensor_families():
    temp = apple.TemperatureSampler.__new__(apple.TemperatureSampler)
    calls = []

    def read(match):
        calls.append(match)
        return (
            {"gas gauge battery": 30}
            if "tdie" in match
            else {"pACC MTR Temp Sensor0": 70, "GPU MTR Temp Sensor0": 60}
        )

    temp.read = read
    assert temp.die_summary() == {"die_max_c": 70, "die_mean_c": 65, "battery_c": 30}
    assert len(calls) == 2
    temp.read = lambda match: {"tdie0": 80, "tdie1": 70}
    assert temp.die_summary()["die_mean_c"] == 75


def test_matching_error_consumes_dictionary_and_cleans_cf_temporaries(
    native, monkeypatch
):
    cf = native
    source = cf.create([])
    desired = cf.create({"IOReportChannels": source})
    before = set(cf.objects)

    def fail(_, matching, iterator):
        cf.CFRelease(matching)
        return 5

    iok = SimpleNamespace(
        IOServiceMatching=lambda _: cf.create({}), IOServiceGetMatchingServices=fail
    )
    monkeypatch.setattr(apple, "_LIBS", (cf, iok, None))
    with pytest.raises(RuntimeError, match="lookup failed: 5"):
        clpc.augment(desired, "27.0.1")
    assert set(cf.released) == set(cf.objects) - before


def test_unknown_os_never_trusts_legacy_cpu_ane_and_constructor_cleans(
    native, monkeypatch
):
    cf = native
    observed = []

    def augment(desired, version):
        observed.append(version)
        raise RuntimeError("unknown OS catalog")

    def subscription(_, desired, subbed, *args):
        c.cast(subbed, c.POINTER(apple._vp)).contents.value = cf.create({})
        return cf.create({})

    ior = SimpleNamespace(
        IOReportCopyChannelsInGroup=lambda *a: cf.create({}),
        IOReportMergeChannels=lambda *a: None,
        IOReportCreateSubscription=subscription,
        IOReportCreateSamples=lambda *a: cf.create({}),
    )
    monkeypatch.setattr(apple, "_LIBS", (cf, None, ior))
    monkeypatch.setattr(apple.platform, "mac_ver", lambda: ("", (), ""))
    monkeypatch.setattr(clpc, "augment", augment)
    energy = apple.EnergySampler()
    assert energy._untrusted_energy_model is True
    assert energy._clpc_reason == "unknown OS catalog"
    assert observed == [""]
    energy.close()
    energy.close()
    assert sorted(cf.released) == sorted(cf.objects)
