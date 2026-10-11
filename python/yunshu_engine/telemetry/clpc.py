# Upstream (MIT): vladkens/macmon src/clpc.rs + src/sources.rs
# @ 7df49f55d9a1b9072e31fc8ba991abda84593563. Derived catalog and CF descriptors.
# Copyright (c) 2024 vladkens; full MIT terms in THIRD_PARTY_NOTICES.md.
"""Qualified macOS 27 CLPC counters; unknown OS/driver IDs are never guessed."""

from __future__ import annotations

import ctypes as c

from . import apple

_KEYS: dict[str, tuple[int, int, int]] = {
    "com.apple.driver.AppleT6000CLPCv3": (0x4747_9059, 0x4519_BF1E, 0x9AAA_17F8),
    "com.apple.driver.AppleT6002CLPC": (0xB501_816B, 0x9E2C_3E8B, 0x14D5_A574),
    "com.apple.driver.AppleT6020CLPC": (0xE301_CD75, 0x8F34_CC7A, 0x7415_FB0F),
    "com.apple.driver.AppleT6022CLPC": (0x76F5_FD21, 0x9062_4499, 0x3C09_7307),
    "com.apple.driver.AppleT6030CLPC": (0xBE03_A01A, 0x472D_6C9C, 0xB899_0C38),
    "com.apple.driver.AppleT6031CLPC": (0xD81E_0084, 0x1085_F98B, 0x313A_3BE1),
    "com.apple.driver.AppleT6032CLPC": (0xF4A2_1308, 0x85E8_0513, 0x63DC_8E25),
    "com.apple.driver.AppleT6041CLPC": (0x263C_F1F0, 0x22E4_CF59, 0x7849_FB5C),
    "com.apple.driver.AppleT6050CLPC": (0xD3AB_60BB, 0x27CD_4CE5, 0x8C0F_1C59),
    "com.apple.driver.AppleT6050dCLPC": (0x18B3_010E, 0xEDEC_A11A, 0x188B_D883),
    "com.apple.driver.AppleT8103CLPCv3": (0x2CDD_31F2, 0x21BE_CF42, 0x0667_FB81),
    "com.apple.driver.AppleT8112CLPC": (0xB543_5137, 0x638D_9D52, 0xEA08_9C36),
    "com.apple.driver.AppleT8122CLPC": (0x54B5_1C99, 0xCE2D_D9DC, 0x960A_FE2F),
    "com.apple.driver.AppleT8132CLPC": (0x4A85_4D94, 0x3B3E_3B79, 0x1FF4_A9EC),
    "com.apple.driver.AppleT8140CLPC": (0x9BF5_4436, 0xEF70_589C, 0x569D_941B),
    "com.apple.driver.AppleT8142CLPC": (0x4370_8744, 0x6A01_0933, 0xBC5E_FDF0),
    "com.apple.driver.AppleT8152CLPC": (0x485D_4BFC, 0x1178_4654, 0x9380_C6BB),
}


def energy_channels(bundle: str, version: str) -> list[tuple[int, str]]:
    if version.split(".")[0] != "27" or bundle not in _KEYS:
        return []
    return [
        ((index << 32) | key, name)
        for index, key, name in zip(
            (16, 25, 24),
            _KEYS[bundle],
            ("CPU Energy", "GPU Energy", "ANE"),
            strict=True,
        )
    ]


def _channel(driver: int, report_id: int, name: str):
    cf = apple._LIBS[0]
    refs = []

    def own(ref, label):
        refs.append(apple._owned(ref, label))
        return ref

    def string(value):
        return own(apple._cfs(value), "CLPC string")

    def number(value):
        scalar = c.c_int64(value)
        return own(cf.CFNumberCreate(None, 4, c.byref(scalar)), "CLPC number")

    def dictionary(pairs):
        keys = (apple._vp * len(pairs))(*(string(key) for key, _ in pairs))
        values = (apple._vp * len(pairs))(*(value for _, value in pairs))
        return own(
            cf.CFDictionaryCreate(
                None, keys, values, len(pairs), *apple._DICT_CALLBACKS
            ),
            "CLPC dictionary",
        )

    try:
        values = (apple._vp * 3)(
            number(report_id), number(1 | (2 << 16) | (1 << 32)), string(name)
        )
        legend = own(
            cf.CFArrayCreate(None, values, 3, apple._ARRAY_CALLBACKS), "CLPC legend"
        )
        info = dictionary([("IOReportChannelUnit", number((3 << 56) | (118 << 32)))])
        desc = dictionary(
            [
                ("DriverID", number(driver)),
                ("DriverName", string("AppleCLPC")),
                ("IOReportGroupName", string("CLPC")),
                ("IOReportSubGroupName", string("Energy Counters")),
                ("IOReportChannelInfo", info),
                ("LegendChannel", legend),
            ]
        )
        # IOReport getters cache properties, requiring a mutable outer dictionary.
        return apple._owned(
            cf.CFDictionaryCreateMutableCopy(None, 0, desc), "CLPC mutable descriptor"
        )
    finally:
        apple._release(*reversed(refs))


def augment(desired, version: str) -> None:
    """Atomically append qualified descriptors; release all CF/IOKit ownership."""
    if version.split(".")[0] != "27":
        raise RuntimeError(f"no qualified CLPC catalog for macOS {version}")
    cf, iok, _ = apple._LIBS
    key = bundle_key = selected = None
    iterator = c.c_uint32()
    try:
        key = apple._cfs("IOReportChannels")
        original = apple._owned(
            cf.CFDictionaryGetValue(desired, key), "IOReportChannels"
        )  # borrowed
        selected = apple._owned(
            cf.CFArrayCreateMutableCopy(None, 0, original), "CLPC selected channels"
        )
        bundle_key = apple._cfs("CFBundleIdentifier")
        matching = apple._owned(
            iok.IOServiceMatching(b"AppleCLPC"), "IOServiceMatching(AppleCLPC)"
        )
        try:
            rc = iok.IOServiceGetMatchingServices(0, matching, c.byref(iterator))
        except Exception:
            apple._release(matching)  # Python failed before the consuming C call.
            raise
        # IOServiceGetMatchingServices consumes matching even on IOReturn failure.
        if rc:
            raise RuntimeError(f"AppleCLPC service lookup failed: {rc}")
        if not iterator.value:
            raise RuntimeError("no AppleCLPC services")
        seen = set()
        while entry := iok.IOIteratorNext(iterator):
            try:
                driver = c.c_uint64()
                if iok.IORegistryEntryGetRegistryEntryID(entry, c.byref(driver)):
                    continue
                prop = iok.IORegistryEntryCreateCFProperty(entry, bundle_key, None, 0)
                try:
                    bundle = apple._str(prop) or ""
                finally:
                    apple._release(prop)
                for report_id, name in energy_channels(bundle, version):
                    identity = (driver.value, report_id & 0xFFFFFFFF)
                    if identity in seen:
                        continue
                    desc = _channel(driver.value, report_id, name)
                    try:
                        cf.CFArrayAppendValue(selected, desc)
                    finally:
                        apple._release(desc)
                    seen.add(identity)
            finally:
                iok.IOObjectRelease(entry)
        if not seen:
            raise RuntimeError("no qualified AppleCLPC driver counters")
        cf.CFDictionarySetValue(desired, key, selected)
    finally:
        if iterator.value:
            iok.IOObjectRelease(iterator)
        apple._release(selected, bundle_key, key)
