"""Optional native ANE research runtime; never imported by the serving path.

Compilation and caches require an explicit directory. Importing this module does
not load MLX, Metal or Apple's private ANE framework. Missing tools, selectors,
changed signatures, compile/load/evaluation failures all make a session unavailable.
"""

from __future__ import annotations

import ctypes as c
import hashlib
import os
import platform
import subprocess
import threading
from pathlib import Path

_BUILD_LOCK = threading.Lock()


def build(cache: Path) -> Path:
    """Build a source-addressed dylib on first use, atomically, without Xcode."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise OSError("native ANE needs Apple Silicon macOS")
    source = Path(__file__).parent
    files = sorted([*source.glob("*.hpp"), *source.glob("*.mm")])
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(platform.mac_ver()[0].encode())
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"libyunshu_ane-{digest.hexdigest()[:20]}.dylib"
    with _BUILD_LOCK:
        if not target.exists():
            partial = target.with_suffix(f".{os.getpid()}.partial.dylib")
            try:
                subprocess.run(
                    [
                        "clang++",
                        "-std=c++20",
                        "-fobjc-arc",
                        "-fblocks",
                        "-dynamiclib",
                        "-O2",
                        "-framework",
                        "Foundation",
                        "-framework",
                        "Metal",
                        "-framework",
                        "IOSurface",
                        *[str(path) for path in files if path.suffix == ".mm"],
                        "-o",
                        str(partial),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
    return target


def _library(cache: Path):
    lib = c.CDLL(str(build(cache)))
    lib.ys_ane_error.restype = c.c_char_p
    lib.ys_ane_available.restype = c.c_int
    lib.ys_ane_create.argtypes = [
        c.c_char_p,
        c.c_void_p,
        c.c_size_t,
        c.c_char_p,
        c.POINTER(c.c_uint32),
        c.POINTER(c.c_uint32),
        c.c_uint32,
        c.c_uint32,
        c.c_uint32,
    ]
    lib.ys_ane_create.restype = c.c_void_p
    lib.ys_ane_data.argtypes = [c.c_void_p, c.c_uint32]
    lib.ys_ane_data.restype = c.c_void_p
    lib.ys_ane_stride.argtypes = [c.c_void_p, c.c_uint32]
    lib.ys_ane_stride.restype = c.c_uint32
    lib.ys_ane_run.argtypes = [c.c_void_p, c.c_uint32, c.POINTER(c.c_double)]
    lib.ys_ane_run.restype = c.c_int
    lib.ys_ane_destroy.argtypes = [c.c_void_p]
    lib.ys_ane_destroy.restype = None
    return lib


class NativeSession:
    """Fixed MIL program/surfaces. Use try_create and run's None for GPU fallback.

    No writable surface escapes the session: copy_input/output hold its lock and
    cannot race evaluation or destruction. Timings from run exclude these copies.
    """

    def __init__(self, lib, handle, shapes):
        self._lib, self._handle, self._shapes = lib, handle, shapes
        self._lock = threading.Lock()
        self.reason: str | None = None

    @classmethod
    def try_create(
        cls,
        mil: str,
        weights: bytes,
        inputs: list[tuple[int, int]],
        output: tuple[int, int],
        cache: Path,
    ) -> tuple[NativeSession | None, str | None]:
        try:
            if (
                not inputs
                or len(inputs) > 32
                or any(
                    len(s) != 2
                    or any(not isinstance(n, int) or n <= 0 or n > 65536 for n in s)
                    for s in [*inputs, output]
                )
            ):
                raise ValueError("invalid ANE surface dimensions")
            lib = _library(cache)
            if not lib.ys_ane_available():
                return None, lib.ys_ane_error().decode()
            heights = (c.c_uint32 * len(inputs))(*(s[0] for s in inputs))
            widths = (c.c_uint32 * len(inputs))(*(s[1] for s in inputs))
            handle = lib.ys_ane_create(
                mil.encode(),
                weights,
                len(weights),
                str(cache).encode(),
                heights,
                widths,
                len(inputs),
                *output,
            )
            if not handle:
                return None, lib.ys_ane_error().decode()
            return cls(lib, handle, [*inputs, output]), None
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return None, str(exc)

    def _view(self, index):
        import numpy as np

        if not self._handle or self.reason:
            raise RuntimeError(self.reason or "native session closed")
        height, width = self._shapes[index]
        stride = self._lib.ys_ane_stride(self._handle, index)
        address = self._lib.ys_ane_data(self._handle, index)
        buf = (c.c_uint8 * (height * stride)).from_address(address)
        return np.ndarray(
            (height, width), dtype=np.float16, buffer=buf, strides=(stride, 2)
        )

    def copy_input(self, value, index=0):
        with self._lock:
            if index < 0 or index >= len(self._shapes) - 1:
                raise IndexError(index)
            if value.shape != self._shapes[index]:
                raise ValueError("ANE input shape mismatch")
            self._view(index)[:] = value

    def copy_output(self):
        with self._lock:
            return self._view(len(self._shapes) - 1).copy()

    def run(self, repeats: int = 1) -> float | None:
        """Mean native evaluate+completion milliseconds, or None (discard output)."""
        if not isinstance(repeats, int) or not 1 <= repeats <= 10000:
            raise ValueError("repeats must be in 1..10000")
        with self._lock:
            if not self._handle or self.reason:
                return None
            ms = c.c_double()
            if not self._lib.ys_ane_run(self._handle, repeats, c.byref(ms)):
                self.reason = self._lib.ys_ane_error().decode()
                return None
            return ms.value

    def close(self):
        with self._lock:
            if self._handle:
                self._lib.ys_ane_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
