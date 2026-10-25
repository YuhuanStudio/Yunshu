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
from contextlib import suppress
from pathlib import Path

_BUILD_LOCK = threading.Lock()


def build(cache: Path, mlx_root: Path | None = None) -> Path:
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
    extra = []
    if mlx_root is not None:
        mlx_root = mlx_root.resolve()
        library = mlx_root / "lib/libmlx.dylib"
        digest.update(str(mlx_root).encode())
        digest.update(hashlib.sha256(library.read_bytes()).digest())
        extra = [
            "-DYS_ANE_MLX",
            f"-I{mlx_root / 'include'}",
            f"-I{mlx_root / 'include/metal_cpp'}",
            f"-L{mlx_root / 'lib'}",
            f"-Wl,-rpath,{mlx_root / 'lib'}",
            "-lmlx",
        ]
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
                        *extra,
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


def _library(cache: Path, mlx_root: Path | None = None):
    return _configure_library(c.CDLL(str(build(cache, mlx_root))), mlx_root is not None)


def _configure_library(lib, with_mlx=False):
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
    lib.ys_ane_retain.argtypes = [c.c_void_p]
    lib.ys_ane_retain.restype = c.c_void_p
    lib.ys_ane_busy.argtypes = [c.c_void_p]
    lib.ys_ane_busy.restype = c.c_int
    if with_mlx:
        lib.ys_ane_mlx_begin.argtypes = [c.c_void_p]
        lib.ys_ane_mlx_begin.restype = c.c_void_p
        for name in ("ys_ane_mlx_wait", "ys_ane_mlx_finish"):
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = [c.c_void_p], c.c_int
        lib.ys_ane_mlx_destroy.argtypes = [c.c_void_p]
        lib.ys_ane_mlx_destroy.restype = None
    return lib


class _SurfaceOwner:
    def __init__(self, lib, handle):
        self.lib, self.handle = lib, handle

    def close(self):
        if self.handle:
            self.lib.ys_ane_destroy(self.handle)
            self.handle = None

    def __del__(self):
        with suppress(Exception):  # interpreter shutdown
            self.close()


class NativeSession:
    """Fixed MIL program/surfaces. Use try_create and run's None for GPU fallback.

    Copy methods hold its lock. Optional MLX aliases retain the native allocation
    independently of close(); their producer/consumer ordering is explicit through
    a shared-event ticket. Timings from run exclude copies.
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
        *,
        mlx_root: Path | None = None,
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
            lib = _library(cache, mlx_root) if mlx_root is not None else _library(cache)
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
        if self._lib.ys_ane_busy(self._handle):
            raise RuntimeError("native ANE session is in flight")
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

    def try_alias(self, index=0):
        """MLX view with copy=False and retained IOSurface ownership, or unavailable.

        An alias is mutable external memory. Never reuse it before the consumer
        finishes; use an MLX shared-event ticket for GPU/ANE ordering.
        """
        owner = None
        try:
            import mlx.core as mx

            with self._lock:
                if index < 0 or index >= len(self._shapes):
                    raise IndexError(index)
                view = self._view(index)
                retained = self._lib.ys_ane_retain(self._handle)
                if not retained:
                    raise RuntimeError(self._lib.ys_ane_error().decode())
                owner = _SurfaceOwner(self._lib, retained)
                view.base._ane_owner = owner
                alias = mx.from_dlpack(view, copy=False)
                return alias, None
        except (
            ImportError,
            OSError,
            ValueError,
            RuntimeError,
            BufferError,
            AttributeError,
            TypeError,
        ) as exc:
            if owner is not None:
                owner.close()
            self.reason = str(exc)
            return None, self.reason

    def try_submit_mlx(self):
        """After async_eval(producer), signal GPU-ready and queue ANE without a CPU wait."""
        with self._lock:
            if not self._handle or self.reason:
                return None, self.reason or "native session closed"
            if not hasattr(self._lib, "ys_ane_mlx_begin"):
                return None, "native library was built without MLX interop"
            handle = self._lib.ys_ane_mlx_begin(self._handle)
            if not handle:
                self.reason = self._lib.ys_ane_error().decode()
                return None, self.reason
            return MlxTicket(self, handle), None

    def close(self):
        with self._lock:
            if self._handle:
                self._lib.ys_ane_destroy(self._handle)
                self._handle = None

    def __del__(self):
        with suppress(Exception):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class MlxTicket:
    """Explicit same-stream GPU wait; validate after the final consumer eval.

    A failed or cancelled ticket retires the native service, releases GPU waits
    and retains unreported requests' storage until the service actually settles.
    """

    def __init__(self, session, handle):
        self._session, self._lib, self._handle = session, session._lib, handle
        self.reason = None

    def wait_mlx(self) -> bool:
        if not self._handle:
            return False
        if not self._lib.ys_ane_mlx_wait(self._handle):
            self.reason = self._lib.ys_ane_error().decode()
            with suppress(AttributeError):
                self._session.reason = self.reason
            return False
        return True

    def finish(self) -> bool:
        if not self._handle:
            return False
        if not self._lib.ys_ane_mlx_finish(self._handle):
            self.reason = self._lib.ys_ane_error().decode()
            with suppress(AttributeError):
                self._session.reason = self.reason
            return False
        return True

    def close(self):
        if self._handle:
            self._lib.ys_ane_mlx_destroy(self._handle)
            self._handle = None

    def __del__(self):
        with suppress(Exception):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
