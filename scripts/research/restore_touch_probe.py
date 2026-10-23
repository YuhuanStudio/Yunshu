"""Instrument allocator/lifetime boundaries in warmprobe; GPU execution via gpuq only."""

import argparse
import contextlib
import functools
import json
import runpy
import subprocess
import sys
import time
import weakref
from pathlib import Path


def install():
    import mlx.core as mx
    from mlx_vlm.generate.ar import PromptProcessingBatch

    from yunshu_engine import apc_manager, vlm_batch_runner
    from yunshu_engine.kernels.buffer_cache import restore_buffer_bytes

    def record(event, **extra):
        print(
            "RESTORE_TOUCH "
            + json.dumps(
                dict(
                    event=event,
                    at=time.monotonic(),
                    active=mx.get_active_memory(),
                    spare=restore_buffer_bytes(),
                    pool=mx.get_cache_memory(),
                    peak=mx.get_peak_memory(),
                    **extra,
                )
            ),
            flush=True,
        )

    original = apc_manager._restore_kv

    def restore(c, capacity, targets):
        record(
            "restore_before",
            rows=c.offset,
            source=list(c.keys.shape),
            requested=capacity,
        )
        out = original(c, capacity, targets)
        record("restore_graph", destination=list(out.keys.shape))
        return out

    apc_manager._restore_kv = restore

    finish = vlm_batch_runner.VLMBatchRunner._finish

    def finished(self, group, uid, reason):
        batch = getattr(group.gen, "_generation_batch", None)
        caches = getattr(batch, "prompt_cache", None) or []
        shapes = []
        for i, c in enumerate(caches):
            a = getattr(c, "keys", None)
            if a is not None:
                shapes.append(
                    [
                        i,
                        list(a.shape),
                        int(getattr(c, "offset", 0)),
                        type(c).__module__,
                        type(c).__name__,
                    ]
                )
                with contextlib.suppress(TypeError):
                    weakref.finalize(
                        a, record, "finished_keys_handle_released", layer=i
                    )
        record("finish_before", shapes=shapes)
        result = finish(self, group, uid, reason)
        record("finish_after")
        return result

    vlm_batch_runner.VLMBatchRunner._finish = finished

    def wrap(owner, name):
        fn = getattr(owner, name)

        @functools.wraps(fn)
        def call(*args, **kwargs):
            record(name + "_before")
            try:
                return fn(*args, **kwargs)
            finally:
                record(name + "_after")

        setattr(owner, name, call)

    wrap(vlm_batch_runner, "close_generator")
    wrap(PromptProcessingBatch, "prompt_step")
    # The cloned arrays are evaluated in this function, in 512 MiB groups.
    import mlx_vlm.apc as apc

    wrap(apc, "_clone_prompt_cache_for_apc")


def storage_probe():
    """Can MLX donate a finished buffer to a prefix overwrite, including zero tail?"""
    import mlx.core as mx
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine.apc_manager import _restore_kv

    mx.set_cache_limit(6 << 30)
    caches = []
    for _ in range(16):
        c = KVCache()
        c.keys = mx.full((1, 4, 32768, 256), 0.5, dtype=mx.bfloat16)
        c.values = mx.full(c.keys.shape, -0.5, dtype=mx.bfloat16)
        c.offset = 32768
        caches.append(c)
    mx.eval([c.state for c in caches])
    held = None
    for label in ("base", "reuse", "reuse-r1", "base-r1"):
        if label.startswith("base"):
            held = None
            mx.clear_cache()
        else:
            # Deliberately warm destination storage at the finished request's capacity.
            held = []
            for c in caches:
                held.extend(
                    mx.zeros((1, 4, 33792, 256), dtype=a.dtype) for a in c.state
                )
            mx.eval(held)
        mx.synchronize()
        before = (mx.get_active_memory(), mx.get_cache_memory())
        start = time.perf_counter()
        targets = []
        for c in caches:
            if held is None:
                _restore_kv(c, 33400, targets)
            else:
                for a in c.state:
                    dst = held.pop(0)
                    dst[..., : c.offset, :] = a
                    dst[..., c.offset :, :] = 0
                    targets.append(dst)
            if len(targets) % 8 == 0:
                mx.eval(targets[-8:])
        mx.eval(targets)
        mx.synchronize()
        ms = (time.perf_counter() - start) * 1000
        # Check every valid row and the zero tail, outside the measured interval.
        for i, dst in enumerate(targets):
            assert mx.array_equal(
                dst[..., :32768, :], caches[i // 2].state[i % 2]
            ).item()
            assert mx.all(dst[..., 32768:, :] == 0).item()
        print(
            json.dumps(
                dict(
                    label=label,
                    ms=ms,
                    before=before,
                    active=mx.get_active_memory(),
                    pool=mx.get_cache_memory(),
                    peak=mx.get_peak_memory(),
                )
            ),
            flush=True,
        )
        del targets
    print(json.dumps({"complete": True}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--storage", action="store_true")
    p.add_argument("--plain", action="store_true")
    p.add_argument("--memory", action="store_true")
    args, rest = p.parse_known_args()
    if args.dry_run:
        print(json.dumps({"complete": True, "dry_run": True}))
        return
    if args.storage:
        storage_probe()
        return
    if args.serve:
        install()
        from yunshu_cli import main as cli

        sys.argv = ["yunshu", *rest]
        cli()
        return
    original = subprocess.Popen

    def launch(command, *a, **kw):
        if command and Path(command[0]).name == "yunshu":
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--serve",
                *command[1:],
            ]
        return original(command, *a, **kw)

    if not args.plain:
        subprocess.Popen = launch
    if args.memory:
        import urllib.request

        sys.path.insert(
            0, "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/research"
        )
        import tfbench as t

        start_server = t.start_server

        def start(*a, **kw):
            srv = start_server(*a, **kw)
            kill = srv.kill

            def finish():
                try:
                    for phase, delay in (("finished", 0), ("idle31", 31)):
                        time.sleep(delay)
                        req = urllib.request.Request(
                            srv.url + "/v1/yunshu/memory",
                            headers={"Authorization": "Bearer k"},
                        )
                        with urllib.request.urlopen(req, timeout=10) as response:
                            sample = json.load(response)
                        print(
                            json.dumps(
                                {
                                    "memory_phase": phase,
                                    "mlx": sample["mlx"],
                                    "owners": sample["owners"],
                                }
                            ),
                            flush=True,
                        )
                finally:
                    kill()

            srv.kill = finish
            return srv

        t.start_server = start
    sys.argv = ["warmprobe", *rest]
    runpy.run_path(
        "/Volumes/P5Plus/yunshu-build/lead/warmttft/warmprobe.py", run_name="__main__"
    )
    print(json.dumps({"complete": True}))


if __name__ == "__main__":
    main()
