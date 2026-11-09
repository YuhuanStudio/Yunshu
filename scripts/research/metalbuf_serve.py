"""Server wrapper for the Metal buffer-count investigation: runs `yunshu_cli serve ...` in this process
and a monitor thread that appends counters to $METALBUF_OUT/counters.jsonl every 2 s.

Counters (no GPU work, except on request): live mx.array objects (gc) split by size class, MLX active /
cache memory, APC resident bytes.  When $METALBUF_OUT/probe exists (the client writes it between turns,
server idle) the thread clears MLX's buffer cache and allocates 1 KiB arrays until Metal's resource limit
raises (cap 700k): live_buffers = limit - headroom, i.e. buffers the engine really holds.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time


def classify(arrays) -> dict:
    """Bucket arrays by nbytes: counts of tiny / small / mid / large (pure function, unit-tested)."""
    out = {"n": 0, "lt4k": 0, "lt64k": 0, "lt4m": 0, "ge4m": 0, "bytes": 0}
    for a in arrays:
        nb = int(a.nbytes)
        out["n"] += 1
        out["bytes"] += nb
        key = (
            "lt4k"
            if nb < 4096
            else "lt64k"
            if nb < 65536
            else "lt4m"
            if nb < (4 << 20)
            else "ge4m"
        )
        out[key] += 1
    return out


def top_groups(arrays, k: int = 8) -> list:
    """Most common (shape, dtype) groups: [[shape, dtype, count], ...] (pure, unit-tested)."""
    import collections

    c = collections.Counter((tuple(a.shape), str(a.dtype)) for a in arrays)
    return [[list(sh), dt, n] for (sh, dt), n in c.most_common(k)]


def owners(arrays, shape, depth: int = 8) -> list:
    """Who holds the arrays of ``shape``: type names of the referrer chain for 3 samples."""
    out = []
    for a in [x for x in arrays if tuple(x.shape) == tuple(shape)][:: max(1, 1)][:3]:
        chain, cur = [], a
        for _ in range(depth):
            refs = [
                r for r in gc.get_referrers(cur) if r is not arrays and r is not chain
            ]
            refs = [r for r in refs if r is not chain]
            if not refs:
                break
            r = refs[0]
            desc = type(r).__name__ + f"[{len(r) if hasattr(r, '__len__') else ''}]"
            if type(r).__name__ == "frame":
                desc += ":" + r.f_code.co_name + "@" + r.f_code.co_filename[-30:]
            elif isinstance(r, dict):
                desc += ":" + ",".join(map(str, list(r)[:6]))
            elif hasattr(r, "__dict__"):
                desc += ":" + ",".join(list(vars(r))[:6])
            chain.append(desc)
            cur = r
        out.append(chain)
    return out


def graph_nodes(arrays, cap: int = 3_000_000) -> int:
    """Unevaluated graph size reachable from ``arrays`` (evaluated arrays have no inputs)."""
    import io

    import mlx.core as mx

    arrays = [a for a in arrays if isinstance(a, mx.array)]
    if not arrays:
        return 0
    buf = io.StringIO()
    mx.export_to_dot(buf, *arrays)
    return buf.getvalue().count("[label")


def held_arrays(obj, depth: int = 2):
    """(path, array) for every mx.array reachable from ``obj`` attributes / lists / dicts."""
    import mlx.core as mx

    out = []

    def walk(o, path, d):
        if isinstance(o, mx.array):
            out.append((path, o))
        elif d > 0 and isinstance(o, (list, tuple)):
            for i, x in enumerate(o[:64]):
                walk(x, f"{path}[{i}]", d - 1)
        elif d > 0 and isinstance(o, dict):
            for k, x in list(o.items())[:64]:
                walk(x, f"{path}.{k}", d - 1)
        elif d > 0 and hasattr(o, "__dict__") and not isinstance(o, type):
            for k, x in list(vars(o).items()):
                walk(x, f"{path}.{k}", d - 1)

    walk(obj, "", depth)
    return out


def graph_report(driver) -> dict:
    """Per-attribute unevaluated node counts of the driver, its rows, the batch and the head."""
    rep = {}
    parts = [("driver", driver)]
    parts += [(f"row{i}", r) for i, r in enumerate(driver.rows)]
    for name, obj in parts:
        for path, a in held_arrays(obj, 3):
            n = graph_nodes([a])
            if n > 50:
                rep[f"{name}{path}"] = n
    return dict(sorted(rep.items(), key=lambda kv: -kv[1])[:12])


def probe_headroom(mx, cap: int = 700_000, chunk: int = 10_000) -> int:
    held: list = []
    try:
        while len(held) < cap:
            part = [mx.zeros((256,), mx.float32) + 0 for _ in range(chunk)]
            mx.eval(part)
            held.extend(part)
    except RuntimeError:
        pass
    n = len(held)
    del held
    return n


def monitor(out_dir: str) -> None:
    import mlx.core as mx

    path = os.path.join(out_dir, "counters.jsonl")
    probe_flag = os.path.join(out_dir, "probe")
    while True:
        time.sleep(2)
        rec = {
            "t": round(time.time(), 1),
            "active_mib": mx.get_active_memory() >> 20,
            "cache_mib": mx.get_cache_memory() >> 20,
        }
        try:
            rec["arrays"] = classify(o for o in gc.get_objects() if type(o) is mx.array)
        except Exception as e:  # noqa: BLE001
            rec["arrays_err"] = repr(e)
        if rec.get("arrays", {}).get("n", 0) > 3000:
            live = [o for o in gc.get_objects() if type(o) is mx.array]
            rec["groups"] = top_groups(live)
            rec["owners"] = owners(live, rec["groups"][0][0])
            del live
        if os.path.exists(probe_flag):
            os.unlink(probe_flag)
            rec["cache_mib_before_probe"] = mx.get_cache_memory() >> 20
            mx.clear_cache()
            rec["headroom"] = probe_headroom(mx)
            mx.clear_cache()
            rec["live_buffers_est"] = 499000 - rec["headroom"]
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    out = os.environ["METALBUF_OUT"]
    os.makedirs(out, exist_ok=True)
    if not os.environ.get("METALBUF_NO_MONITOR"):
        threading.Thread(target=monitor, args=(out,), daemon=True).start()
    if os.environ.get("METALBUF_GRAPH"):
        import mlx.core as mx

        from yunshu_engine import keyed_sampling as ks
        from yunshu_engine.round_driver import driver as drv

        glog = os.path.join(out, "graph.jsonl")

        def emit(rec):
            with open(glog, "a") as f:
                f.write(json.dumps(rec) + "\n")

        orig_g = drv.RoundDriver.step
        gcnt = {"n": 0}

        def step_g(self):
            gcnt["n"] += 1
            try:
                res = orig_g(self)
            except RuntimeError as e:
                emit(
                    {
                        "FAIL": str(e)[:80],
                        "step": gcnt["n"],
                        "active_mib": mx.get_active_memory() >> 20,
                        "arrays": sum(
                            1 for o in gc.get_objects() if type(o) is mx.array
                        ),
                        "graph": graph_report(self),
                    }
                )
                raise
            if gcnt["n"] % 5 == 0:
                try:
                    emit(
                        {
                            "step": gcnt["n"],
                            "active_mib": mx.get_active_memory() >> 20,
                            "cache_mib": mx.get_cache_memory() >> 20,
                            "rows": len(self.rows),
                            "prefill_last": getattr(self, "_prefilled_last", None),
                            "graph": graph_report(self),
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    emit({"report_error": repr(e)[:200]})
            return res

        drv.RoundDriver.step = step_g
        orig_c = ks.candidate_rows

        def cand(logprobs, params):
            try:
                return orig_c(logprobs, params)
            except RuntimeError as e:
                emit(
                    {
                        "FAIL_candidate": str(e)[:80],
                        "logprob_graph": graph_nodes([logprobs]),
                    }
                )
                raise

        ks.candidate_rows = cand
    every = int(os.environ.get("METALBUF_CLEAR_EVERY", "0"))
    probe_every = int(os.environ.get("METALBUF_STEP_PROBE", "0"))
    if probe_every:
        # probe on the MLX thread between driver steps: used = buffers the allocator holds
        # (live + cached, cache NOT cleared first); then clear, so the next record is the
        # growth over one interval
        import mlx.core as mx

        from yunshu_engine.round_driver import driver as drv

        orig_p = drv.RoundDriver.step
        cnt = {"n": 0}

        def step_p(self):
            res = orig_p(self)
            cnt["n"] += 1
            if cnt["n"] % probe_every == 0:
                cache_mib = mx.get_cache_memory() >> 20
                head = probe_headroom(mx)
                mx.clear_cache()
                rec = {
                    "step": cnt["n"],
                    "rows": len(self.rows),
                    "cache_mib": cache_mib,
                    "used": 499000 - head,
                }
                with open(os.path.join(out, "steps.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            return res

        drv.RoundDriver.step = step_p
    if every:
        import mlx.core as mx

        from yunshu_engine.round_driver import driver as drv

        orig = drv.RoundDriver.step
        count = {"n": 0}

        def step(self):
            out = orig(self)
            count["n"] += 1
            if count["n"] % every == 0:
                mx.clear_cache()
            return out

        drv.RoundDriver.step = step
    from yunshu_cli import main as cli_main

    sys.argv = ["yunshu"] + sys.argv[1:]
    cli_main()


if __name__ == "__main__":
    main()
