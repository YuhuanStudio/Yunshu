# --- wall-clock probe: per-call durations of the APC hot path, to a side file.
def _timed(cls, name):
    orig = getattr(cls, name, None)
    if orig is None:
        return

    def wrapper(self, *args, **kwargs):
        t0 = _time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            path = _os.environ.get("APC_PROBE_LOG")
            if path:
                dt = (_time.perf_counter() - t0) * 1000
                with open(path + ".t", "a") as f:
                    f.write(
                        _json.dumps(
                            dict(
                                t=round(_time.time(), 3),
                                fn=cls.__name__ + "." + name,
                                ms=round(dt, 3),
                            )
                        )
                        + "\n"
                    )

    setattr(cls, name, wrapper)


for _n in (
    "lookup_exact_cache",
    "_make_room",
    "begin_request",
    "release_superseded",
    "_supersede",
    "store_exact_cache",
    "share_anchor_rows",
    "share_anchor_rows_lazy",
    "finish_anchor_sharing",
):
    _timed(YunshuAPCManager, _n)  # noqa: F821
for _n in ("flush_deferred_checkpoints", "store_checkpoint"):
    _timed(_Coordinator, _n)  # noqa: F821


def _timed_fn(name):
    orig = globals().get(name)
    if orig is None:
        return

    def wrapper(*args, **kwargs):
        t0 = _time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            path = _os.environ.get("APC_PROBE_LOG")
            if path:
                dt = (_time.perf_counter() - t0) * 1000
                with open(path + ".t", "a") as f:
                    f.write(
                        _json.dumps(
                            dict(
                                t=round(_time.time(), 3),
                                fn="module." + name,
                                ms=round(dt, 3),
                                arg=args[0]
                                if args and isinstance(args[0], int)
                                else None,
                            )
                        )
                        + "\n"
                    )

    globals()[name] = wrapper


_timed_fn("release_freed_buffers")
_timed_fn("materialize")
