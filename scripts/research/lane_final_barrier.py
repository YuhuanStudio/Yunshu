"""Research-only removal of the last split-K shared-buffer reuse barrier.

The first barrier publishes partials. The second protects reuse by the next
16-row block; there is no next block on the final iteration. Arithmetic and
partial publication order are unchanged.
"""


def install():
    from yunshu_engine.kernels.tensorfold import lane_qmm, lane_widen

    targets = [
        (lane_qmm, "_MAIN"),
        (lane_qmm, "_MAIN_TILED"),
        (lane_widen, "NIBBLES"),
        (lane_widen, "BYTES"),
    ]
    original = [(obj, name, getattr(obj, name)) for obj, name in targets]
    needle = "      threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n    if (sg == 0)"
    replacement = "      if (t + 1 < TMR) threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n    if (sg == 0)"
    for obj, name, source in original:
        if source.count(needle) != 1:
            raise RuntimeError(f"unexpected reduction source: {name}")
        setattr(obj, name, source.replace(needle, replacement))
    lane_qmm._kernels.clear()

    def uninstall():
        for obj, name, source in original:
            setattr(obj, name, source)
        lane_qmm._kernels.clear()

    return uninstall
