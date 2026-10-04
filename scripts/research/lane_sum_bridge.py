"""Research adapter for exact group-sum reuse through LaneLinear's reshape."""

import inspect
from types import FunctionType


def install():
    import lane_sums_bookkeeping as lane_sums

    from yunshu_engine.kernels import lane_linear

    original = lane_linear.LaneLinear.__call__
    source = inspect.getsource(original)
    source = source[source.index("    def __call__") :]
    before = "        m = int(x2.shape[0])\n"
    after = '        if "bias" in self:\n'
    assert source.count(before) == source.count(after) == 1
    source = source.replace(
        before,
        before
        + "        if m <= 128:\n            _lane_sums.reuse(x, x2, self.group_size)\n",
    )
    source = source.replace(
        after,
        "        if m <= 128:\n            _lane_sums.remember(x, x2, self.group_size)\n"
        + after,
    )
    source = "\n".join(
        line[4:] if line.startswith("    ") else line for line in source.splitlines()
    )
    namespace = {**original.__globals__, "_lane_sums": lane_sums}
    exec(compile(source, "lane_sum_bridge", "exec"), namespace)
    function = namespace["__call__"]
    bridged = FunctionType(
        function.__code__, namespace, original.__name__, original.__defaults__
    )

    def activate(enabled):
        namespace.update(lane_linear.__dict__)
        namespace["_lane_sums"] = lane_sums
        lane_linear.LaneLinear.__call__ = bridged if enabled else original

    return activate
