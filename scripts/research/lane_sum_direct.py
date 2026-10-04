"""Keep the original BF16 input identity at the lane matmul boundary."""

import inspect
from types import FunctionType


def install():
    from yunshu_engine.kernels import lane_linear as module

    original_rows = module.LaneLinear._rows
    original_call = module.LaneLinear.__call__
    if "_SUM_REUSE" in inspect.getsource(original_call):
        return module.set_sum_reuse
    source = inspect.getsource(original_rows)
    source = source.replace(
        "def _rows(self, x2: mx.array)",
        "def _rows(self, x2: mx.array, *, cache_input=None)",
    )
    # New main also has a prefill_narrow keyword; preserve it.
    source = source.replace(
        "*, prefill_narrow: bool = False)",
        "*, prefill_narrow: bool = False, cache_input=None)",
    )
    assert "cache_input=None" in source
    needle = "            x2,\n"
    assert source.count(needle) == 1
    source = source.replace(
        needle, "            cache_input if cache_input is not None else x2,\n"
    )
    source = "\n".join(
        line[4:] if line.startswith("    ") else line for line in source.splitlines()
    )
    namespace = dict(original_rows.__globals__)
    exec(compile(source, "lane_sum_direct_rows", "exec"), namespace)
    rows = namespace["_rows"]
    source = inspect.getsource(original_call)
    needle = "y = self._rows(x2)"
    assert source.count(needle) == 1
    source = source.replace(
        needle,
        "y = self._rows(x2, cache_input=x if dtype == mx.bfloat16 and m <= 128 else None)",
    )
    source = "\n".join(
        line[4:] if line.startswith("    ") else line for line in source.splitlines()
    )
    exec(compile(source, "lane_sum_direct_call", "exec"), namespace)
    call = namespace["__call__"]
    direct_rows = FunctionType(
        rows.__code__, namespace, original_rows.__name__, original_rows.__defaults__
    )
    direct_rows.__kwdefaults__ = rows.__kwdefaults__
    direct_call = FunctionType(
        call.__code__, namespace, original_call.__name__, original_call.__defaults__
    )

    def activate(enabled):
        namespace.update(module.__dict__)
        module.LaneLinear._rows = direct_rows if enabled else original_rows
        module.LaneLinear.__call__ = direct_call if enabled else original_call

    return activate
