"""Diagnostic-only scoped GC suppression; never installed by normal serving."""

import functools
import gc


def defer_index_gc(original, collector=gc):
    """Leave a caller's disabled state intact, restoring after errors/nesting.

    CPython GC is process-wide. This shifts automatic collection out of the
    index scope; it does not guarantee earlier delivery or suppress explicit
    gc.collect(). Never freeze a server's permanent heap here.
    """

    @functools.wraps(original)
    def invoke(*args, **kwargs):
        enabled = collector.isenabled()
        if enabled:
            collector.disable()
        try:
            return original(*args, **kwargs)
        finally:
            if enabled:
                collector.enable()

    return invoke
