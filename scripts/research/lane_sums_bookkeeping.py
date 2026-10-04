"""Carry exact lane group sums across a projection's reshape.

Only sums already produced by the lane matmul are reused. The separate fused
norm helpers use a different summation order and are deliberately not read.
All entries share the lane kernel's existing four-entry bounded cache.
"""

from yunshu_engine.kernels.tensorfold import lane_qmm as q


def _projection_key(view, group):
    return id(view) if group == 64 else (id(view), group)


def _input_key(original, group):
    return ("lane-input", id(original), group)


def _shape_ok(original, view, sums, group):
    rows, width = view.shape
    return (
        original.shape[-1] == width
        and original.size == view.size
        and 0 < rows <= 128
        and width % group == 0
        and tuple(sums.shape) == (width // group, 16 * ((rows + 15) // 16))
    )


def _put(key, value):
    # Refresh the shared input after each projection, so a four-member GDN
    # bundle can reuse it without the first projection's view being retained.
    q._xs_cache.pop(key, None)
    q._xs_cache[key] = value
    while len(q._xs_cache) > 4:
        q._xs_cache.pop(next(iter(q._xs_cache)))


def reuse(original, view, group):
    """Seed the view's normal cache entry from this same input, if available."""
    hit = q._xs_cache.get(_input_key(original, group))
    if (
        hit is None
        or hit[0] is not original
        or not _shape_ok(original, view, hit[1], group)
    ):
        return False
    _put(_projection_key(view, group), (view, hit[1]))
    return True


def remember(original, view, group):
    """Remember the exact standard-kernel sums after this projection built."""
    hit = q._xs_cache.get(_projection_key(view, group))
    if (
        hit is None
        or hit[0] is not view
        or not _shape_ok(original, view, hit[1], group)
    ):
        return False
    _put(_input_key(original, group), (original, hit[1]))
    return True
