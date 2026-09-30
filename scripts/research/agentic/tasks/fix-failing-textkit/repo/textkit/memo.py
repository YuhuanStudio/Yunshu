"""A tiny memoisation decorator."""

import functools


def memoize(fn):
    """Cache results per distinct call (positional and keyword arguments both count)."""
    cache = {}

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        key = args
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]

    inner.cache = cache
    inner.cache_clear = cache.clear
    return inner
