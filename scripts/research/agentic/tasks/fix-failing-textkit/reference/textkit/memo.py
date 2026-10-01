"""A tiny memoisation decorator."""

import functools


def memoize(fn):
    cache = {}

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]

    inner.cache = cache
    inner.cache_clear = cache.clear
    return inner
