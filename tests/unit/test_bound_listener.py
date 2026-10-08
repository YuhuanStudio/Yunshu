"""Reserved listeners are OS-assigned, outside the shared server pool, and stay owned."""

from . import bound_listener as bl


def test_reserved_listener_is_outside_the_shared_pool_and_unique():
    a, b = bl.reserve_listener(), bl.reserve_listener()
    try:
        pa, pb = a.getsockname()[1], b.getsockname()[1]
        assert pa != pb
        assert not 18990 <= pa <= 18999 and not 18990 <= pb <= 18999
    finally:
        a.close()
        b.close()
