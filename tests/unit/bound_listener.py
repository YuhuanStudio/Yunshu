"""Hold an OS-assigned loopback listener across uvicorn startup; never probe then release.

The port is outside the shared 18990-18999 pool that live (gpuq) servers may hold.
"""

from __future__ import annotations

import socket


def reserve_listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
    except OSError:
        listener.close()
        raise
    return listener
