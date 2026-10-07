"""Hold an allowed loopback listener across uvicorn startup; never probe then release."""

from __future__ import annotations

import socket


def reserve_listener():
    for port in range(18990, 19000):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", port))
            listener.listen(128)
            return listener
        except OSError:
            listener.close()
    raise RuntimeError("no free port in 18990-18999")
