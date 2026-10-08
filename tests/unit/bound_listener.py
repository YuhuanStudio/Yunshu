"""Hold an allowed loopback listener across uvicorn startup; never probe then release."""

from __future__ import annotations

import socket
import time


def reserve_listener(*, timeout=600, poll_interval=0.25):
    deadline = time.monotonic() + max(0, timeout)
    while True:
        for port in range(18990, 19000):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # Closed test servers can leave TIME_WAIT entries. This still refuses
                # a live listener, whose reservation is held through child startup.
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", port))
                listener.listen(128)
                return listener
            except OSError:
                listener.close()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("no free port in 18990-18999")
        time.sleep(min(poll_interval, remaining))
