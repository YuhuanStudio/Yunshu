"""Full-capture passthrough proxy: forwards every request to an upstream server unchanged and records
method, path, headers, request body, status, response headers and (bounded) response body to JSONL.

Used between a real agent and a real Yunshu so a session leaves evidence for every endpoint the
agent touched (404s included). Streaming responses are relayed as they arrive.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOP = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "content-length",
    "host",
    "accept-encoding",
    "upgrade",
}
KEEP = 400_000  # bytes of response body kept per request


class CaptureProxy:
    def __init__(self, upstream: str, log: Path, port: int = 0):
        host, _, p = upstream.replace("http://", "").partition(":")
        self.up = (host, int(p))
        self.log = log
        self.lock = threading.Lock()
        log.parent.mkdir(parents=True, exist_ok=True)
        proxy = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def handle_any(self):
                proxy.forward(self)

            do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_OPTIONS = handle_any  # noqa: N815

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def forward(self, h):
        t0 = time.time()
        n = int(h.headers.get("Content-Length") or 0)
        body = h.rfile.read(n) if n else b""
        try:
            req_json = json.loads(body) if body else None
        except ValueError:
            req_json = {"_raw": body[:2000].decode("utf-8", "replace")}
        rec = {
            "t": t0,
            "method": h.command,
            "path": h.path,
            "headers": dict(h.headers.items()),
            "body": req_json,
        }
        buf = bytearray()
        conn = http.client.HTTPConnection(*self.up, timeout=3600)
        try:
            hdr = {k: v for k, v in h.headers.items() if k.lower() not in HOP}
            hdr["Accept-Encoding"] = "identity"
            conn.request(h.command, h.path, body=body or None, headers=hdr)
            resp = conn.getresponse()
            rec["status"] = resp.status
            rec["resp_headers"] = dict(resp.getheaders())
            length = resp.getheader("Content-Length")
            h.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    h.send_header(k, v)
            chunked = length is None
            h.send_header("Transfer-Encoding", "chunked") if chunked else h.send_header(
                "Content-Length", length
            )
            h.end_headers()
            while True:
                chunk = resp.read1(65536) if h.command != "HEAD" else b""
                if not chunk:
                    break
                if len(buf) < KEEP:
                    buf += chunk[: KEEP - len(buf)]
                h.wfile.write(
                    b"%x\r\n%s\r\n" % (len(chunk), chunk) if chunked else chunk
                )
                h.wfile.flush()
            if chunked:
                h.wfile.write(b"0\r\n\r\n")
                h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            rec["client_aborted"] = True
        except Exception as e:  # noqa: BLE001
            rec["proxy_error"] = f"{type(e).__name__}: {e}"
            with contextlib.suppress(Exception):
                msg = json.dumps({"error": {"message": rec["proxy_error"]}}).encode()
                h.send_response(502)
                h.send_header("Content-Length", str(len(msg)))
                h.end_headers()
                h.wfile.write(msg)
        finally:
            conn.close()
        rec["secs"] = round(time.time() - t0, 3)
        rec["resp_body"] = bytes(buf).decode("utf-8", "replace")
        with self.lock, self.log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
