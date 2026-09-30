"""A tiny SearXNG-compatible search server over a fixed corpus (for tests; no network).

    GET /search?q=...&format=json  ->  {"results": [{"title", "url", "content", "publishedDate"}]}
    GET /page/<slug>               ->  an HTML page (so web_fetch can follow the result links)

Every query returns the corpus entries whose text shares a word with the query (best first), so a
model asking about "Yunshu" or "MLX" gets stable, checkable facts.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

CORPUS = {
    "yunshu-overview": (
        "Yunshu overview",
        "Yunshu is a local inference engine for Apple Silicon. It serves OpenAI and Anthropic compatible "
        "endpoints from one process. The reference release code name is Cloud Book 42.",
        "2026-05-01",
    ),
    "yunshu-search": (
        "Yunshu server-side web search",
        "Yunshu runs web_search and web_fetch inside the generation loop. The secret verification word for "
        "this page is TANGERINE-7.",
        "2026-06-12",
    ),
    "mlx-arrays": (
        "MLX array framework",
        "MLX is an array framework for machine learning on Apple Silicon developed by Apple machine learning "
        "research.",
        "2025-12-05",
    ),
}


def make_server(port: int = 0) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlsplit(self.path)
            base = f"http://127.0.0.1:{self.server.server_address[1]}"
            if u.path == "/search":
                q = (parse_qs(u.query).get("q") or [""])[0].lower()
                words = set(re.findall(r"\w+", q))
                scored = []
                for slug, (title, text, date) in CORPUS.items():
                    hit = len(
                        words & set(re.findall(r"\w+", (title + " " + text).lower()))
                    )
                    if hit:
                        scored.append((hit, slug, title, text, date))
                scored.sort(reverse=True)
                results = [
                    {
                        "title": t,
                        "url": f"{base}/page/{s}",
                        "content": x,
                        "publishedDate": d,
                    }
                    for _, s, t, x, d in scored
                ]
                return self._send(
                    200,
                    json.dumps({"query": q, "results": results}).encode(),
                    "application/json",
                )
            if u.path.startswith("/page/") and u.path[6:] in CORPUS:
                title, text, _ = CORPUS[u.path[6:]]
                html = f"<html><head><title>{title}</title></head><body><h1>{title}</h1><p>{text}</p></body></html>"
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            self._send(404, b"not found", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18998)
    a = ap.parse_args()
    s = make_server(a.port)
    print(f"http://127.0.0.1:{s.server_address[1]}", flush=True)
    threading.Event().wait()
