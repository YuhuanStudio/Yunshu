import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

TODOS = {}
NEXT = [1]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, status, obj=None):
        body = b"" if obj is None else json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def err(self, status, msg):
        self.send(status, {"error": msg})

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"")
        except ValueError:
            return None, "invalid json"
        if not isinstance(data, dict):
            return None, "body must be an object"
        return data, None

    def route(self):
        u = urlsplit(self.path)
        if u.path == "/health":
            return "health", None, u
        if u.path == "/todos":
            return "todos", None, u
        m = re.fullmatch(r"/todos/(\d+)", u.path)
        if m:
            return "todo", int(m.group(1)), u
        return None, None, u

    def handle_any(self, method):
        name, tid, u = self.route()
        if name is None:
            return self.err(404, "not found")
        allowed = {"health": {"GET"}, "todos": {"GET", "POST"}, "todo": {"GET", "PATCH", "DELETE"}}[name]
        if method not in allowed:
            return self.err(405, "method not allowed")
        if name == "health":
            return self.send(200, {"status": "ok"})
        if name == "todos" and method == "GET":
            q = parse_qs(u.query).get("done")
            items = [TODOS[i] for i in sorted(TODOS)]
            if q:
                if q[0] not in ("true", "false"):
                    return self.err(400, "done must be true or false")
                items = [t for t in items if t["done"] == (q[0] == "true")]
            return self.send(200, items)
        if name == "todos":
            data, e = self.read_json()
            if e:
                return self.err(400, e)
            title = data.get("title")
            done = data.get("done", False)
            if not isinstance(title, str) or not title.strip() or not isinstance(done, bool):
                return self.err(400, "invalid title or done")
            t = {"id": NEXT[0], "title": title, "done": done}
            TODOS[NEXT[0]] = t
            NEXT[0] += 1
            return self.send(201, t)
        if tid not in TODOS:
            return self.err(404, "no such todo")
        if method == "GET":
            return self.send(200, TODOS[tid])
        if method == "DELETE":
            del TODOS[tid]
            return self.send(204)
        data, e = self.read_json()
        if e:
            return self.err(400, e)
        if "title" in data and (not isinstance(data["title"], str) or not data["title"].strip()):
            return self.err(400, "invalid title")
        if "done" in data and not isinstance(data["done"], bool):
            return self.err(400, "invalid done")
        for k in ("title", "done"):
            if k in data:
                TODOS[tid][k] = data[k]
        return self.send(200, TODOS[tid])

    def do_GET(self):
        self.handle_any("GET")

    def do_POST(self):
        self.handle_any("POST")

    def do_PATCH(self):
        self.handle_any("PATCH")

    def do_DELETE(self):
        self.handle_any("DELETE")

    def do_PUT(self):
        self.handle_any("PUT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
