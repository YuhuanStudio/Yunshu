"""scripts/research/ab_gpu_sampler.py: the pieces that decide the YUNSHU_GPU_SAMPLER flag."""

import importlib.util
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research/ab_gpu_sampler.py"
spec = importlib.util.spec_from_file_location("ab_gpu_sampler", SCRIPT)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = body["max_tokens"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(n):
            ev = {"choices": [{"delta": {"content": f"t{i % 3} "}}]}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        usage = {"choices": [], "usage": {"completion_tokens": n}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())


@pytest.fixture
def stub():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    class S:
        url = f"http://127.0.0.1:{httpd.server_port}"
        model = "m"

    yield S
    httpd.shutdown()


def test_tv_distance():
    a, b = Counter({"x": 3, "y": 1}), Counter({"x": 1, "y": 3})
    assert ab.tv_distance(a, a) == 0
    assert ab.tv_distance(a, b) == pytest.approx(0.5)
    assert ab.tv_distance(Counter({"x": 5}), Counter({"y": 9})) == 1


def test_stream_chat_parses_text_usage_and_speed(stub):
    r = ab.stream_chat(stub.url, stub.model, "hi", 8, temperature=0.7, seed=1)
    assert r["completion_tokens"] == 8 and r["text"].startswith("t0 t1 t2 t0")
    assert r["ttft"] >= 0 and len(r["gaps"]) == 7 and r["tok_s"] > 0


def test_speed_and_first_token_rows(stub, monkeypatch):
    monkeypatch.setattr(ab, "PROMPTS", ab.PROMPTS[:2])
    rows = ab.speed_rows(stub, max_tokens=6, reps=2)
    assert set(rows) == set(ab.CONFIGS)
    assert all(len(r["tok_s"]) == 4 and r["median_tok_s"] > 0 for r in rows.values())
    counts = ab.first_token_counts(stub, samples=5)
    assert all(sum(c.values()) == 5 for c in counts.values())


def _report(gpu_tok_s, tv_cross, tv_noise, gpu_seed=True):
    def rnd(arm, v):
        return {"arm": arm, "speed": {c: {"median_tok_s": v} for c in ab.CONFIGS}}

    return {
        "rounds": [rnd("numpy", 100.0), rnd("gpu", gpu_tok_s), rnd("numpy", 100.0)],
        "tv_numpy_vs_gpu": tv_cross,
        "tv_numpy_vs_numpy": tv_noise,
        "seed_repro": {"numpy": True, "gpu": gpu_seed},
    }


def test_verdict_flips_only_when_faster_equivalent_and_reproducible():
    flip = ab.verdict(_report(120.0, 0.04, 0.05), 1.05, 1.5, 0.05)
    assert flip["flip_default_to_gpu"] and flip["ratios_gpu_over_numpy"]["t0.7"] == 1.2
    assert not ab.verdict(_report(101.0, 0.04, 0.05), 1.05, 1.5, 0.05)[
        "flip_default_to_gpu"
    ]
    assert not ab.verdict(_report(120.0, 0.30, 0.05), 1.05, 1.5, 0.05)[
        "flip_default_to_gpu"
    ]
    assert not ab.verdict(_report(120.0, 0.04, 0.05, gpu_seed=False), 1.05, 1.5, 0.05)[
        "flip_default_to_gpu"
    ]
    # a single round has no noise estimate: only the floor applies
    assert ab.verdict(_report(120.0, 0.04, None), 1.05, 1.5, 0.05)[
        "flip_default_to_gpu"
    ]
