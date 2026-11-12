"""Real-server probe: an over-long prompt on /v1/chat/completions and /v1/messages, and a bad
reasoning_effort. Prints the raw HTTP status and body of each (stdlib only).

Run it against a server that is already listening: python scripts/dev/overflow_probe.py PORT MODEL
"""

import json
import sys
import time
import urllib.error
import urllib.request


def post(base: str, path: str, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def wait_ready(base: str, seconds: int = 900) -> None:
    end = time.time() + seconds
    while time.time() < end:
        try:
            with urllib.request.urlopen(base + "/health/ready", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise SystemExit("server did not become ready")


def main() -> None:
    port, model = sys.argv[1], sys.argv[2]
    base = f"http://127.0.0.1:{port}"
    wait_ready(base)
    print("== ready", flush=True)
    long_text = "The quick brown fox jumps over the lazy dog. " * 20000
    cases = [
        (
            "over-long prompt, /v1/chat/completions",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": long_text}],
                "max_tokens": 16,
            },
        ),
        (
            "over-long prompt, /v1/messages",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": long_text}],
            },
        ),
        (
            "bad reasoning_effort, /v1/chat/completions",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "reasoning_effort": "ultra",
            },
        ),
    ]
    for name, path, body in cases:
        print(f"== sending: {name}", flush=True)
        status, text = post(base, path, body)
        print(f"== {name}\nHTTP {status}\n{text}\n", flush=True)


if __name__ == "__main__":
    main()
