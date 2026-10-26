"""Fail-fast Flash-Next HTTP serving sanity gate, one model/drafter per process."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROMPT = "Write a Python function that returns the n-th Fibonacci number, then one sentence explaining it."


def validate(reply):
    if "error" in reply:
        raise ValueError(str(reply["error"]))
    message = reply["choices"][0]["message"]
    text = message.get("content") or ""
    if "def " not in text or "return " not in text:
        raise ValueError(f"Fibonacci sanity failed: {text[:160]!r}")
    return text


def request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18996, choices=range(18990, 19000))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"complete": True, "dry_run": True}))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    home = args.out.parent / (args.out.stem + "-home")
    home.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "python"),
        HOME=str(home),
        YUNSHU_VLM_APC_DISK="0",
        YUNSHU_CONSOLE="0",
        YUNSHU_VLM_DRAFT=args.draft,
    )
    log_path = args.out.with_suffix(".server.log")
    result = {"model": args.model, "draft": args.draft, "complete": False}
    url = f"http://127.0.0.1:{args.port}"
    # Refuse an occupied port before starting; never query another worker's server.
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "yunshu_cli",
                "serve",
                "-m",
                args.model,
                "--port",
                str(args.port),
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"server exited {server.returncode}; see {log_path}"
                    )
                try:
                    request(url + "/health")
                    break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("server not ready after 600 seconds")
            model_id = request(url + "/v1/models")["data"][0]["id"]
            reply = request(
                url + "/v1/chat/completions",
                {
                    "model": model_id,
                    "temperature": 0,
                    "max_tokens": 160,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": PROMPT}],
                },
            )
            result["reply"] = reply
            validate(reply)
            result["complete"] = True
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            result["engagement"] = [
                line.strip()
                for line in log_path.read_text().splitlines()
                if "VLM batch runner" in line or "Speculative decoding" in line
            ]
            args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "reply"}), flush=True)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
