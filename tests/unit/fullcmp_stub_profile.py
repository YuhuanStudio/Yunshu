"""Test-only stub with layer-2 profile output and B4 behaviour routes (no model, no GPU).

Composes the base stub (fullcmp_stub_server.py) instead of changing it:

* Profile: when the stub process runs with YUNSHU_DEBUG_TREE_PROFILE=1 and the engine is yunshu,
  every request also logs the three lines the real engine prints (dflash_fast.py: "DFlash round
  profile", "DFlash host between rounds", "Tree verify stage profile"). The real engine does NOT yet
  print the layer-2 fields that are not in those lines (plan, copy_lookup, verify_weights, ...,
  host_sync_count, op_census). The stub therefore also logs one structured line,
  ``YUNSHU_LAYER2 {json}``, the contract fullcmp_profile reads for them. Until the engine prints it,
  a real --full profile rep fails closed (see the profile tests).
* Behaviour: /v1/messages and /v1/responses routes, chat tool_calls (tools present),
  JSON-schema content (response_format), and per-token logprobs (logprobs=true). Non-stream only;
  every other request goes to the base stub unchanged.

Used by tests only: tfbench runs it through TFB_STUB_SERVER + TFB_STUB_SCRIPT=fullcmp_stub_profile.py;
fullcmp_behaviour tests start it in-process via start().
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullcmp_stub_server as base  # noqa: E402

PROFILE_LINES = [
    "DFlash round profile: 1 rounds, 7.5 nodes/round, draft 1.20 ms, verify 40.00 ms, commit+host 3.00 ms",
    "DFlash host between rounds: consumer (yield) 0.50 ms/round; commit parts {'m_a': 0.1}",
    "Tree verify stage profile (1 rounds, ms/round): {'attn': 12.0, 'mlp': 20.0}",
]
# Layer-2 fields the real engine does not print yet (fullcmp_attribution.LAYER2_YUNSHU).
LAYER2_CONTRACT = {
    "stages": {
        "decode_round": {
            "drafter": 1.2,
            "plan": 0.4,
            "copy_lookup": 0.1,
            "verify_weights": 30.0,
            "verify_attention": 8.0,
            "commit": 3.0,
        },
        "prefill": {
            "chunk_compute": 100.0,
            "eval_wait": 2.0,
            "apc_store": 1.0,
            "drafter_context": 4.0,
        },
        "ttft": {
            "template": 0.2,
            "lookup": 0.1,
            "restore": 0.0,
            "prefill": 100.0,
            "drafter_rebuild": 1.0,
            "first_round": 2.0,
        },
    },
    "host_sync_count": 3,
    "op_census": {"matmul": 120, "rope": 40},
}


def profile_enabled(engine: str) -> bool:
    return engine == "yunshu" and os.environ.get("YUNSHU_DEBUG_TREE_PROFILE") == "1"


class ProfileState(base.StubState):
    """Base state plus the profile lines once per request when the profile switch is on."""

    def log(self, lines: list[str]) -> None:
        extra = []
        if any(line.startswith("[round-profile]") for line in lines) and profile_enabled(
            self.engine
        ):
            extra = PROFILE_LINES + ["YUNSHU_LAYER2 " + json.dumps(LAYER2_CONTRACT)]
        super().log(list(lines) + extra)


def _behaviour_content(body: dict) -> str:
    if body.get("response_format"):
        return json.dumps({"city": "Paris", "ok": True})
    return "hello from the stub"


def make_handler(state: base.StubState):
    Base = base.make_handler(state)

    class Handler(Base):
        def _send(self, code, obj):
            return self._json(code, obj)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {}
            if self.path.startswith("/v1/messages"):
                return self._messages(body)
            if self.path.startswith("/v1/responses"):
                return self._responses(body)
            if (
                self.path.startswith("/v1/chat/completions")
                and not body.get("stream")
                and (body.get("tools") or body.get("response_format") or body.get("logprobs"))
            ):
                return self._special_chat(body)
            self.rfile = io.BytesIO(raw)  # the base handler reads the body again
            return super().do_POST()

        def _special_chat(self, body):
            with state.lock:
                state.round += 1
                n = state.round
            state.log(base.request_lines(state.engine, n, 1, drafted=4, accepted=3))
            prompt_tokens = len(base.prompt_text(body).split())
            message: dict = {"role": "assistant", "content": None}
            finish = "stop"
            if body.get("tools"):
                name = body["tools"][0]["function"]["name"]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{n}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps({"city": "Paris"}),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                message["content"] = _behaviour_content(body)
            choice: dict = {"index": 0, "message": message, "finish_reason": finish}
            tokens = (message.get("content") or "x").split(" ")
            if body.get("logprobs"):
                choice["logprobs"] = {
                    "content": [
                        {"token": t, "logprob": -0.25, "top_logprobs": []} for t in tokens
                    ]
                }
            return self._send(
                200,
                {
                    "id": f"chatcmpl-stub-{n}",
                    "object": "chat.completion",
                    "model": state.model,
                    "choices": [choice],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(tokens),
                        "total_tokens": prompt_tokens + len(tokens),
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                },
            )

        def _messages(self, body):
            text = _behaviour_content(body)
            return self._send(
                200,
                {
                    "id": "msg_stub",
                    "type": "message",
                    "role": "assistant",
                    "model": state.model,
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        def _responses(self, body):
            text = _behaviour_content(body)
            return self._send(
                200,
                {
                    "id": "resp_stub",
                    "object": "response",
                    "model": state.model,
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

    return Handler


def start(engine: str, model: str, log_path, port: int = 0, version: str = "stub"):
    """In-process start (behaviour tests). Returns (server, port, state)."""
    state = ProfileState(engine, model, Path(log_path), version)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state


def main(argv=None) -> int:
    """Same argv as fullcmp_stub_server.main (tfbench launches it through TFB_STUB_SCRIPT)."""
    ap = argparse.ArgumentParser(description="fullcmp profile/behaviour test stub")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", default="stub-model")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--version", default="stub")
    a = ap.parse_args(argv)
    state = ProfileState(a.engine, a.model, a.log, a.version)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state))
    # exit with the parent (the test or tfbench that launched this stub), as the base stub does
    base.watch_parent(os.getppid(), max_lifetime=base.MAX_LIFETIME_S)
    print(f"stub {a.engine} listening on 127.0.0.1:{a.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
