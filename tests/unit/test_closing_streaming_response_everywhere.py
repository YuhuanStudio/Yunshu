"""Streaming inference responses must close their body iterator themselves when ASGI `send`
fails after a client disconnect (ClosingStreamingResponse); a plain StreamingResponse leaves
the nested generators (cancel_event, lease release, staged-file unlink) to GC finalizers.
Static replays of an already-built payload hold no engine resources and may stay plain."""

import ast
import pathlib

ROUTERS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "yunshu_gateway"
    / "routers"
)

# first-argument names that only replay a finished payload / a canned error
STATIC_BODIES = {"replay", "_replay", "frames", "gen", "_model_not_found_stream"}


def _plain_streaming_calls(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "StreamingResponse"
        ):
            arg = node.args[0] if node.args else None
            name = None
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                name = arg.func.id
            yield node.lineno, name


def test_inference_streams_use_closing_response():
    offenders = []
    for fname in ("chat.py", "completions.py", "responses.py", "anthropic.py"):
        for lineno, name in _plain_streaming_calls(ROUTERS / fname):
            if name not in STATIC_BODIES:
                offenders.append(f"{fname}:{lineno} StreamingResponse({name}(...))")
    assert not offenders, offenders
