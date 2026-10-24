"""Five-question local-engine documentation smoke. Run only through gpuq.

Starts the existing tfbench server driver and calls the console ASGI app, retaining
answers/sources and a final complete record. Parser/contract tests use no MLX.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

QUESTIONS = [
    ("en", "How do I start Yunshu with a local model?"),
    ("en", "How do I authenticate API requests using a bearer token?"),
    ("en", "How do I stream chat completions?"),
    ("zh-TW", "什麼是 APC 提示詞快取？"),
    ("zh-CN", "如何使用 OpenAI Python SDK 连接 Yunshu？"),
]


def parse_answer(wire: str) -> dict:
    """Fail closed: a useful answer needs sources, visible text and terminal done."""
    result: dict = {"answer": "", "sources": [], "complete": False}
    for frame in wire.split("\n\n"):
        lines = frame.splitlines()
        kind = next((line[7:] for line in lines if line.startswith("event: ")), "")
        data = next((line[6:] for line in lines if line.startswith("data: ")), None)
        if data is None:
            continue
        value = json.loads(data)
        if kind == "sources":
            result.update(value)
        elif kind == "token":
            result["answer"] += value["text"]
        elif kind == "error":
            raise RuntimeError(f"docs stream error: {value}")
        elif kind == "done":
            result["complete"] = True
    if not result["complete"] or not result["answer"].strip() or not result["sources"]:
        raise RuntimeError("incomplete docs answer")
    return result


async def questions(engine_url: str, static_dir: Path, out: Path) -> None:
    import httpx

    from yunshu_console.app import create_app
    from yunshu_engine import settings

    settings.set_override("YUNSHU_DOCS_ASK_MAX_TOKENS", 384)

    app = create_app(engine_url, poll=False, static_dir=static_dir)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://console", timeout=240
    ) as client:
        with out.open("w") as output:
            for number, (language, question) in enumerate(QUESTIONS):
                response = await client.post(
                    "/console/api/docs/ask",
                    headers={"Authorization": "Bearer k"},
                    json={
                        "tab_id": "docsask-smoke",
                        "question": question,
                        "language": language,
                    },
                )
                response.raise_for_status()
                record = {
                    "number": number,
                    "question": question,
                    "language": language,
                    **parse_answer(response.text),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)
            output.write(
                json.dumps({"complete": True, "questions": len(QUESTIONS)}) + "\n"
            )
    await app.state.http.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        parser.error("model directory missing")
    root = Path(__file__).resolve().parents[2]
    static_dir = root / "python/yunshu_gateway/console_static"
    if not (static_dir / "docs-assets/ask-en.json").is_file():
        parser.error("build frontend first")
    import tfbench

    args.out.parent.mkdir(parents=True, exist_ok=True)
    server = tfbench.start_server(
        "yunshu",
        {
            "YUNSHU_CONSOLE": "0",
            "YUNSHU_AUTH_TOKEN": "k",
            "YUNSHU_AUTH_DISABLED": "0",
            "YUNSHU_VLM_DRAFT": "off",
            "YUNSHU_VLM_APC_DISK": "0",
            "YUNSHU_DOCS_ASK_MAX_TOKENS": "384",
        },
        "docsask-smoke",
        model=args.model,
        ctx_tokens=8192,
    )
    try:
        asyncio.run(questions(server.url, static_dir, args.out))
    finally:
        server.kill()


if __name__ == "__main__":
    main()
