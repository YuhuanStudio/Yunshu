"""Line-based offline ASGI bridge for Node SDK tests (no listening TCP socket)."""

import asyncio
import json
import sys

import httpx
from tavily_fixture import create_fixture


async def main():
    app = create_fixture()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://fixture"
    ) as client:
        while line := await asyncio.to_thread(sys.stdin.readline):
            try:
                request = json.loads(line)
                response = await client.request(
                    request.get("method", "POST"),
                    request["path"],
                    json=request.get("body"),
                    headers=request.get("headers"),
                )
                value = {
                    "status": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.text,
                }
            except Exception as exc:
                value = {
                    "status": 500,
                    "headers": {},
                    "body": json.dumps({"detail": {"error": str(exc)}}),
                }
            print(json.dumps(value), flush=True)
    await app.state.tavily_service.close()


if __name__ == "__main__":
    asyncio.run(main())
