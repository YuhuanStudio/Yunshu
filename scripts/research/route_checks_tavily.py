"""Tavily API checks against a real server and loopback SERP/page fixtures."""

from __future__ import annotations

import json
import time

from route_checks import check, expect


@check(
    "tavily",
    "POST /tavily/search",
    "POST /tavily/extract",
    "POST /tavily/crawl",
    "POST /tavily/map",
    "POST /tavily/research",
    "GET /tavily/research/{request_id}",
    "POST /tavily/feedback",
    "POST /tavily/logs",
    "GET /tavily/usage",
    "GET /tavily/providers",
    "POST /tavily/mcp",
    needs="multi",
    served=True,
)
def tavily_routes(c):
    expect(c.fake is not None, "Tavily real check requires fixture backend")
    root = c.fake.url + "/page"
    response = c.req(
        "POST",
        "/tavily/search",
        json={
            "query": "Paris weather",
            "include_answer": True,
            "include_usage": True,
            "include_images": True,
        },
    )
    expect(response.status_code == 200, response.text)
    result = response.json()
    expect(result["results"] and isinstance(result["images"], list), result)
    expect(isinstance(result["answer"], str) and result["answer"], result)
    expect("server-timing" in response.headers, "missing stage timings")
    extracted = c.req(
        "POST",
        "/tavily/extract",
        json={"urls": [root], "query": "weather", "format": "markdown"},
    ).json()
    expect(extracted["results"] and extracted["failed_results"] == [], extracted)
    for endpoint in ("map", "crawl"):
        response = c.req("POST", "/tavily/" + endpoint, json={"url": root, "limit": 1})
        expect(
            response.status_code == 200 and response.json()["results"], response.text
        )
    task = c.req(
        "POST",
        "/tavily/research",
        json={
            "input": "What is the weather in Paris?",
            "model": "mini",
            "output_schema": {
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Answer grounded in the supplied weather sources",
                    }
                },
                "required": ["answer"],
            },
        },
    )
    expect(task.status_code == 200, task.text)
    identity = task.json()["request_id"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        response = c.req(
            "GET", "/tavily/research/" + identity, params={"include_usage": "true"}
        )
        expect(response.status_code in (200, 202), response.text)
        research = response.json()
        if research["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    expect(
        research["status"] == "completed" and isinstance(research["content"], dict),
        research,
    )
    feedback = c.req(
        "POST",
        "/tavily/feedback",
        json={"request_id": result["request_id"], "human_score": 1},
    ).json()
    expect(feedback["success"], feedback)
    logs = c.req("POST", "/tavily/logs", json={"limit": 10}).json()
    expect(logs["count"] > 0, logs)
    expect(c.req("GET", "/tavily/usage").json()["key"]["usage"] > 0, "no local usage")
    expect("providers" in c.req("GET", "/tavily/providers").json(), "missing health")
    rpc = c.req(
        "POST",
        "/tavily/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "tavily_extract", "arguments": {"urls": [root]}},
        },
    ).json()
    expect(
        not rpc["result"]["isError"]
        and json.loads(rpc["result"]["content"][0]["text"])["results"],
        rpc,
    )
    return {
        "search_results": len(result["results"]),
        "extracted": len(extracted["results"]),
        "research_status": research["status"],
        "schema_valid": True,
        "model_generated": True,
    }
