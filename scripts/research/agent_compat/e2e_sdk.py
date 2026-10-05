"""Official-SDK checks of the server-side tools against a real Yunshu server (run with the SDK venv).

    python e2e_sdk.py --url http://127.0.0.1:18990 --model <id> --search-fact "Cloud Book 42" \
        --page-url http://127.0.0.1:P/page/yunshu-search --mcp-url http://127.0.0.1:Q/mcp

Anthropic: messages.create / messages.stream with web_search_20250305, beta web_fetch_20250910 and
beta mcp_servers. OpenAI: responses.create with web_search and an mcp tool (both streamed and not).
Prints one JSON line per check and exits non-zero if any fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import anthropic
import openai

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--search-fact", default="Cloud Book 42")
ap.add_argument("--page-url", default="")
ap.add_argument("--mcp-url", default="")
ap.add_argument("--skip", default="")
a = ap.parse_args()

results: list[dict] = []
ac = anthropic.Anthropic(base_url=a.url, api_key="k", timeout=600)
oc = openai.OpenAI(base_url=a.url + "/v1", api_key="k", timeout=600)


def check(name, fn):
    if name in a.skip.split(","):
        return
    t0 = time.time()
    try:
        info = fn()
        results.append(
            {
                "check": name,
                "ok": True,
                "secs": round(time.time() - t0, 1),
                **(info or {}),
            }
        )
    except Exception as e:  # noqa: BLE001
        results.append(
            {
                "check": name,
                "ok": False,
                "secs": round(time.time() - t0, 1),
                "error": repr(e)[:400],
            }
        )
    print(json.dumps(results[-1]), flush=True)


def kinds(msg):
    return [b.type for b in msg.content]


def anth_search():
    m = ac.messages.create(
        model=a.model,
        max_tokens=4000,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        messages=[
            {
                "role": "user",
                "content": "Search the web: what is the reference release code name of Yunshu? "
                "Answer in one sentence and cite the source with [1].",
            }
        ],
    )
    ks = kinds(m)
    assert "server_tool_use" in ks and "web_search_tool_result" in ks, ks
    res = next(b for b in m.content if b.type == "web_search_tool_result")
    assert isinstance(res.content, list) and res.content, res.content
    text = "".join(getattr(b, "text", "") for b in m.content if b.type == "text")
    assert a.search_fact.lower() in text.lower(), text
    cites = [c for b in m.content if b.type == "text" for c in (b.citations or [])]
    assert m.usage.server_tool_use.web_search_requests >= 1
    xs = ((m.model_extra or {}).get("x_yunshu") or {}).get("server_tools") or {}
    return {
        "kinds": ks,
        "round_usage": xs.get("round_usage"),
        "citations": len(cites),
        "text": text[:200],
        "usage": m.usage.model_dump(),
    }


def anth_search_followup():
    """Second turn: the earlier server_tool_use / web_search_tool_result blocks go back in the history."""
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    q1 = "Search the web: what is the reference release code name of Yunshu? Answer in one sentence."
    m1 = ac.messages.create(
        model=a.model,
        max_tokens=4000,
        tools=tools,
        messages=[{"role": "user", "content": q1}],
    )
    assert "web_search_tool_result" in kinds(m1), kinds(m1)
    hist = [
        {"role": "user", "content": q1},
        {
            "role": "assistant",
            "content": [b.model_dump(exclude_none=True) for b in m1.content],
        },
        {
            "role": "user",
            "content": "Without searching again: which page URL did that answer come from? Give the URL.",
        },
    ]
    m2 = ac.messages.create(model=a.model, max_tokens=4000, tools=tools, messages=hist)
    text = "".join(getattr(b, "text", "") for b in m2.content if b.type == "text")
    assert "/page/" in text or "127.0.0.1" in text, text
    return {"kinds": kinds(m2), "text": text[:200]}


def anth_search_stream():
    with ac.messages.stream(
        model=a.model,
        max_tokens=4000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[
            {
                "role": "user",
                "content": "Search the web for the Yunshu reference release code name and tell me it.",
            }
        ],
    ) as s:
        events = [e.type for e in s]
        m = s.get_final_message()
    assert "server_tool_use" in kinds(m) and "web_search_tool_result" in kinds(m), (
        kinds(m)
    )
    text = "".join(getattr(b, "text", "") for b in m.content if b.type == "text")
    assert a.search_fact.lower() in text.lower(), text
    return {"kinds": kinds(m), "n_events": len(events), "text": text[:200]}


def anth_fetch():
    m = ac.beta.messages.create(
        model=a.model,
        max_tokens=4000,
        betas=["web-fetch-2025-09-10"],
        tools=[{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 2}],
        messages=[
            {
                "role": "user",
                "content": f"Fetch {a.page_url} and tell me the verification word on that page.",
            }
        ],
    )
    ks = kinds(m)
    assert "web_fetch_tool_result" in ks, ks
    text = "".join(getattr(b, "text", "") for b in m.content if b.type == "text")
    assert "tangerine-7" in text.lower(), text
    return {"kinds": ks, "text": text[:200]}


def anth_fetch_public():
    """web_fetch against a real public site (network access from the server, SSRF guard on)."""
    m = ac.beta.messages.create(
        model=a.model,
        max_tokens=4000,
        betas=["web-fetch-2025-09-10"],
        tools=[{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 2}],
        messages=[
            {
                "role": "user",
                "content": "Fetch https://example.com/ and quote its main heading.",
            }
        ],
    )
    res = [b for b in m.content if b.type == "web_fetch_tool_result"]
    assert res, kinds(m)
    c = res[0].content
    assert getattr(c, "type", "") == "web_fetch_result", c
    assert "example domain" in (c.content.title or "").lower(), c.content.title
    assert "documentation examples" in c.content.source.data.lower(), (
        c.content.source.data[:200]
    )
    text = "".join(getattr(b, "text", "") for b in m.content if b.type == "text")
    return {"kinds": kinds(m), "title": c.content.title, "text": text[:160]}


def anth_mcp():
    m = ac.beta.messages.create(
        model=a.model,
        max_tokens=4000,
        betas=["mcp-client-2025-04-04"],
        mcp_servers=[{"type": "url", "url": a.mcp_url, "name": "tiny"}],
        messages=[
            {
                "role": "user",
                "content": "Use the add tool to compute 19 + 23 and give me the result.",
            }
        ],
    )
    ks = kinds(m)
    assert "mcp_tool_use" in ks and "mcp_tool_result" in ks, ks
    text = "".join(getattr(b, "text", "") for b in m.content if b.type == "text")
    assert "42" in text, text
    return {"kinds": ks, "text": text[:200]}


def resp_search(stream):
    kw = dict(
        model=a.model,
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        input="Search the web: what is the reference release code name of Yunshu? Answer in one sentence.",
    )
    if stream:
        with oc.responses.stream(**kw) as s:
            n = sum(1 for _ in s)
            r = s.get_final_response()
    else:
        r = oc.responses.create(**kw)
        n = 0
    types = [o.type for o in r.output]
    assert "web_search_call" in types, types
    assert a.search_fact.lower() in r.output_text.lower(), (
        types,
        r.output_text,
        r.status,
        r.incomplete_details,
    )
    return {"types": types, "events": n, "text": r.output_text[:200]}


def resp_mcp():
    r = oc.responses.create(
        model=a.model,
        tools=[
            {
                "type": "mcp",
                "server_label": "tiny",
                "server_url": a.mcp_url,
                "require_approval": "never",
            }
        ],
        input="Use the add tool to compute 19 + 23 and give me the result.",
    )
    types = [o.type for o in r.output]
    assert "mcp_list_tools" in types and "mcp_call" in types, types
    assert "42" in r.output_text, r.output_text
    return {"types": types, "text": r.output_text[:200]}


check("anthropic_web_search", anth_search)
check("anthropic_web_search_stream", anth_search_stream)
check("anthropic_web_search_followup", anth_search_followup)
if a.page_url:
    check("anthropic_web_fetch", anth_fetch)
check("anthropic_web_fetch_public", anth_fetch_public)
if a.mcp_url:
    check("anthropic_mcp", anth_mcp)
check("responses_web_search", lambda: resp_search(False))
check("responses_web_search_stream", lambda: resp_search(True))
if a.mcp_url:
    check("responses_mcp", resp_mcp)
sys.exit(0 if all(r["ok"] for r in results) else 1)
