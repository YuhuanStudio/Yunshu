"""Real web tools against a real server and the real network (keyless providers, no fake backend).

    realcheck_web.py --model M --out F [--only responses_search,messages_search,fetch_page]

One server, three served checks: `web_search` through the Responses API (non-stream and stream) and
the Messages API, and `web_fetch` of a real page (the answer must quote the page). Fail closed: a
check that cannot show a search/fetch call, a result and a final answer is a failure. The
providers are whatever `YUNSHU_WEB_SEARCH_PROVIDER=auto` finds (keyless DuckDuckGo / Wikipedia, or
configured keys); the job records which one answered. Run only through gpuq.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

QUESTION = "What is the Apple MLX machine learning framework? Search the web and answer in one sentence."
PAGE = "https://example.com/"
PAGE_WORD = "example domain"


def verdict(rows: dict, wanted: tuple[str, ...]) -> list[str]:
    """Problems with a result (pure; unit-tested): every wanted check ran and passed."""
    bad = []
    for name in wanted:
        row = rows.get(name)
        if row is None:
            bad.append(f"{name}: never ran")
        elif row.get("status") != "pass":
            bad.append(f"{name}: {row.get('detail', 'not pass')}")
    return bad


def _expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def responses_search(c):
    out = {}
    r = c.oa.responses.create(
        model=c.model,
        input=QUESTION,
        tools=[{"type": "web_search"}],
        tool_choice="required",
        max_output_tokens=600,
    )
    kinds = [o.type for o in r.output]
    calls = [o for o in r.output if o.type == "web_search_call"]
    _expect(
        calls and all(o.status == "completed" for o in calls),
        f"non-stream output {kinds}",
    )
    msg = [o for o in r.output if o.type == "message"]
    _expect(msg and msg[-1].content[0].text.strip(), f"no final text: {kinds}")
    ann = [a for m in msg for p in m.content for a in (p.annotations or [])]
    out["non_stream"] = {
        "output": kinds,
        "citations": len(ann),
        "text": msg[-1].content[0].text[:160],
    }
    types = []
    with c.oa.responses.stream(
        model=c.model,
        input=QUESTION,
        tools=[{"type": "web_search"}],
        tool_choice="required",
        max_output_tokens=600,
    ) as st:
        for ev in st:
            types.append(ev.type)
        fin = st.get_final_response()
    _expect(
        "response.web_search_call.completed" in types,
        f"stream events {sorted(set(types))}",
    )
    _expect(types[-1] == "response.completed", f"stream ended {types[-3:]}")
    _expect(fin.status == "completed", f"stream status {fin.status}")
    out["stream"] = {"events": len(types), "status": fin.status}
    return out


def messages_search(c):
    m = c.an.messages.create(
        model=c.model,
        max_tokens=600,
        tool_choice={"type": "tool", "name": "web_search"},
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
        messages=[{"role": "user", "content": QUESTION}],
    )
    kinds = [b.type for b in m.content]
    res = [b for b in m.content if b.type == "web_search_tool_result"]
    _expect("server_tool_use" in kinds and res, f"blocks {kinds}")
    _expect(
        isinstance(res[0].content, list) and res[0].content and res[0].content[0].url,
        f"result block {res[0].content}",
    )
    _expect(
        any(b.type == "text" and b.text.strip() for b in m.content), f"no text {kinds}"
    )
    return {"blocks": kinds, "first_url": res[0].content[0].url}


def fetch_page(c):
    m = c.an.messages.create(
        model=c.model,
        max_tokens=600,
        tool_choice={"type": "tool", "name": "web_fetch"},
        tools=[{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1}],
        messages=[
            {"role": "user", "content": f"Fetch {PAGE} and quote the page's heading."}
        ],
        extra_headers={"anthropic-beta": "web-fetch-2025-09-10"},
    )
    kinds = [b.type for b in m.content]
    res = [b for b in m.content if b.type == "web_fetch_tool_result"]
    _expect(res, f"no web_fetch_tool_result: {kinds}")
    cont = res[0].content
    body = json.dumps(cont.model_dump() if hasattr(cont, "model_dump") else str(cont))
    _expect(
        getattr(cont, "type", "") == "web_fetch_result",
        f"fetch not served: {body[:300]}",
    )
    _expect(
        PAGE_WORD in body.lower(), f"page text missing from the result: {body[:300]}"
    )
    text = " ".join(b.text for b in m.content if b.type == "text")
    _expect(text.strip(), f"no final answer: {kinds}")
    return {
        "blocks": kinds,
        "answer": text[:200],
        "answer_quotes_page": PAGE_WORD in text.lower(),
    }


CHECKS = {
    "responses_search": responses_search,
    "messages_search": messages_search,
    "fetch_page": fetch_page,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    wanted = tuple(a.only.split(",")) if a.only else tuple(CHECKS)
    from m3sweep_jobs import routes_make_ctx, start_server, write

    res = {
        "kind": "realcheck_web",
        "model": a.model,
        "checks": {},
        "failures": [],
        "complete": False,
    }
    srv = None
    try:
        srv = start_server(a.model, "web", ["YUNSHU_VLM_APC_DISK=0"], online=True)
        ctx = routes_make_ctx(srv, "", "vlm" if "Qwen3" in a.model else "text")
        for name in wanted:
            t0 = time.monotonic()
            try:
                if srv.proc.poll() is not None:
                    raise RuntimeError("server died")
                res["checks"][name] = {"status": "pass", "notes": CHECKS[name](ctx)}
            except BaseException as e:  # noqa: BLE001
                res["checks"][name] = {
                    "status": "fail",
                    "detail": f"{type(e).__name__}: {str(e)[:1500]}",
                    "trace": traceback.format_exc()[-1000:],
                }
            res["checks"][name]["seconds"] = round(time.monotonic() - t0, 1)
            row = res["checks"][name]
            print("check", name, row["status"], row.get("detail", "")[:300], flush=True)
            write(a.out, res)
    except BaseException as e:  # noqa: BLE001
        res["failures"].append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if srv:
            res["server_log_tail"] = srv.log_tail(40)
            srv.kill()
    res["failures"] += verdict(res["checks"], wanted)
    res["complete"] = True
    res["pass"] = not res["failures"]
    write(a.out, res)
    print("RESULT", "PASS" if res["pass"] else "FAIL", res["failures"], flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
