"""Server-side tool checks of the route registry (web search, web fetch, MCP connector) against a
local fake provider. Imported at the bottom of route_checks.py, which registers them."""

from __future__ import annotations

import json

from route_checks import Ctx, check, expect


class FakeBackend:
    """One loopback HTTP server standing in for a SearXNG instance (`GET /search`), an MCP server
    (`POST /mcp`, JSON-RPC) and a plain page (`GET /page`); it records every request so a check can
    prove the Yunshu server really called out (or, for the SSRF guard, did not)."""

    def __init__(self, port: int):
        import http.server
        import threading

        self.searches: list[str] = []
        self.mcp: list[dict] = []
        self.pages: list[str] = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                import urllib.parse

                u = urllib.parse.urlparse(self.path)
                if u.path == "/search":
                    q = urllib.parse.parse_qs(u.query).get("q", [""])[0]
                    outer.searches.append(q)
                    self._send(
                        200,
                        {
                            "results": [
                                {
                                    "url": "https://example.org/paris-weather",
                                    "title": "Paris weather today",
                                    "content": "Paris is sunny, 21 degrees Celsius.",
                                },
                                {
                                    "url": "https://example.org/paris-forecast",
                                    "title": "Paris forecast",
                                    "content": "Dry all week in Paris.",
                                },
                            ]
                        },
                    )
                elif u.path == "/page":
                    outer.pages.append(self.path)
                    self._send(
                        200, b"<html><body>SECRET PAGE</body></html>", "text/html"
                    )
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                msg = json.loads(self.rfile.read(n) or b"{}")
                outer.mcp.append(msg)
                m, i = msg.get("method"), msg.get("id")
                if i is None:
                    self._send(202, b"", "text/plain")
                elif m == "initialize":
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": i,
                            "result": {
                                "protocolVersion": "2025-03-26",
                                "capabilities": {"tools": {}},
                                "serverInfo": {"name": "fake-mcp", "version": "1"},
                            },
                        },
                    )
                elif m == "tools/list":
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": i,
                            "result": {
                                "tools": [
                                    {
                                        "name": "echo_upper",
                                        "description": "Upper-case the text.",
                                        "inputSchema": {
                                            "type": "object",
                                            "properties": {"text": {"type": "string"}},
                                            "required": ["text"],
                                        },
                                    }
                                ]
                            },
                        },
                    )
                elif m == "tools/call":
                    t = (msg.get("params", {}).get("arguments") or {}).get("text", "")
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": i,
                            "result": {
                                "content": [{"type": "text", "text": str(t).upper()}]
                            },
                        },
                    )
                else:
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": i,
                            "error": {"code": -32601, "message": "no"},
                        },
                    )

        self.port = port
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


SEARCH_PROMPT = "Search the web for the weather in Paris."
SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}
FORCE_SEARCH = {"type": "tool", "name": "web_search"}


@check("web_search_provider", "POST /v1/messages", "POST /v1/responses", needs="multi")
def _web_search_provider(c: Ctx):
    """web_search with a configured provider (a fake SearXNG): the server runs the search inside the
    generation loop and the typed SDK objects carry the results."""
    expect(c.fake is not None, "needs the fake backend")
    before = len(c.fake.searches)
    m = c.an.messages.create(
        model=c.model,
        max_tokens=512,
        tool_choice=FORCE_SEARCH,
        tools=[{**SEARCH_TOOL, "max_uses": 2}],
        messages=[{"role": "user", "content": SEARCH_PROMPT}],
    )
    kinds = [b.type for b in m.content]
    uses = [b for b in m.content if b.type == "server_tool_use"]
    results = [b for b in m.content if b.type == "web_search_tool_result"]
    expect(uses and results, f"no server tool blocks: {kinds}")
    ok = [r for r in results if isinstance(r.content, list)]
    expect(
        ok, f"no successful web_search_tool_result: {[r.content for r in results][:2]}"
    )
    expect(
        ok[0].content[0].url.startswith("https://example.org/")
        and ok[0].content[0].title,
        f"result block {ok[0].content[0]}",
    )
    expect(len(ok) <= 2, f"{len(ok)} searches ran with max_uses=2")
    ran = len(c.fake.searches) - before
    expect(ran == len(ok), f"backend saw {ran} searches, response shows {len(ok)}")
    expect(
        all(isinstance(u.input, dict) and u.input.get("query") for u in uses),
        "server_tool_use without a query",
    )
    su = getattr(m.usage, "server_tool_use", None)
    expect(
        su is not None and su.web_search_requests >= 1, f"usage.server_tool_use {su}"
    )
    c.notes["web_search_blocks"] = kinds
    # streaming: the same blocks arrive as events and assemble into the same shape
    with c.an.messages.stream(
        model=c.model,
        max_tokens=512,
        tool_choice=FORCE_SEARCH,
        tools=[{**SEARCH_TOOL, "max_uses": 1}],
        messages=[{"role": "user", "content": SEARCH_PROMPT}],
    ) as st:
        fin = st.get_final_message()
    expect(
        any(b.type == "web_search_tool_result" for b in fin.content),
        f"stream blocks {[b.type for b in fin.content]}",
    )
    # OpenAI Responses
    before = len(c.fake.searches)
    r = c.oa.responses.create(
        model=c.model,
        input=SEARCH_PROMPT,
        tools=[{"type": "web_search"}],
        tool_choice="required",
        max_output_tokens=512,
    )
    calls = [o for o in r.output if o.type == "web_search_call"]
    expect(
        calls and all(o.status == "completed" for o in calls),
        f"output {[o.type for o in r.output]}",
    )
    expect(
        len(c.fake.searches) > before,
        "Responses web_search never reached the provider",
    )
    c.notes["web_search_responses_blocks"] = [o.type for o in r.output]


@check("web_fetch_ssrf", "POST /v1/messages", needs="multi")
def _web_fetch_ssrf(c: Ctx):
    """web_fetch refuses a loopback page (SSRF guard) with the API's error block, and the page is
    never requested."""
    expect(c.fake is not None, "needs the fake backend")
    m = c.an.messages.create(
        model=c.model,
        max_tokens=256,
        tool_choice={"type": "tool", "name": "web_fetch"},
        tools=[{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1}],
        messages=[
            {"role": "user", "content": f"Fetch {c.fake.url}/page and quote it."}
        ],
        extra_headers={"anthropic-beta": "web-fetch-2025-09-10"},
    )
    res = [b for b in m.content if b.type == "web_fetch_tool_result"]
    expect(res, f"no web_fetch_tool_result: {[b.type for b in m.content]}")
    first = res[0].content
    expect(
        getattr(first, "type", None) == "web_fetch_tool_result_error"
        and first.error_code,
        f"loopback fetch not refused: {first}",
    )
    expect(not c.fake.pages, f"the loopback page was requested: {c.fake.pages}")
    c.notes["web_fetch_error_code"] = first.error_code


@check("mcp_connector", "POST /v1/messages", "POST /v1/responses", needs="multi")
def _mcp_connector(c: Ctx):
    """The MCP connector: the server connects to a (fake) MCP server, lists its tools and calls them
    on the model's behalf, in both dialects."""
    expect(c.fake is not None, "needs the fake backend")
    n0 = len(c.fake.mcp)
    r = c.oa.responses.create(
        model=c.model,
        input="Use the echo_upper tool on the text hello.",
        tools=[
            {
                "type": "mcp",
                "server_label": "fake",
                "server_url": c.fake.url + "/mcp",
                "require_approval": "never",
            }
        ],
        tool_choice="required",
        max_output_tokens=512,
    )
    types = [o.type for o in r.output]
    lst = [o for o in r.output if o.type == "mcp_list_tools"]
    expect(
        lst and [t.name for t in lst[0].tools] == ["echo_upper"],
        f"mcp_list_tools {types}",
    )
    methods = [m.get("method") for m in c.fake.mcp[n0:]]
    expect(
        "initialize" in methods and "tools/list" in methods,
        f"fake MCP saw {methods}",
    )
    calls = [o for o in r.output if o.type == "mcp_call"]
    if calls:
        expect(
            calls[0].server_label == "fake" and calls[0].name == "echo_upper",
            f"mcp_call {calls[0]}",
        )
        expect(calls[0].error is None, f"mcp_call error {calls[0].error}")
        expect(
            "HELLO" in (calls[0].output or "").upper(),
            f"mcp output {calls[0].output!r}",
        )
    else:
        c.notes["mcp_responses_note"] = f"model did not call the tool; output {types}"
    c.notes["mcp_responses"] = types
    # Anthropic connector (beta mcp-client)
    n1 = len(c.fake.mcp)
    m = c.an.beta.messages.create(
        model=c.model,
        max_tokens=512,
        betas=["mcp-client-2025-11-20"],
        mcp_servers=[{"type": "url", "url": c.fake.url + "/mcp", "name": "fake"}],
        tools=[{"type": "mcp_toolset", "mcp_server_name": "fake"}],
        tool_choice={"type": "any"},
        messages=[
            {"role": "user", "content": "Use the echo_upper tool on the text hello."}
        ],
    )
    methods = [x.get("method") for x in c.fake.mcp[n1:]]
    expect(
        "tools/list" in methods, f"Anthropic connector never listed tools: {methods}"
    )
    kinds = [b.type for b in m.content]
    c.notes["mcp_messages"] = kinds
    if any(b.type == "mcp_tool_use" for b in m.content):
        res = [b for b in m.content if b.type == "mcp_tool_result"]
        expect(res and not res[0].is_error, f"mcp_tool_result {res[:1]}")
