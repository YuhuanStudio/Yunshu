# Web search and fetch

Coding agents can request server-side search through Responses `web_search` /
`web_search_preview` or Anthropic `web_search_20250305`. Anthropic also supports
`web_fetch_20250910`. Results use their dialect's tool-result and citation structures.
See [agent compatibility](AGENT_COMPAT.md) for client coverage and limitations.

## Select providers

`YUNSHU_WEB_SEARCH_PROVIDER=auto` runs lightweight DuckDuckGo HTML, Wikipedia and
configured keyed providers in parallel, with deduplication, reciprocal-rank fusion
and provider health backoff. DuckDuckGo is best effort and may block automation.
SearXNG is optional; configure an instance explicitly to use it.

```bash
yunshu serve -m <model> --set web_search_provider=auto
# Disable server-side search:
yunshu serve -m <model> --set web_search_provider=none
```

Available selectors include `searxng`, `brave`, `tavily`, `exa`, `serper`, `perplexity`,
`ddg_html`, `wikipedia`, `mojeek`, `marginalia` and `mwmbl`. Keyed services require
credentials; consult [Configuration](../CONFIGURATION.md). Mwmbl is an explicit
noncommercial opt-in (`YUNSHU_WEB_MWMBL`), with attribution/share-alike terms.
`YUNSHU_WEB_KEYLESS=0` disables the keyless sources. Provider health is available
through `yunshu config` and `/tavily/providers` without storing query text in the
health snapshot.

## Fetch and privacy

Queries, destination URLs and the client's IP leave the Mac to reach providers/pages.
Search is not offline inference. `YUNSHU_WEB_FETCH=0` disables server-side fetch.
Fetch blocks private, loopback and link-local addresses by default, including after
DNS resolution and redirects; keep `YUNSHU_WEB_FETCH_ALLOW_PRIVATE` off for this policy.
Body, timeout and extracted-text limits are registered settings.

Plain Tavily search uses lexical ranking and deduplication without an LLM. Optional
answers and research synthesis use the served chat model. Origin-page enrichment is
an opt-in (`YUNSHU_WEB_RESEARCH`); model-backed enrichment never loads a second model
implicitly. Optional Chromium rendering is off by default and requires the
`web-render` extra plus an installed browser.

## Tavily-compatible clients

The local retrieval API is mounted at `http://127.0.0.1:8000/tavily`, with search,
extract, crawl, map, research, feedback, logs, usage and provider routes. Native MCP
is at `/tavily/mcp`. It is a local implementation: cloud credit billing and proprietary
ranking/research behavior are not reproduced. For SDK configuration, authentication,
limits and reproducible checks, see [Tavily API](TAVILY.md).
