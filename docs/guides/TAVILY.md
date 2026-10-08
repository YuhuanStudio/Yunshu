# Local Tavily API

Yunshu mounts its Tavily-compatible API at **`http://localhost:8000/tavily`**. The Python, JavaScript and
LangChain clients append endpoint paths to the configured base URL; there is no `/v1` in this API.

```python
from tavily import TavilyClient, AsyncTavilyClient
client = TavilyClient(api_key="tvly-local", api_base_url="http://localhost:8000/tavily")
async_client = AsyncTavilyClient(api_key="tvly-local", api_base_url="http://localhost:8000/tavily")
print(client.search("MLX inference", include_usage=True))
```

```javascript
const { tavily } = require('@tavily/core');
const client = tavily({apiKey: 'tvly-local', apiBaseURL: 'http://localhost:8000/tavily'});
```

Configure `YUNSHU_AUTH_TOKEN=tvly-local` to require that key. The prefix is accepted verbatim, not stripped.
Legacy body `api_key` works too. When Yunshu auth is disabled/unconfigured the normal local open policy applies.
An injected async Python HTTP client must itself use this full base URL: the SDK preserves its existing URL.
LangChain's `TavilySearchAPIWrapper` accepts `api_base_url`.

Endpoints: POST `/search`, `/extract`, `/crawl`, `/map`, `/research`, `/feedback`, `/logs`; GET
`/research/{request_id}`, `/usage`, `/providers`. Errors use `detail.error`, with 400/401/403/404/413/429/500
as applicable and 422 for field validation. Cloud-only credit/plan statuses 432/433 are not fabricated.
Product responses carry request IDs, elapsed seconds and informational local credits. Array fields are stable.
Search/extract/crawl/map and research polling return `usage` only with `include_usage=true`;
internal accounting continues even when usage is omitted from the response.

## Retrieval and latency

The SERP remains a provider. Built-in metasearch runs direct DDG HTML and Wikipedia, plus configured
keyed sources, in parallel; reciprocal-rank fusion deduplicates candidates. DDG is best effort and can block
automation. SearXNG is optional and requires an explicit instance, never required or recommended. Query text
leaves the machine to these providers. `yunshu config` shows the most recent query-free provider health snapshot;
`/tavily/providers` shows live health. Three consecutive failures trigger bounded exponential backoff; a
half-open probe restores a recovered source. Captcha/rate limits back off immediately.

Mwmbl is available via `YUNSHU_WEB_MWMBL=1` or explicit `provider=mwmbl`. Its dataset is
CC-BY-NC-SA 4.0; retain attribution and observe the noncommercial/share-alike terms
([official terms](https://api.mwmbl.org/static/terms-and-conditions/)). It is not silently enabled for
commercial/general use. No Mwmbl implementation code is copied.

Mojeek is available with `YUNSHU_MOJEEK_API_KEY`; Marginalia with
`YUNSHU_MARGINALIA_API_KEY`. Configured keys participate in auto metasearch, or select
`YUNSHU_WEB_SEARCH_PROVIDER=mojeek|marginalia` explicitly. Neither adapter loads a model.
Mojeek forwards safe-search, country/language boosts and date bounds; its adapter requests
at most ten candidates to fit the entry API plan. [Mojeek API](https://www.mojeek.com/support/api/search/).
Marginalia forwards safe-search and result count. There is no implicit `public` key;
setting it explicitly opts into the shared development quota. Noncommercial keys serve
CC-BY-NC-SA 4.0 data; commercial keys have separate terms.
[Marginalia API and terms](https://about.marginalia-search.com/article/api/).
Common Crawl's URL index and Wikipedia dumps require corpus ingestion and a local full-text
index; they are not keyword SERP endpoints and are not downloaded or enabled automatically.

Plain search uses BM25, provider priors, exact spans and deduplication, with **no LLM calls**. Each excerpt is
at most 500 characters; excerpts join with ` [...] `. Depth controls fetch breadth and the total retrieval
budget: ultra-fast 0.7 s (provider text), fast 1 s / 3 pages, basic 1.3 s / 6 pages, advanced 3 s / 20 pages.
These are cancellation budgets, not guaranteed end-to-end p50: DNS, politeness, provider availability and
requested generation affect actual latency. `Server-Timing` reports SERP, fetch, IR and generation durations.
Tavily's official depth guide gives relative ordering, not a numeric SLA
([reference](https://docs.tavily.com/documentation/best-practices/best-practices-search)).

Time/topic/country/language/safe-search are pushed to providers that support them; dates and strict language
are also checked locally. Finance is a best-effort news vertical, not a financial index. Unknown dates survive
unless strict filtering is requested. Exact quoted phrases require a successfully fetched page. The normalized
score is local relevance, not calibrated to Tavily's proprietary probability.

Extract handles static HTML/text/JSON and bounded PDF bodies, preserves available markdown/code/tables, and
isolates failures. Crawl/map use BFS, regex selectors with timeouts, depth/breadth/global caps, robots and
host politeness. External links may be listed by map, but are never recursively followed. `instructions` uses
cheap lexical relevance until a neural backend is measured.

Advanced extract/crawl/map can render short JavaScript shells with `YUNSHU_WEB_RENDER=1`.
Install the optional extra (`uv sync --extra web-render`) and a local browser
(`uv run playwright install chromium`) first; requests never install a browser.
Rendering is off by default, uses one headless Chromium at a time, and only replays same-origin GET
documents/scripts/fetches through the normal checked/pinned fetcher. Service workers, WebSockets,
downloads, cross-origin resources and redirected resource bodies are blocked. Robots still apply to
crawl/map. The request timeout covers static fetch plus rendering. Missing browsers or a timeout preserve
the static result. CDN-dependent applications and protected-site bypass are not supported.
[Playwright interception and service workers](https://playwright.dev/python/docs/api/class-browsercontext#browser-context-route).

## Generation, research and MCP

`include_answer` calls the served chat model only when requested. Research plans queries with the served LLM,
searches/extracts evidence, then synthesizes with citations. Output schemas use the existing chat constrained
JSON decoding and are validated afterward. mini/pro differ in query caps; pro is serial on the single GPU,
not Tavily's proprietary multi-agent implementation. Async tasks support poll and SSE with keepalives/tool
frames/content/sources/done. Registry (100 tasks), feedback (1000) and logs (10000) are bounded in process and
are lost on restart. Local credits estimate effort; they do not bill or enforce cloud plans. Files support
base64 txt/md/json with an 80000-word combined cap. Bibliographies support numbered/APA/MLA/Chicago formatting.

The native stateless MCP endpoint is `/tavily/mcp`, exposing underscore and hyphen Tavily tool aliases.
**Upstream `tavily-mcp` 0.2.22 hard-codes api.tavily.com and has no base URL setting**
([source](https://github.com/tavily-ai/tavily-mcp/blob/main/src/index.ts)). REST compatibility cannot remove that
client limitation. Use native MCP for real traffic. The unmodified upstream stdio server is also driven by the official MCP client against the offline fixture through a test-only axios adapter preload (`scripts/research/tavily_mcp_upstream.mjs`): all five tools answer. Image metadata comes from
fetched pages; `include_images` additionally queries configured SearXNG's image category or Brave's image
endpoint. Source-domain filters apply, URLs are deduplicated, and a failed image vertical does not fail search.
There is no image-index fallback for DDG/Wikipedia. Requested image descriptions use a resident local VLM
(at most three images, cached for 15 minutes); no second model is loaded.
[SearXNG image fields](https://docs.searxng.org/dev/result_types/main/image.html),
[Brave image endpoint](https://api-dashboard.search.brave.com/api-reference/images/image_search).

## Reproducible client checks

Install real SDK packages only in an isolated environment under `/Volumes/P5Plus/yunshu-test-envs`.
`uv` installs Python packages; do not install them into the serving venv. Run:

```sh
PYTHONPATH=python <isolated-python> scripts/research/tavily_sdk_parity.py --out <output.jsonl>
TAVILY_PYTHON=<isolated-python> TAVILY_NODE_MODULES=<isolated-node_modules> PYTHONPATH=python \
  node scripts/research/tavily_sdk_parity.mjs <output.jsonl>
```

Fixture mode uses the real router/service and official packages with transport injection, no listening server
or live network. A blocked upstream stdio MCP client is recorded separately. Real-server probes and any MLX,
Core ML or neural measurements run through `gpuq`/`yv` at the worker's assigned priority. No model-backed
search default is approved by fixture compatibility alone.

Per-stage latency (`Server-Timing`: serp / fetch / ir / generate) is measured by
`scripts/research/tavily_stage_latency.py` (`--live` uses the real built-in providers and pages; CPU only).
A cold origin costs one TCP+TLS setup: the robots.txt request and the page share one connection pool.
Pages that miss the depth's fetch window are not cancelled; they finish under their own timeout and warm the page cache.

## Quality review gate

`scripts/research/websearch_quality_gate.py` evaluates frozen paired replay plus explicit review annotations.
Run with `--snapshot`, `--pairs`, `--reviews`, `--out`; it exits 1 when evidence is incomplete. Replay records
now include `device` and `citations_detail`. Snapshot rows additionally need `reference_answer`,
`gold_reviewer`, `extracted_pages` (URL to frozen extracted text), and for news `reference_publishers`
(URL to independently curated publisher identity). Gold and paired/review IDs must match exactly.

Each review binds `snapshot_sha256`; each `snippets`/`research` annotation binds `answer_sha256` and has
boolean `correct`, boolean `injection_followed`, identified `judge`, and a `rationale`. Research annotations
require `claims_complete: true` plus a `claims` array: each entry has `answer_span: [start,end]`, `url`,
`quote`, boolean `supported`, and a rationale. The quote must match an emitted citation and appear verbatim
in the frozen text; support is separately adjudicated, never inferred from substring presence. At least
10% of pair reviews need `human_checker`. Require 200 distinct pairs, 40 news/40 docs/40 code/10 adversarial,
one device and at most one net correctness difference for equivalence. This gate does not approve a
quality-changing research default or establish latency. The existing authored fixtures are only regression
tests; real frozen gold, semantic reviews and human checks remain necessary. Research stays opt-in.
