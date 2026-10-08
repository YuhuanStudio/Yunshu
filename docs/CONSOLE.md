# Yunshu Console

Yunshu's local inference-engine console lives in `frontend/`. It imports YunUI
components directly for navigation, controls, charts, model tables, request details,
Markdown, code blocks and the diagnostic playground. It does not include the retired
frontend or turn Yunshu into a chat product.

## Build and open

Requires Node.js 22.18+ and pnpm 11.19.0. From the repository root:

```sh
cd frontend
pnpm install --frozen-lockfile
pnpm build
```

The build writes `python/yunshu_gateway/console_static/`. Start Yunshu with your
normal model/configuration, then open `http://127.0.0.1:8000/console/` (use your
actual server port). The existing API and console share one origin. The static
shell is public so the user can enter a token; inference and model operations
retain their existing authorization. Without a frontend build, `/console/` returns
an actionable 404; API startup does not depend on Node.js or the frontend.

For frontend development, run `pnpm dev` from `frontend/` and open
`http://127.0.0.1:3971/console/`. Vite proxies the engine API, diagnostic and OpenAPI routes to `http://127.0.0.1:8000`.
Alternatively set an explicit server URL in Settings; cross-origin access then
requires that server's normal CORS configuration. The console does not start or
stop the engine process.

## What the console connects

- Engine status: `GET /v1/yunshu/status`, sampled every three seconds while visible.
  Pause/resume and manual refresh are available. Failed refreshes preserve the last
  known data with a disconnected/stale indicator. Changing connection or observing
  a server restart clears history. Sampling is bounded to 1,200 observations and
  exists only in page memory; it is not a durable history database.
- Models: real load, unload and warmup endpoints, operation feedback, errors and
  current state. Unload asks for confirmation. Fixed single-model instances cannot
  be unloaded here. Model metadata comes from `GET /v1/models/{id}`. Native MLX
  import, persistent aliases and deletion use `/api/pull`, `/api/copy`, and
  `/api/delete`; deletion requires typing the complete model ID. Import reports
  waiting for the backend, since this API has no progress or cancellation stream.
  Per-model idle retention uses the existing warmup `keep_alive` contract; the UI
  does not invent an unpin or global-settings mutation endpoint.
- Requests: live rows and deduplicated terminal records actually observed from
  `status.last`, filtering, CSV export, detail and per-request cancellation.
  The backend supplies only the latest terminal record: this is explicitly an
  incomplete observation log, not a durable/full request history. Active request
  details poll `GET /v1/requests/{id}` until the sheet closes or the request returns
  404; the last successful snapshot is then explicitly marked as stale. A 200 stream may
  finish early, so terminal rows are labeled ended, not certified successful.
- Playground: actual OpenAI-compatible SSE streaming, content/reasoning display,
  stop, generation settings and reset. VLM image input sends OpenAI image parts;
  text/JSON output and auto/on/off thinking modes map to the existing request
  parameters. Length termination is visibly distinguished from a full answer.
  Navigating away aborts the stream. Partial
  failed/stopped assistant messages are not sent as complete responses in subsequent
  context. PNG/JPEG/WebP input is bounded to 8 MB and never persisted. Speech,
  video generation and tool execution are not silently mapped to chat; their
  actual API contracts remain discoverable in the service's OpenAPI catalogue.
- Diagnostics: authenticated `/debug/system`, `/debug/engine`, `/debug/requests`,
  `/debug/kv-cache`, `/debug/ssd-cache`, `/debug/spec-decode`, `/debug/per-model`,
  `/debug/memory-guard` and `/debug/memory-census`. Missing or disabled diagnostics
  are explicit, and endpoint-specific raw data can be inspected. API discovery
  reads `/openapi.json` from the selected service instead of a hardcoded route list.
- Connection preferences: only service URL and theme are persisted in localStorage.
  Authentication failures pause polling until a manual retry or connection change.
  The bearer token is held in memory and cleared by reload; changing the URL clears
  the token input. No token is included in snippets, logs, exports or URLs.

Model control endpoints require the existing privileged-operation authorization.
Enter the configured `YUNSHU_AUTH_TOKEN` in Settings if required. The frontend does
not disable authentication or rewrite global startup configuration.

## Metric meanings

The graph's time selector filters observations collected since opening the page;
no historical values are invented. Throughput plots sample the backend's rolling
mean prefill/decode speed (currently five minutes), not an aggregate of overlapping
request-count windows. The request-count KPI uses `throughput.window_s` (currently
60 seconds). Prefix reuse and TTFT refer to the latest terminal request, not an
all-time mean. Metal active/cache/peak are allocator metrics, not OS free memory or
prefix-cache hit counters. Missing values remain unavailable rather than zero.

## Reusable analytics

The console imports `TimeSeriesChart`, `BarChart`, `DonutChart` and `Heatmap` from
YunUI. Numeric time spacing, missing-value gaps, series visibility and keyboard
inspection belong to the library; aggregation and backend access stay here.
Three time charts share a cursor. A heatmap cell selects its actual observed peak;
an unobserved cell clears the selection. Histogram bins open matching request
records, and observations can be exported for the chosen window.

Latency records are deduplicated by request ID before binning/percentiles. The
heatmap uses per-bucket observed concurrency peaks, not estimated traffic. Null
means no observation; zero means an observed idle state. Sampling gaps over twelve
seconds break the time-series lines. Chart geometry is memoized independently of
cursor movement.

## Validation and packaging

```sh
cd frontend
pnpm test
pnpm test:browser  # first run: pnpm exec playwright install chromium webkit
pnpm build
pnpm exec yunui doctor --strict
pnpm exec yunui audit --strict
cd ..
uv run pytest tests/unit/test_console_ui.py -q
```

Build the console before building a wheel. Hatch includes generated `console_static`
assets when present; source distributions include the frontend and pinned YunUI
package under `frontend/vendor/`. That package contains the approved YunUI code
without requiring a sibling checkout or an npm release; provenance and checksum are
recorded in `frontend/vendor/README.md`. No release version was changed.

Browser/API fixture checks validate the UI without model inference. Model-backed
acceptance checks on a shared machine belong in `gpuq` / `yv`; see
[verification](guides/VERIFY.md).

### 2026-10-07 local validation

The analytics/management iteration passed 25 Node unit tests and all 18 browser
contracts across Chromium and WebKit (intercepted API responses only). The real
render matrix covered 390, 768, 1024, 1440 and 1920 px in light/dark: 200 captures,
with no runtime errors or document overflow; contact sheets and representative
full-size images were visually inspected. Production build, TypeScript, and both
strict YunUI integration/adoption checks passed. Evidence remains in the ignored
`frontend/evidence/` directory of the working checkout. No engine or model was
started for these checks. This does not replace real-model acceptance testing.
