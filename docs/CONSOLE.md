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
`http://127.0.0.1:3971/console/`. Vite proxies `/v1` to `http://127.0.0.1:8000`.
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
  be unloaded here. Model metadata comes from `GET /v1/models/{id}`. Registering or
  downloading new models stays in the engine's existing configuration/CLI workflow.
- Requests: live rows and deduplicated terminal records actually observed from
  `status.last`, filtering, CSV export, detail and per-request cancellation.
  The backend supplies only the latest terminal record: this is explicitly an
  incomplete observation log, not a durable/full request history. A 200 stream may
  finish early, so terminal rows are labeled ended, not certified successful.
- Playground: actual OpenAI-compatible SSE streaming, content/reasoning display,
  stop, generation settings and reset. Navigating away aborts the stream. Partial
  failed/stopped assistant messages are not sent as complete responses in subsequent
  context. This is a text diagnostic surface; file/media/tool-execution controls
  are not exposed until they have working host integration.
- Connection preferences: only service URL and theme are persisted in localStorage.
  The bearer token is held in memory and cleared by reload; changing the URL clears
  the token input. No token is included in snippets, logs, exports or URLs.

Model control endpoints require the existing privileged-operation authorization.
Enter the configured `YUNSHU_AUTH_TOKEN` in Settings if required. The frontend does
not disable authentication or change the engine's settings.

## Metric meanings

The graph's time selector filters observations collected since opening the page;
no historical values are invented. Throughput plots sample the backend's rolling
mean prefill/decode speed (currently five minutes), not an aggregate of overlapping
request-count windows. The request-count KPI uses `throughput.window_s` (currently
60 seconds). Prefix reuse and TTFT refer to the latest terminal request, not an
all-time mean. Metal active/cache/peak are allocator metrics, not OS free memory or
prefix-cache hit counters. Missing values remain unavailable rather than zero.

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
