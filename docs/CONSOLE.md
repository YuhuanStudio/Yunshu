# Yunshu console

A local web console for the engine: see what every request is doing, manage models, the prefix cache, keys
and settings, and try a model, without leaving the browser. It is a YunUI-based single-page app served by
its own light process (`yunshu console`, see [The console process](#the-console-process)), so it stays usable
and keeps recording while the engine restarts or crashes. It is an operator tool for one machine, not a chat
product, and it does not start or stop the engine process.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/overview-dark.webp">
  <img src="images/console/en/overview-light.webp" alt="Engine overview in the Yunshu console">
</picture>

## Open it

```bash
yunshu serve -m <model>        # the engine on :8000, the console process next to it on :8100
open http://127.0.0.1:8100/console/
```

The console ships inside the `yunshu` wheel (`yunshu_gateway/console_static/`), so a pip, uv tool or
Homebrew install serves it with no build step. `http://127.0.0.1:8000/console/` (the engine) still works for
one release as a redirect to the console process, and says how to start it when nothing answers there.

The static shell is public so you can enter a token; inference, model operations, keys and settings keep
their normal authorization ([Authentication](guides/AUTH_AND_KEYS.md)). If `YUNSHU_AUTH_TOKEN` is set, enter
it in Settings: it stays in page memory unless you choose to remember it on this device.

## The console process

`yunshu console` is a separate, light process. It never imports MLX, mlx-lm or mlx-vlm (a unit test asserts
that in a fresh interpreter) and idles at about 70 MiB of resident memory, so it starts instantly and an
engine crash cannot reach it. It does four things:

- **Serves** the web console and the docs (`/console/`).
- **Proxies** the engine API on one origin: every request the console makes goes to this process, which
  forwards it unchanged (methods, bodies, `Authorization` and `x-api-key`, SSE streams, WebSockets). When the
  engine is not reachable it answers `502` with `engine_unreachable` at once instead of hanging.
- **Records** the history. Once a second it reads the engine's cheap endpoints (`/v1/yunshu/status`, finished
  requests by cursor `/v1/yunshu/requests/recent?after_seq=`, and every five seconds the host telemetry), the
  same reads the console page makes, and writes them to `~/.yunshu/console-history.sqlite` (WAL, batched every
  ten seconds): 1 s resolution for the last hour, 10 s for 24 hours, 1 minute for 30 days
  (`YUNSHU_CONSOLE_RETENTION_DAYS`), plus the request log (id, model, route, status, timestamps, token counts,
  cached tokens, TTFT, decode speed, finish reason; metadata only, never prompts or outputs). The file is
  capped (`YUNSHU_CONSOLE_DB_MAX_MB`, 64 MiB) and keeps the newest rows. The request cursor is
  `(boot_id, seq)`: none is skipped or repeated across polls, an engine restart or a console restart.
- **Remembers outages.** While the engine does not answer, no rows are written: that span *is* the gap, and
  it is reported as a gap, never interpolated. Events say why: engine unreachable / reachable again (and for
  how long), engine restarted, a model loaded or unloaded, a model that failed to load (with its message).

The history is served by the console process itself, so it works while the engine is down:
`GET /v1/yunshu/metrics/history?since=&until=&step=` (columnar rows, `gaps` and `events`; the finest table that
covers `since` and is no finer than `step`), `GET /v1/yunshu/requests/history?limit=&before=` (newest first, a
cursor) and `GET /v1/yunshu/console` (the process's own view of the engine). Access follows the engine's rule:
open on a default local setup, otherwise the token or an API key (the console process shares the engine's
settings and key file).

**Ports.** The engine stays on 8000. The console defaults to **8100**: next to the engine's number but clear
of it and of the usual dev servers (3000, 5173, 8080, 8888) and the 18990-18999 range agents use. Change it with
`--console-port` or `YUNSHU_CONSOLE_PORT`.

**Starting it.**

| How | What runs |
|---|---|
| `yunshu serve -m <model>` | the engine, and the console process as a sibling (`--no-console` to skip, `--console-port` to move it). If a console already answers on that port, such as the service's, none is started. The sibling exits with the engine. |
| `yunshu console --engine URL` | the console process alone: watch an engine on another machine or on the LAN (`--engine-token` when it needs one; `--host 0.0.0.0` to serve the network, with a token set). |
| `yunshu service install` | two launchd jobs: the engine, and the console as its own job (always kept alive), so recording continues while the engine restarts. `--no-console` installs the engine only; `yunshu service logs --console` shows its log. |

**When the engine is down** the console says so ("engine offline since 18:02:11, retrying in 4 s"; retries back
off with jitter up to about ten seconds and run at once when the tab regains focus or the network returns),
every monitoring page keeps its last data, dimmed and marked stale, and the docs and settings are unaffected.
If the gateway is up but the model failed to load, the page shows the error and keeps Load model usable. A
page that throws is contained by a per-page error boundary. When the engine returns, the live feed resumes and
the stretch that was missed is filled from the recorded history.

## Docs inside the console

The 文件 / Docs section renders the user documentation (getting started, API reference, guides,
developer pages) from `frontend/docs/` (MDX in English, 繁體中文 and 简体中文), compiled into the
console build, so it works offline and matches the installed version. `⌘K` searches it; the API,
Settings and Keys pages link to the page that explains them. These MDX pages are the source of
the user-facing guides; the same-named files in `docs/guides/` are their GitHub-facing counterparts
(each names its MDX page), and the configuration reference is generated from the settings registry
(`frontend/scripts/gen_docs_config.py`). Checks: `pnpm test` (links and heading anchors),
`tests/unit/test_docs_routes.py` (every route named in the docs is registered; generated pages are current).

## Status island

A floating pill at the bottom of every page shows the live engine state: running model, memory, prefill or
decode progress and active requests. Opening it shows the current phase of each request, Metal memory and
pressure, and host power, GPU clock and temperature without leaving the page you are on. On a phone it
collapses to a single compact pill.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/island-dark.webp">
  <img src="images/console/en/island-light.webp" alt="Status island opened over the overview">
</picture>

## Pages

### Engine overview

One screen for how the Mac is handling inference right now: a health banner (memory, queue, error rate), live **request phases** (queued, prefill, decode) with a prefill progress bar whose **cache-hit segment** shows how many prompt tokens came from the prefix cache, decode and prefill tok/s, time to first token, prefix hit rate, Metal memory, and throughput charts. A host panel adds power, GPU clock and die temperature when host telemetry is available ([Telemetry](guides/TELEMETRY.md)). It samples every few seconds while the page is visible; Pause and Refresh are in the header.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/overview-dark.webp">
  <img src="images/console/en/overview-light.webp" alt="Engine overview page">
</picture>

### Requests and performance

Live and recently finished requests: latency distribution (time to first token or queue time, cold versus warm by cache hit), P50/P90 per group, speculative-decode acceptance by draft depth, per-request detail and cancel, request history read from the metadata-only serve log (no prompts), and CSV export. Failed requests are not counted in first-token latency.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/requests-dark.webp">
  <img src="images/console/en/requests-light.webp" alt="Requests and performance page">
</picture>

### Logs

The engine's in-memory log ring (2,000 records, credentials already redacted when emitted): level filter, text search, follow/pause, and copy or download of the visible lines. The buffer is cleared on restart.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/logs-dark.webp">
  <img src="images/console/en/logs-light.webp" alt="Logs page">
</picture>

### Engine diagnostics

Resource readouts (CPU, unified memory, Metal), health checks derived from real responses, a Realtime connection probe (browser to engine WebSocket, no audio, no GPU), a preview of the support bundle, and copy or download of the diagnostics bundle (`GET /v1/yunshu/bundle`; nothing is uploaded and it never contains prompts).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/diagnostics-dark.webp">
  <img src="images/console/en/diagnostics-light.webp" alt="Engine diagnostics page">
</picture>

### Models

Registered and loaded models with state, size and idle retention; load, unload, warm up and test in the playground; a fit check that says whether a model fits and what would be evicted; import a local or Hugging Face snapshot without copying weights; models found on disk but not registered; cancel an in-progress load.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/models-dark.webp">
  <img src="images/console/en/models-light.webp" alt="Models page">
</picture>

### Downloads

Pull a Hugging Face repository (optional revision and file patterns) with progress, rate, ETA, cancel and resume, and free-space display. Finished downloads are registered and appear under Models.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/downloads-dark.webp">
  <img src="images/console/en/downloads-light.webp" alt="Downloads page">
</picture>

### Cache

Prefix cache (APC) per model: usage and cap of the RAM, WARM and SSD tiers, entries, hit rate by tier, the largest entries, and clear by tier. Clear is refused while a request is running.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/cache-dark.webp">
  <img src="images/console/en/cache-light.webp" alt="Cache page">
</picture>

### Playground

Chat against the loaded model over real SSE streaming with reasoning shown separately, side-by-side compare, saved presets, image input for vision models, a tool-call tester, per-token confidence (logprobs) and a view-code panel that emits the same request as curl or SDK code.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/playground-dark.webp">
  <img src="images/console/en/playground-light.webp" alt="Playground page">
</picture>

### API access

Base URLs for the OpenAI and Anthropic dialects and ready-to-paste setup for Claude Code, Codex, opencode, the SDKs and curl, generated for the selected model. The same wiring is available from the terminal with `yunshu launch`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/api-dark.webp">
  <img src="images/console/en/api-light.webp" alt="API access page">
</picture>

### API keys

Create, edit, rotate, disable and delete API keys with scopes, expiry and daily quotas (requests, tokens, concurrency), plus per-key usage over 7, 14 or 30 days. The secret is shown once. See [Authentication and keys](guides/AUTH_AND_KEYS.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/keys-dark.webp">
  <img src="images/console/en/keys-light.webp" alt="API keys page">
</picture>

### Settings

Engine connection and token, appearance and language (English, Traditional and Simplified Chinese), model idle retention, **effective settings** (every registered `YUNSHU_*` setting with its value, source and when a change applies; edits that need a restart prompt, and a restart happens only under launchd), the launchd service state, network and a **CORS editor**, and keyboard shortcuts.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/console/en/settings-dark.webp">
  <img src="images/console/en/settings-light.webp" alt="Settings page">
</picture>

## On a phone

The layout reflows below tablet width: the navigation becomes a drawer, cards stack and the status island
becomes a compact pill. Checked in WebKit at 402 px wide.

<p>
  <img src="images/console/en/mobile-overview-dark.webp" alt="Overview on a phone" width="260">
  <img src="images/console/en/mobile-requests-dark.webp" alt="Requests on a phone" width="260">
</p>

## What it connects to

Everything the console shows comes from the engine's own routes: `GET /v1/yunshu/status` (every few seconds),
`/v1/yunshu/requests/recent` and `/history`, `/v1/yunshu/host`, `/v1/yunshu/memory`, `/v1/yunshu/cache`,
`/v1/yunshu/logs`, `/v1/yunshu/keys`, `/v1/yunshu/config`, `/v1/yunshu/cors`, `/v1/yunshu/service`, the model
load, unload, warmup and download routes, and the authenticated `/debug/*` diagnostics. See the
[API surface](guides/API_SURFACE.md#single-operator-console-backend). Missing data is shown as unavailable,
never as zero.

## Developing the console

The source is in `frontend/` (Node.js 22.18+, pnpm 11.19.0). A source checkout needs `pnpm install --frozen-lockfile && pnpm build` once to write `console_static/` (a built wheel already contains it); without it `/console/` returns an actionable 404 and the API is unaffected. With the engine and its console process running, run `pnpm dev` in `frontend/` and open `http://127.0.0.1:3971/console/`. Vite proxies the
API and the history endpoints to the console process on `http://127.0.0.1:8100` (`YUNSHU_CONSOLE_DEV_PROXY` points it elsewhere), or set another server URL in Settings (cross-origin access needs CORS
configured, see [Authentication](guides/AUTH_AND_KEYS.md#cors)).

## Metric meanings

The graph's time selector filters observations collected since opening the page;
no historical values are invented. Throughput plots sample the backend's rolling
mean prefill/decode speed (currently five minutes), not an aggregate of overlapping
request-count windows. The request-count KPI uses `throughput.window_s` (currently
60 seconds). Prefix reuse and TTFT refer to the latest terminal request, not an
all-time mean. Metal active/cache/peak are allocator metrics, not OS free memory or
prefix-cache hit counters. Missing values remain unavailable rather than zero.

## Reproducing the screenshots

The images in `docs/images/console/` are rendered from the real console against route mocks (no engine and
no GPU), so the numbers in them are illustrative fixtures, not measurements:
`node scripts/docs/console_shots.mjs <vite-url> <out-dir>` (see the script header).

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

Build the console before building a wheel; the release workflow checks that `index.html` is in it. Hatch includes generated `console_static`
assets when present; source distributions include the frontend and pinned YunUI
package under `frontend/vendor/`. That package contains the approved YunUI code
without requiring a sibling checkout or an npm release; provenance and checksum are
recorded in `frontend/vendor/README.md`. No release version was changed.

Browser/API fixture checks validate the UI without model inference. Model-backed
acceptance checks on a shared machine belong in `gpuq` / `yv`; see
[verification](guides/VERIFY.md).
