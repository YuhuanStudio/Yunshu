# Yunshu Console

A YunUI consumer for the local inference engine. It uses real API responses and
contains no application demo-data mode. Mock responses exist only in tests.

```sh
pnpm install --frozen-lockfile
pnpm dev                 # http://127.0.0.1:3971/console/; /v1 proxies to :8000
pnpm test                # API contracts and incremental SSE parsing
pnpm test:browser        # Chromium/WebKit, controlled API responses; no GPU
pnpm build               # writes ../python/yunshu_gateway/console_static
```

First browser-test run: `pnpm exec playwright install chromium webkit`.
The built app is served by Yunshu at `/console/`. Enter an optional service token
in Settings; tokens stay in memory. The console does not launch the engine.

Charts show observations collected since page load, not historical fixtures.
No missing metric is invented. Model operations and request cancellation follow
the server's existing authorization and return errors without optimistic success.

See [the console guide](../docs/CONSOLE.md) for metric definitions and packaging,
and [the pinned YunUI package](vendor/README.md) for provenance. Keep using public
YunUI components when extending this application; do not recreate its controls.
