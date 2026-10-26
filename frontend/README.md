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
No missing metric is invented. Reusable YunUI time-series, bar, donut and heatmap
components provide synchronized inspection and host-owned drilldowns. Model operations and request cancellation follow
the server's existing authorization and return errors without optimistic success. Import/alias/delete, idle retention,
VLM image and JSON/thinking controls, diagnostics, and live OpenAPI discovery use
the backend's existing endpoints.

See [the console guide](../docs/CONSOLE.md) for metric definitions and packaging,
and [the pinned YunUI package](vendor/README.md) for provenance. Keep using public
YunUI components when extending this application; do not recreate its controls.

Current agent validation is CPU/fixture-only: do not use GPU inference or restart
the test engine without renewed explicit owner instruction.

## Markdown dependencies

`package.json` lists `react-markdown`, `remark-gfm`, `remark-math`, `rehype-raw`, `rehype-katex` and
`rehype-slug` although no file in `src/` imports them. They are peer dependencies (marked optional) of the pinned
`@yuhuanowo/yunui`, and its `dist/content.js` imports `react-markdown`, `remark-math`, `rehype-raw` and `rehype-katex`
at module top level. The console loads that module (`src/Playground.tsx` imports `@yuhuanowo/yunui/content`, and
`main.tsx` imports `content.css`) to render assistant Markdown, including math and raw HTML (sanitized by the
package). Removing any of them breaks the Playground chunk's build, so keep them; recheck when the pinned
YunUI package changes.
