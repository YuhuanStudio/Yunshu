# Console documentation platform

Docs remain at `/console/#/docs` on the console port. Fumadocs MDX's Vite plugin compiles the async collection in `src/docs/source.ts`; its source loader builds the locale-specific sidebar from `docs/**/meta.json`. The console's existing hash router is adapted through `FrameworkProvider`, with the console owning theme and command palette.

Install with `pnpm install`, then `pnpm build`. The build exports the real FastAPI schema without app lifespan or MLX operations. Install the repository's Python dependencies with `uv` first, or set `DOCS_PYTHON` to an existing Python environment containing those dependencies. No engine needs to run. Do not replace the generated schema with a hand-written route list.

`src/docs/mdx-components.tsx` exposes Fumadocs components alongside the existing `Endpoint`, `Support`, `SSE`, and `NotServed` names. Existing `Tabs/Tab`, `Callout`, `Cards/Card`, `Steps/Step`, and `Accordions/Accordion` authoring remains supported. It also exposes `TypeTable`, `Files/File/Folder`, `ImageZoom`, `Banner`, and `Mermaid` (chart string prop or string children); fenced `mermaid` code renders the same lazy component. Fenced code uses Fumadocs/Shiki metadata for titles, highlighted lines and copy; use Fumadocs `CodeBlockTabs` syntax for grouped code examples.

Search indexes are built once per locale with Orama and loaded only when the console command palette searches. Traditional and Simplified Chinese use character/bigram tokens. Markdown downloads and `llms[.locale].txt` / `llms-full[.locale].txt` are emitted under `/console/`; they retain source MDX and include the generated OpenAPI schema. Last-updated timestamps come from each source file's git history.

The generated OpenAPI reference starts with GET `/v1/models`. Its playground uses the browser's origin, so the console proxy handles requests. Bearer and `x-api-key` schemes are exposed as optional auth inputs; users provide their own keys. SDK examples live beside the generated operations. Endpoint components link to the matching operation and reference pages link back to authored API guides.

Validation: `pnpm exec playwright test browser/docs.spec.ts browser/docs-platform.spec.ts --project chromium`, `node --test tests/docs*.test.ts`, `pytest frontend/scripts/test_dump_openapi.py tests/unit/test_docs_routes.py`, and the existing bundle/i18n guards. Screenshots are written to `frontend/evidence/docsplat/` (ignored).
