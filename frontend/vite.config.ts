import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import mdx from "@mdx-js/rollup";
import remarkGfm from "remark-gfm";
import remarkFrontmatter from "remark-frontmatter";
import rehypeSlug from "rehype-slug";
import { buildSearch, buildToc } from "./scripts/docs-index.mjs";

// The 文件 section: pages are MDX compiled at build time (one lazy chunk each); the table of
// contents and the search text come from scripts/docs-index.mjs as per-locale virtual modules,
// loaded only when the docs or the command palette ask for them.
const VIRTUAL = /^virtual:docs-(toc|search)-(en|zh-TW|zh-CN)$/;
const docsIndex = () => ({
  name: "yunshu-docs-index",
  resolveId(id: string) {
    return VIRTUAL.test(id) ? "\0" + id : undefined;
  },
  load(id: string) {
    const m = VIRTUAL.exec(id.replace("\0", ""));
    if (!m) return undefined;
    const data = m[1] === "toc" ? buildToc(m[2]) : buildSearch(m[2]);
    return `export default ${JSON.stringify(data)};`;
  },
  handleHotUpdate({
    file,
    server,
  }: {
    file: string;
    server: { ws: { send: (m: object) => void } };
  }) {
    if (file.includes("/frontend/docs/"))
      server.ws.send({ type: "full-reload" });
  },
});
// Dev proxy target: the console process (`yunshu console`, which proxies the engine API and serves the
// history), else its default port. VITE_CONSOLE_PROXY=http://127.0.0.1:8000 points it at an engine
// directly (no recorded history then).
const engine = process.env.VITE_CONSOLE_PROXY ?? "http://127.0.0.1:8100";
export default defineConfig({
  plugins: [
    docsIndex(),
    {
      enforce: "pre",
      ...mdx({
        remarkPlugins: [remarkGfm, remarkFrontmatter],
        rehypePlugins: [rehypeSlug],
      }),
    },
    react({ include: /\.(mdx|tsx|ts|jsx|js)$/ }),
  ],
  base: "/console/",
  resolve: { dedupe: ["react", "react-dom"] },
  build: {
    outDir: "../python/yunshu_gateway/console_static",
    emptyOutDir: true,
    rolldownOptions: {
      output: {
        codeSplitting: {
          // Vendor code is grouped by which routes use it, so a library that
          // only some lazy pages need stays out of the entry chunk.
          includeDependenciesRecursively: false,
          groups: [
            { name: "vendor", test: /node_modules/, entriesAware: true },
          ],
        },
      },
    },
    license: { fileName: "licenses/bundled-dependencies.md" },
  },
  server: {
    proxy: {
      "/v1": engine,
      "/health": engine,
      "/api": engine,
      "/debug": engine,
      "/openapi.json": engine,
      "/docs": engine,
    },
  },
});
