import { create, insertMultiple, save } from "@orama/orama";
import { tokenizeDocs } from "./src/docs/orama-tokenizer";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fumadocsMdx } from "fumadocs-mdx/vite";
import { remarkMermaid } from "./scripts/remark-mermaid.mjs";
import { docsAssets } from "./scripts/docs-assets.mjs";
import { buildSearch, buildToc } from "./scripts/docs-index.mjs";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// The 文件 section: pages are MDX compiled at build time (one lazy chunk each); the table of
// contents and the search text come from scripts/docs-index.mjs as per-locale virtual modules,
// loaded only when the docs or the command palette ask for them.
const VIRTUAL = /^virtual:docs-(toc|search)-(en|zh-TW|zh-CN)$/;
const docsIndex = () => ({
  name: "yunshu-docs-index",
  resolveId(id: string) {
    return VIRTUAL.test(id) ? "\0" + id : undefined;
  },
  async load(id: string) {
    const m = VIRTUAL.exec(id.replace("\0", ""));
    if (!m) return undefined;
    let data: unknown;
    if (m[1] === "toc") data = buildToc(m[2]);
    else {
      const entries = buildSearch(m[2]);
      const db = create({
        schema: {
          slug: "string",
          title: "string",
          description: "string",
          text: "string",
          headings: "string",
        },
        components: {
          tokenizer: { language: "english", tokenize: tokenizeDocs },
        },
      });
      await insertMultiple(
        db,
        entries.map((e) => ({
          ...e,
          headings: e.headings.map((h) => h.text).join(" "),
        })),
      );
      data = { data: save(db), entries };
    }
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

// The dev/preview server binds where the console process would: YUNSHU_CONSOLE_HOST from the environment,
// else from the Yunshu config file (YUNSHU_CONFIG or ~/.yunshu/config.toml), else this Mac only.
function consoleHost(): string {
  if (process.env.YUNSHU_CONSOLE_HOST) return process.env.YUNSHU_CONSOLE_HOST;
  try {
    const file =
      process.env.YUNSHU_CONFIG ?? join(homedir(), ".yunshu", "config.toml");
    const m = /^\s*YUNSHU_CONSOLE_HOST\s*=\s*"([^"]+)"/m.exec(
      readFileSync(file, "utf8"),
    );
    if (m) return m[1];
  } catch {
    // no config file: the default below
  }
  return "127.0.0.1";
}
const host = consoleHost();
export default defineConfig({
  plugins: [
    docsIndex(),
    docsAssets(),
    ...fumadocsMdx({
      index: false,
      globalOptions: { mdxOptions: { remarkPlugins: [remarkMermaid] } },
    }),
    react({ include: /\.(mdx|tsx|ts|jsx|js)$/ }),
  ],
  base: "/console/",
  resolve: { dedupe: ["react", "react-dom"] },
  build: {
    outDir: "../python/yunshu_gateway/console_static",
    emptyOutDir: true,
    // Let native module imports fetch JS dependencies on demand. Keep CSS
    // preloads, without embedding every lazy docs/vendor JS URL in the shell.
    modulePreload: {
      polyfill: false,
      resolveDependencies: (_filename, deps) =>
        deps.filter((d) => d.endsWith(".css")),
    },
    rolldownOptions: {
      output: {
        chunkFileNames: (chunk) =>
          chunk.name.startsWith("vendor")
            ? "assets/v-[hash].js"
            : "assets/[name]-[hash].js",
        codeSplitting: {
          includeDependenciesRecursively: false,
          groups: [
            {
              name: "react-client",
              test: /node_modules\/(?:react\/(?:index\.js|jsx-runtime\.js|cjs\/react(?:-jsx-runtime)?\.production\.js)|react-dom\/(?:client\.js|index\.js|cjs\/react-dom(?:-client)?\.production\.js)|scheduler\/(?:index\.js|cjs\/scheduler\.production\.js))$/,
              priority: 30,
            },
            { name: "vendor", test: /node_modules/, entriesAware: true },
          ],
        },
      },
    },
    license: { fileName: "licenses/bundled-dependencies.md" },
  },
  preview: { host, port: 3971, strictPort: true },
  server: {
    host,
    port: 3971,
    strictPort: true,
    proxy: {
      "/console/api": engine,
      "/v1": engine,
      "/health": engine,
      "/api": engine,
      "/debug": engine,
      "/openapi.json": engine,
      "/docs": engine,
    },
  },
});
