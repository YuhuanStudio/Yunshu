import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import {
  listFiles,
  pickFile,
  parseFrontmatter,
  LOCALES,
} from "./docs-index.mjs";

/** Build and dev serve the same static artifacts; no runtime Python/MLX in docs. */
export function docsAssets() {
  let assets;
  const collect = () => {
    if (assets) return assets;
    const root = new URL("../../", import.meta.url).pathname;
    if (process.env.DOCS_PYTHON)
      execFileSync(
        process.env.DOCS_PYTHON,
        ["frontend/scripts/dump_openapi.py"],
        { cwd: root },
      );
    else
      execFileSync(
        "uv",
        ["run", "--no-sync", "python", "frontend/scripts/dump_openapi.py"],
        { cwd: root },
      );
    assets = new Map([
      [
        "docs-assets/openapi.json",
        readFileSync(
          new URL("../generated/openapi.json", import.meta.url),
          "utf8",
        ),
      ],
    ]);
    const files = listFiles();
    for (const locale of LOCALES) {
      const entries = [];
      for (const slug of new Set(files.map((f) => f.slug))) {
        const file = pickFile(files, slug, locale);
        const text = readFileSync(file.file, "utf8");
        const { meta } = parseFrontmatter(text);
        const path = `docs-assets/${locale}/${slug}.md`;
        assets.set(path, text);
        entries.push({
          slug,
          title: meta.title,
          description: meta.description,
          text,
          path,
        });
      }
      const index =
        `# Yunshu\n\nLocal LLM / VLM inference engine for Apple Silicon.\n\n` +
        entries
          .map(
            (e) => `- [${e.title}](/console/${e.path}): ${e.description ?? ""}`,
          )
          .join("\n") +
        "\n\n## Generated API reference\n\n- [OpenAPI schema](/console/docs-assets/openapi.json)\n";
      const full =
        entries
          .map((e) => `<!-- ${e.slug} -->\n${e.text}`)
          .join("\n\n---\n\n") +
        "\n\n## Generated OpenAPI\n\n```json\n" +
        assets.get("docs-assets/openapi.json") +
        "\n```\n";
      assets.set(`llms.${locale}.txt`, index);
      assets.set(`llms-full.${locale}.txt`, full);
      if (locale === "en") {
        assets.set("llms.txt", index);
        assets.set("llms-full.txt", full);
      }
    }
    return assets;
  };
  return {
    name: "yunshu-docs-assets",
    buildStart() {
      collect();
    },
    generateBundle() {
      for (const [fileName, source] of collect())
        this.emitFile({ type: "asset", fileName, source });
    },
    configureServer(server) {
      server.watcher.on("change", (file) => {
        if (file.includes("/docs/")) assets = undefined;
      });
      server.middlewares.use((req, res, next) => {
        const path = decodeURI((req.url ?? "").split("?")[0])
          .replace(/^\/console\//, "")
          .replace(/^\//, "");
        if (!/^(docs-assets\/|llms)/.test(path)) return next();
        const value = collect().get(path);
        if (value === undefined) return next();
        res.setHeader(
          "Content-Type",
          path.endsWith(".json")
            ? "application/json"
            : "text/plain; charset=utf-8",
        );
        res.end(value);
      });
    },
  };
}
