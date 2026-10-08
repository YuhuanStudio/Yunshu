import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createApp } from "./app.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const siteRoot = path.resolve(here, "..");

function mainCheckout(): string {
  if (process.env.RS_REPO) return process.env.RS_REPO;
  const common = execFileSync("git", ["-C", siteRoot, "rev-parse", "--path-format=absolute", "--git-common-dir"], { encoding: "utf8" }).trim();
  return path.dirname(common);
}

const repo = mainCheckout();
const app = createApp({
  roots: {
    research: path.join(repo, "docs", "research"),
    jobs: process.env.RS_JOBS ?? "/Volumes/P5Plus/yunshu-gpuq/jobs",
    codex: process.env.RS_CODEX ?? "/Volumes/P5Plus/yunshu-build/codex",
  },
  indexScript: path.resolve(siteRoot, "..", "..", "scripts", "dev", "research_index.py"),
  staticDir: path.join(siteRoot, "dist"),
});
app.start();
const port = Number(process.env.RS_PORT ?? 3990);
app.server.listen(port, process.env.RS_HOST ?? "0.0.0.0", () => console.log(`research-site on http://0.0.0.0:${port} (read-only; research=${repo}/docs/research)`));
