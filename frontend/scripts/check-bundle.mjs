// Size budget: build the console to a temp dir and fail when a route's first load (gzip) or the
// shared entry grows past the guard in bundle-size.mjs. Part of `pnpm test`.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { report } from "./bundle-size.mjs";

const root = new URL("..", import.meta.url).pathname;
const out = mkdtempSync(join(tmpdir(), "yunshu-console-bundle-"));
try {
  execFileSync("npx", ["vite", "build", "--outDir", out, "--emptyOutDir", "--logLevel", "error"], { cwd: root, stdio: ["ignore", "ignore", "inherit"] });
  const { lines, problems } = report(out);
  console.log(lines.join("\n"));
  if (problems.length) {
    console.error(`check-bundle: ${problems.join("; ")}`);
    process.exitCode = 1;
  } else console.log("check-bundle: ok");
} finally {
  rmSync(out, { recursive: true, force: true });
}
