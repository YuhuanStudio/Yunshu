// Per-route first-load size: the entry's static closure plus the route chunk's static closure,
// gzip-measured. Usage: node scripts/bundle-size.mjs <distDir> [--check]
// The budget (gzip KB per route) is checked with --check; `pnpm test` runs this on a fresh build.
import { readFileSync, readdirSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { join } from "node:path";

/** The goal for a route's first load (gzip KB); not met yet: the shared shell (React, YunUI, Radix) is ~250 KB. */
export const TARGET_KB = 250;
/** The regression guard `pnpm test` enforces today (gzip KB first load per route, and for the shared entry). */
export const BUDGET_KB = 300;
export const ENTRY_BUDGET_KB = 270;
// Fumadocs and the OpenAPI playground are only loaded inside Docs. The shell
// and all other routes retain their existing budgets.
export const DOCS_BUDGET_KB = Object.freeze({ Docs: 350, ApiReference: 650 });
export const budgetForRoute = (name) => DOCS_BUDGET_KB[name] ?? BUDGET_KB;

/** Static `import ... from "./x.js"` and `import "./x.js"` edges of one chunk (dynamic imports excluded). */
export function staticImports(code) {
  const out = new Set();
  for (const m of code.matchAll(
    /(?:^|[;\n}])\s*(?:import|export)\s*(?:[^'"`()]*?from\s*)?["'`](\.\/[^"'`]+\.js)["'`]/g,
  ))
    out.add(m[1].slice(2));
  return [...out];
}
export function dynamicImports(code) {
  const out = new Set();
  for (const m of code.matchAll(/import\(\s*["'`]\.\/([^"'`]+\.js)["'`]\s*\)/g))
    out.add(m[1]);
  return [...out];
}
export function closure(files, start) {
  const seen = new Set();
  const walk = (f) => {
    if (seen.has(f) || !(f in files)) return;
    seen.add(f);
    for (const d of staticImports(files[f])) walk(d);
  };
  walk(start);
  return seen;
}
export function routeSizes(dist) {
  const dir = join(dist, "assets");
  const files = Object.fromEntries(
    readdirSync(dir)
      .filter((n) => n.endsWith(".js"))
      .map((n) => [n, readFileSync(join(dir, n), "utf8")]),
  );
  const gz = (names) =>
    [...names].reduce((n, f) => n + gzipSync(files[f]).length, 0);
  const entry = Object.keys(files).find((n) => /^index-/.test(n));
  const base = closure(files, entry);
  const routes = {};
  for (const target of dynamicImports(files[entry])) {
    const name = target.replace(/-[A-Za-z0-9_-]{8}\.js$/, "");
    const all = new Set([...base, ...closure(files, target)]);
    routes[name] = gz(all);
    if (name === "Docs")
      for (const child of dynamicImports(files[target])) {
        if (!/^ApiReference-/.test(child)) continue;
        routes.ApiReference = gz(new Set([...all, ...closure(files, child)]));
      }
  }
  return { entry: gz(base), routes };
}
export function report(dist) {
  const { entry, routes } = routeSizes(dist);
  const kb = (n) => (n / 1024).toFixed(1);
  const lines = [
    `entry ${kb(entry)} KB gzip (budget ${ENTRY_BUDGET_KB}, target ${TARGET_KB})`,
  ];
  const problems = [];
  if (entry / 1024 > ENTRY_BUDGET_KB)
    problems.push(`entry ${kb(entry)} KB > ${ENTRY_BUDGET_KB} KB`);
  for (const [r, n] of Object.entries(routes).sort((a, b) => b[1] - a[1])) {
    const budget = budgetForRoute(r);
    const mark =
      n / 1024 > budget
        ? "OVER "
        : n / 1024 > TARGET_KB
          ? "above target "
          : "ok ";
    lines.push(
      `${mark}${r.padEnd(14)} ${kb(n)} KB gzip first load (budget ${budget})`,
    );
    if (n / 1024 > budget) problems.push(`${r} ${kb(n)} KB > ${budget} KB`);
  }
  if (!Object.keys(routes).length)
    problems.push(
      "no lazy routes found in the entry chunk (parser out of date?)",
    );
  return { lines, problems };
}
if (
  process.argv[1] &&
  process.argv[1].endsWith("bundle-size.mjs") &&
  process.argv[2]
) {
  const { lines, problems } = report(process.argv[2]);
  console.log(lines.join("\n"));
  if (problems.length && process.argv.includes("--check")) {
    console.error(problems.join("\n"));
    process.exit(1);
  }
}
