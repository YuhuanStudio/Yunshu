// Shared by tests/routes.test.mjs: finds every route mentioned in the docs content and checks
// it against generated/routes.json (dumped from the live FastAPI app by scripts/dump_openapi.py).
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;
export const CONTENT = join(ROOT, "content/docs");

export function loadRoutes() {
  const routes = JSON.parse(readFileSync(join(ROOT, "generated/routes.json"), "utf8"));
  return routes.map((r) => {
    const [method, path] = r.split(" ");
    const re = new RegExp(
      "^" +
        path
          .replace(/[.+*?^$()|[\]\\]/g, "\\$&")
          .replace(/\{[^}]*:path\}/g, ".+")
          .replace(/\{[^}]*\}/g, "[^/]+") +
        "$",
    );
    return { method, path, re };
  });
}

export function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else if (name.endsWith(".mdx")) out.push(p);
  }
  return out;
}

const METHOD_PATH = /\b(GET|POST|DELETE|PUT|PATCH|WS)\s+(\/[A-Za-z0-9_\-./{}:*]*)/g;
const URL_PATH = /(?:localhost|127\.0\.0\.1):\d+(\/[A-Za-z0-9_\-./{}:]*)/g;

function clean(p) {
  p = p.replace(/[.,;:]+$/, "");
  if (p.length > 1) p = p.replace(/\/+$/, "");
  return p;
}

export function findRouteMentions(text) {
  const stripped = text.replace(/<NotServed>[\s\S]*?<\/NotServed>/g, "");
  const found = [];
  for (const m of stripped.matchAll(METHOD_PATH)) found.push({ method: m[1], path: clean(m[2]) });
  for (const m of stripped.matchAll(URL_PATH)) found.push({ method: null, path: clean(m[1]) });
  return found;
}

export function checkContent() {
  const routes = loadRoutes();
  const bad = [];
  for (const file of walk(CONTENT)) {
    const text = readFileSync(file, "utf8");
    for (const { method, path } of findRouteMentions(text)) {
      const ok =
        routes.some((r) => (method ? r.method === method : true) && r.re.test(path)) ||
        // a bare base URL such as http://localhost:8000/v1 is a prefix of served routes
        (!method && routes.some((r) => r.path.startsWith(path + "/")));
      if (!ok) bad.push(`${relative(CONTENT, file)}: ${method ?? "URL"} ${path}`);
    }
  }
  return bad;
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const bad = checkContent();
  if (bad.length) {
    console.error("Routes mentioned in the docs that the server does not register:\n  " + bad.join("\n  "));
    process.exit(1);
  }
  console.log("all routes mentioned in the docs exist");
}
