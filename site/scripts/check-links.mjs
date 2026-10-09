// Crawls the static export in out/ and fails on any broken internal link or missing #anchor.
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, posix } from "node:path";

const OUT = new URL("../out", import.meta.url).pathname;
const BASE = process.env.DOCS_BASE_PATH ?? "";

function walk(dir) {
  const files = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) files.push(...walk(p));
    else files.push(p);
  }
  return files;
}

function resolveTarget(urlPath) {
  let p = urlPath;
  if (BASE && p.startsWith(BASE)) p = p.slice(BASE.length) || "/";
  const candidates = [join(OUT, p), join(OUT, p, "index.html"), join(OUT, p + ".html")];
  return candidates.find((c) => existsSync(c) && statSync(c).isFile());
}

const pages = walk(OUT).filter((f) => f.endsWith(".html"));
const idCache = new Map();
function idsOf(file) {
  if (!idCache.has(file)) {
    const html = readFileSync(file, "utf8");
    idCache.set(file, new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1])));
  }
  return idCache.get(file);
}

const broken = [];
let checked = 0;
for (const file of pages) {
  const html = readFileSync(file, "utf8");
  const rel = file.slice(OUT.length);
  const pageUrl = BASE + rel.replace(/index\.html$/, "");
  for (const m of html.matchAll(/<a\b[^>]*?\shref="([^"]+)"/g)) {
    let href = m[1].replace(/&amp;/g, "&");
    if (/^(https?:|mailto:|tel:|javascript:|data:)/.test(href)) continue;
    const [pathPart, hash] = href.split("#");
    const base = pathPart === "" ? pageUrl : posix.resolve(pageUrl, pathPart.split("?")[0]);
    const urlPath = pathPart === "" ? pageUrl : pathPart.startsWith("/") ? pathPart.split("?")[0] : base;
    const target = resolveTarget(urlPath);
    checked++;
    if (!target) {
      broken.push(`${rel}: ${href} -> no such page`);
      continue;
    }
    if (hash && target.endsWith(".html") && !idsOf(target).has(decodeURIComponent(hash))) {
      broken.push(`${rel}: ${href} -> missing anchor #${hash}`);
    }
  }
}
console.log(`checked ${checked} internal links in ${pages.length} pages`);
if (broken.length) {
  console.error("Broken links:\n  " + [...new Set(broken)].join("\n  "));
  process.exit(1);
}
