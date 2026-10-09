// Minimal static server for the exported site (site/out): `node scripts/serve-out.mjs [port]`.
// Serves directory indexes, falls back to 404.html, binds 0.0.0.0 so a phone on the LAN can open it.
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const ROOT = new URL("../out", import.meta.url).pathname;
const PORT = Number(process.argv[2] ?? process.env.PORT ?? 3991);
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".svg": "image/svg+xml",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
};

async function pick(urlPath) {
  const clean = normalize(decodeURIComponent(urlPath.split("?")[0])).replace(/^(\.\.[/\\])+/, "");
  for (const c of [join(ROOT, clean), join(ROOT, clean, "index.html"), join(ROOT, clean + ".html")]) {
    try {
      if ((await stat(c)).isFile()) return c;
    } catch {
      /* next candidate */
    }
  }
  return null;
}

createServer(async (req, res) => {
  const file = await pick(req.url ?? "/");
  const status = file ? 200 : 404;
  const target = file ?? join(ROOT, "404.html");
  try {
    const body = await readFile(target);
    res.writeHead(status, { "Content-Type": TYPES[extname(target)] ?? "text/markdown; charset=utf-8" });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(PORT, "0.0.0.0", () => console.log(`serving ${ROOT} on 0.0.0.0:${PORT}`));
