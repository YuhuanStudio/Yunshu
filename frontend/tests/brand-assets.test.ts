import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { MANIFEST, SHARED } from "../scripts/brand-icons.mjs";

const root = new URL("..", import.meta.url).pathname;
const html = readFileSync(`${root}index.html`, "utf8");
// index.html names public files without the base ("/x"; vite serves them at /console/x); the manifest, which
// vite does not rewrite, names them with it
const inPublic = (href: string) =>
  existsSync(`${root}public/${href.replace(/^\/console\//, "/").slice(1)}`);

test("index.html declares the icon set and every link resolves to a file", () => {
  const links = [...html.matchAll(/<link\b[^>]*>/g)].map((m) => m[0]);
  const rel = (name: string) =>
    links.filter((l) => new RegExp(`rel="${name}"`).test(l));
  assert.ok(rel("icon").length >= 2, "an svg and an ico icon");
  assert.ok(rel("apple-touch-icon").length >= 1);
  assert.ok(rel("manifest").length === 1);
  for (const l of links) {
    const href = l.match(/href="([^"]+)"/)?.[1];
    if (!href || !href.startsWith("/")) continue;
    assert.ok(inPublic(href), `${href} is missing from public/`);
  }
  const themes = [...html.matchAll(/<meta\s+name="theme-color"[^>]*>/g)];
  assert.ok(themes.length >= 2, "theme-color for light and dark");
});

test("the manifest in public/ is the generated one and points at real files", () => {
  const onDisk = JSON.parse(
    readFileSync(`${root}public/manifest.json`, "utf8"),
  );
  assert.deepEqual(onDisk, MANIFEST);
  for (const icon of onDisk.icons) assert.ok(inPublic(icon.src), icon.src);
});

test("every brand image is the shared YuhuanStudio icon (same bytes as Yunxin / YunUI)", () => {
  for (const [name, sha] of Object.entries(SHARED))
    assert.equal(
      createHash("sha1").update(readFileSync(`${root}public/${name}`)).digest("hex"),
      sha,
      name,
    );
});
