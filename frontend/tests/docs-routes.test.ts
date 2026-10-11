import assert from "node:assert/strict";
import test from "node:test";
import { findRouteMentions, loadRoutes, walk, CONTENT } from "../scripts/docs-routes.mjs";

// The check against the routes the server really registers is tests/unit/test_docs_routes.py
// (it dumps them from the live app); these pin the finder itself.

test("finder: extracts method+path and URL paths, honours NotServed", () => {
  const t =
    "POST /v1/foo and curl http://localhost:8000/v1/bar?x=1 <NotServed>GET /v1/skip</NotServed>";
  const m = findRouteMentions(t);
  assert.deepEqual(m, [
    { method: "POST", path: "/v1/foo" },
    { method: null, path: "/v1/bar" },
  ]);
  assert.ok(!m.some((x) => x.path === "/v1/skip"));
});

test("a made-up route does not match a registered one", async () => {
  const { writeFileSync, mkdtempSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const f = join(mkdtempSync(join(tmpdir(), "routes-")), "routes.json");
  writeFileSync(f, JSON.stringify(["POST /v1/chat/completions", "GET /v1/files/{file_id}"]));
  const routes = loadRoutes(f);
  assert.ok(routes.some((r: { re: RegExp }) => r.re.test("/v1/chat/completions")));
  assert.ok(routes.some((r: { re: RegExp }) => r.re.test("/v1/files/abc")));
  assert.ok(!routes.some((r: { re: RegExp }) => r.re.test("/v1/does-not-exist")));
});

test("the docs content directory has the three-locale page sets", () => {
  const files = walk(CONTENT).map((f: string) => f.slice(CONTENT.length + 1));
  assert.ok(files.length > 100);
  assert.ok(files.includes("index.mdx") && files.includes("index.zh-TW.mdx"));
});
