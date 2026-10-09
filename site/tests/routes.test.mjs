import { test } from "node:test";
import assert from "node:assert/strict";
import { checkContent, findRouteMentions, loadRoutes } from "../scripts/route-check.mjs";

test("the OpenAPI dump is present and has the core routes", () => {
  const routes = loadRoutes().map((r) => `${r.method} ${r.path}`);
  assert.ok(routes.includes("POST /v1/chat/completions"));
  assert.ok(routes.includes("WS /v1/realtime"));
});

test("finder: extracts method+path and URL paths, honours NotServed", () => {
  const t = "POST /v1/foo and curl http://localhost:8000/v1/bar?x=1 <NotServed>GET /v1/skip</NotServed>";
  const m = findRouteMentions(t);
  assert.deepEqual(m, [
    { method: "POST", path: "/v1/foo" },
    { method: null, path: "/v1/bar" },
  ]);
  assert.ok(!m.some((x) => x.path === "/v1/skip"));
});

test("a made-up route is reported", () => {
  const routes = loadRoutes();
  assert.ok(!routes.some((r) => r.re.test("/v1/does-not-exist")));
});

test("every route mentioned in the docs exists in the OpenAPI schema", () => {
  const bad = checkContent();
  assert.deepEqual(bad, [], "docs mention routes the server does not register:\n" + bad.join("\n"));
});
