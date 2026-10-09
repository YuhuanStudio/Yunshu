import assert from "node:assert/strict";
import test from "node:test";
import { parseRoute, routeHref } from "../src/route.ts";

test("page, sub and query are separated", () => {
  const r = parseRoute("#/models/org%2FQwen-9B?action=load");
  assert.equal(r.page, "models");
  assert.equal(r.sub, "org/Qwen-9B");
  assert.equal(r.query.get("action"), "load");
});
test("every new page routes; an unknown one falls back to overview", () => {
  for (const p of ["downloads", "cache", "keys", "logs"])
    assert.equal(parseRoute(`#/${p}`).page, p);
  assert.equal(parseRoute("#/nope").page, "overview");
  assert.equal(parseRoute("").page, "overview");
});
test("a query on a page without sub does not leak into the page name", () => {
  const r = parseRoute("#/logs?from=1&to=2");
  assert.equal(r.page, "logs");
  assert.equal(r.query.get("to"), "2");
  assert.equal(r.sub, null);
});
test("routeHref round-trips", () => {
  const href = routeHref("keys", { action: "create" });
  assert.equal(href, "#/keys?action=create");
  assert.equal(parseRoute(href).query.get("action"), "create");
  assert.equal(parseRoute(routeHref("models", {}, "a/b")).sub, "a/b");
});

import { CHORDS, PAGES, VERBS, isTypingTarget } from "../src/route.ts";
test("every verb and chord leads to a real page", () => {
  for (const v of VERBS) assert.ok(PAGES.includes(parseRoute(v.href).page));
  assert.equal(parseRoute(VERBS[0].href).query.get("action"), "load");
  for (const page of Object.values(CHORDS)) assert.ok(PAGES.includes(page));
  assert.equal(new Set(Object.values(CHORDS)).size, Object.keys(CHORDS).length);
});
test("typing targets are left alone", () => {
  assert.ok(isTypingTarget({ tagName: "INPUT" }));
  assert.ok(isTypingTarget({ tagName: "DIV", isContentEditable: true }));
  assert.ok(!isTypingTarget({ tagName: "BUTTON" }));
  assert.ok(!isTypingTarget(null));
});

import { withoutIntent } from "../src/route.ts";
test("one-shot intent keys are stripped, other keys and the sub-route stay", () => {
  assert.equal(
    withoutIntent("#/models/org%2Fm?action=load&model=x&tab=a"),
    "#/models/org%2Fm?tab=a",
  );
  assert.equal(withoutIntent("#/cache?action=clear"), "#/cache");
  assert.equal(withoutIntent("#/logs?from=1&to=2"), "#/logs?from=1&to=2");
});
