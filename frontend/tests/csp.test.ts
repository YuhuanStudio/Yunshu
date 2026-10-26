import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("the console page only loads its own or attached images (no remote pixels from model Markdown)", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const m = /http-equiv="Content-Security-Policy"\s+content="([^"]+)"/.exec(
    html,
  );
  assert.ok(m, "index.html carries a Content-Security-Policy");
  const img = /(?:^|;)\s*img-src ([^;]+)/.exec(m[1]);
  assert.ok(img, "the policy restricts img-src");
  assert.deepEqual(img[1].trim().split(/\s+/).sort(), [
    "'self'",
    "blob:",
    "data:",
  ]);
});
