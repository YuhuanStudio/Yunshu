import assert from "node:assert/strict";
import test from "node:test";
import { pageOwnsEscape } from "../src/escape-guard.ts";

const key = (over: Record<string, unknown> = {}) => ({
  key: "Escape",
  defaultPrevented: false,
  target: { tagName: "BODY" },
  ...over,
});

test("Escape closes the page's inspector only when nothing else wants the key", () => {
  assert.equal(pageOwnsEscape(key(), false), true);
  assert.equal(pageOwnsEscape(key({ key: "Enter" }), false), false);
  assert.equal(pageOwnsEscape(key(), true), false);
  assert.equal(pageOwnsEscape(key({ defaultPrevented: true }), false), false);
  assert.equal(
    pageOwnsEscape(key({ target: { tagName: "INPUT" } }), false),
    false,
  );
  assert.equal(
    pageOwnsEscape(key({ target: { tagName: "TEXTAREA" } }), false),
    false,
  );
  assert.equal(
    pageOwnsEscape(
      key({ target: { tagName: "DIV", isContentEditable: true } }),
      false,
    ),
    false,
  );
});
