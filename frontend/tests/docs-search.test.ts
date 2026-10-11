import assert from "node:assert/strict";
import test from "node:test";
import { searchDocs } from "../src/docs/search.ts";
import { docHref, resolveDocLink, slugOf } from "../src/docs/links.ts";
import type { SearchEntry } from "../src/docs/types.ts";

const e = (
  slug: string,
  title: string,
  text: string,
  headings: SearchEntry["headings"] = [],
): SearchEntry => ({
  slug,
  title,
  description: "",
  headings,
  text,
});
const entries = [
  e(
    "guides/prompt-caching",
    "Prompt caching",
    "Reuse the cached prefix of a prompt across requests.",
  ),
  e(
    "api/models",
    "Models API",
    "Load and unload models. The prefix cache is cleared on unload.",
    [{ id: "unload", text: "Unload a model", level: 2 }],
  ),
  e("guides/cli", "CLI reference", "Every command and option."),
];

test("every word must match; titles outrank body text", () => {
  const hits = searchDocs(entries, "prompt cache");
  assert.deepEqual(
    hits.map((h) => h.slug),
    ["guides/prompt-caching"],
  );
  const pair = [
    e("a", "Intro", "all about the cli tool"),
    e("b", "CLI reference", "commands"),
  ];
  assert.deepEqual(
    searchDocs(pair, "cli").map((h) => h.slug),
    ["b", "a"],
  );
});

test("a heading match points at the section", () => {
  const hit = searchDocs(entries, "unload")[0];
  assert.equal(hit.slug, "api/models");
  assert.equal(hit.heading?.id, "unload");
});

test("no hit, empty query and the limit", () => {
  assert.deepEqual(searchDocs(entries, "zzzz"), []);
  assert.deepEqual(searchDocs(entries, "   "), []);
  assert.equal(searchDocs(entries, "e", 1).length, 1);
});

test("snippets start on a word and stay short", () => {
  const long = e(
    "x",
    "X",
    "alpha ".repeat(40) + "needle " + "omega ".repeat(40),
  );
  const s = searchDocs([long], "needle")[0].snippet;
  assert.ok(s.includes("needle") && s.length < 150);
  assert.match(s, /^… (alpha|needle)/);
});

test("doc links become console routes", () => {
  assert.equal(docHref("index"), "#/docs");
  assert.equal(docHref("api/audio", "x y"), "#/docs/api/audio?h=x%20y");
  assert.equal(slugOf(null), "index");
  assert.equal(resolveDocLink("/docs/api/audio", "index"), "#/docs/api/audio");
  assert.equal(
    resolveDocLink("/docs/api/audio#voices", "index"),
    "#/docs/api/audio?h=voices",
  );
  assert.equal(
    resolveDocLink("#errors", "api/audio"),
    "#/docs/api/audio?h=errors",
  );
  assert.equal(resolveDocLink("/docs", "x"), "#/docs");
  assert.equal(resolveDocLink("https://example.com", "x"), null);
});
