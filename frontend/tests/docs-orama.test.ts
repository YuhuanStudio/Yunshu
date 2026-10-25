import assert from "node:assert/strict";
import test from "node:test";
import { insertMultiple, save } from "@orama/orama";
import { createDocsDatabase, queryDocs } from "../src/docs/orama.ts";
import { buildSearch, LOCALES } from "../scripts/docs-index.mjs";

for (const [locale, term] of [
  ["en", "prompt caching"],
  ["zh-TW", "快取"],
  ["zh-CN", "缓存"],
]) {
  test(`static Orama roundtrip retrieves ${locale} page and section`, async () => {
    const entries = buildSearch(locale);
    const db = createDocsDatabase();
    await insertMultiple(
      db,
      entries.map((e) => ({
        ...e,
        headings: e.headings.map((h) => h.text).join(" "),
      })),
    );
    const index = JSON.parse(JSON.stringify({ data: save(db), entries }));
    const hits = await queryDocs(index, term);
    assert.ok(hits.some((h) => h.slug === "guides/prompt-caching"));
    assert.deepEqual(await queryDocs(index, "not_a_real_word_97"), []);
    assert.deepEqual(await queryDocs(index, "  "), []);
    assert.equal((await queryDocs(index, term, 1)).length, 1);
  });
}

test("three locale indexes never accidentally use another locale's text", () => {
  const titles = LOCALES.map(
    (locale) =>
      buildSearch(locale).find((e) => e.slug === "getting-started/install")!
        .title,
  );
  assert.equal(new Set(titles).size, 3);
});
