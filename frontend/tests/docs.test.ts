import assert from "node:assert/strict";
import test from "node:test";
import { buildSearch, buildToc, flatten, headingsOf, LOCALES, parseName } from "../scripts/docs-index.mjs";
import { checkLinks } from "../scripts/docs-links.mjs";

test("file names map to slug and locale", () => {
  assert.deepEqual(parseName("api/audio.zh-TW.mdx"), { slug: "api/audio", locale: "zh-TW" });
  assert.deepEqual(parseName("index.mdx"), { slug: "index", locale: "en" });
  assert.equal(parseName("meta.json"), null);
});

test("headings carry the ids rehype-slug gives them, duplicates numbered, code fences skipped", () => {
  const h = headingsOf("# T\n\n## Setup `x`\n\n```\n## not a heading\n```\n\n## Setup `x`\n\n### 中文 標題\n");
  assert.deepEqual(h, [
    { id: "setup-x", text: "Setup x", level: 2 },
    { id: "setup-x-1", text: "Setup x", level: 2 },
    { id: "中文-標題", text: "中文 標題", level: 3 },
  ]);
});

for (const locale of LOCALES) {
  test(`${locale}: the tree lists every page once, in order, with titles`, () => {
    const { tree, pages } = buildToc(locale);
    const order = flatten(tree);
    assert.equal(new Set(order).size, order.length);
    assert.deepEqual([...order].sort(), Object.keys(pages).sort());
    assert.ok(order.length >= 40);
    for (const slug of order) assert.ok(pages[slug].title, `${slug} has a title`);
    assert.equal(order[0], "index");
  });
}

test("zh pages use their own translation, never the English text", () => {
  const { pages } = buildToc("zh-TW");
  assert.equal(pages["getting-started/install"].locale, "zh-TW");
  assert.ok(/[一-鿿]/.test(pages["getting-started/install"].title));
});

test("search entries carry plain text for every page", () => {
  const entries = buildSearch("en");
  const chat = entries.find((e) => e.slug === "api/chat-completions");
  assert.ok(chat && chat.text.includes("completions") && !chat.text.includes("<Endpoint"));
  assert.ok(entries.every((e) => e.title && e.text.length > 50));
});

test("every internal docs link resolves to a page and a heading", () => {
  const { broken, checked } = checkLinks();
  assert.deepEqual(broken, []);
  assert.ok(checked > 500);
});
