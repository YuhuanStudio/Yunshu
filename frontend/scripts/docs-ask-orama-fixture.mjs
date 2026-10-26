// Regenerate the cross-language BM25 parity fixture after an intentional Orama upgrade.
import { create, insertMultiple, search } from "@orama/orama";
import { writeFileSync } from "node:fs";
import { tokenizeDocs } from "../src/docs/orama-tokenizer.ts";
const chunks = [
  {
    id: "cache",
    title: "Prompt caching",
    heading: "APC prefix cache",
    description: "Reuse prefill work",
    searchText: "APC cache disk RAM prefix 快取 快取命中 缓存",
    text: "Cache",
  },
  {
    id: "api",
    title: "Chat completions",
    heading: "Streaming",
    description: "OpenAI API",
    searchText: "stream chat completions API code curl 串流 流式",
    text: "Chat",
  },
  {
    id: "start",
    title: "Quickstart",
    heading: "Start the engine",
    description: "Local models",
    searchText: "yunshu serve model start Python SDK OpenAI 本機 本地",
    text: "Start",
  },
  {
    id: "settings",
    title: "Configuration",
    heading: "Cache options",
    description: "Settings",
    searchText: "YUNSHU_VLM_APC_DISK disk cache tokens context",
    text: "Settings",
  },
];
const db = create({
  schema: {
    id: "string",
    title: "string",
    heading: "string",
    description: "string",
    searchText: "string",
  },
  components: { tokenizer: { language: "english", tokenize: tokenizeDocs } },
});
await insertMultiple(db, chunks);
const queries = [
  "APC prefix",
  "cache",
  "stream chat API",
  "OpenAI SDK",
  "快取",
  "缓存",
  "YUNSHU_VLM_APC_DISK",
  "noresults",
  "How do I stream chat completions?",
  "what is the OpenAI SDK",
  "如何設定快取？",
  "怎么使用缓存",
];
const tokenCases = [
  "APC 快取快取",
  "How do I start the engine with a local model?",
  "如何使用 OpenAI Python SDK 連接 Yunshu？",
  "缓存",
  "a 快 I",
];
const expected = {};
for (const query of queries)
  expected[query] = (
    await search(db, {
      term: query,
      properties: ["title", "heading", "description", "searchText"],
      boost: { title: 6, heading: 3, description: 2 },
      threshold: 1,
      limit: 10,
    })
  ).hits.map((h) => h.document.id);
writeFileSync(
  new URL("../../tests/fixtures/docsask/orama.json", import.meta.url),
  JSON.stringify(
    {
      chunks,
      expected,
      tokens: Object.fromEntries(tokenCases.map((t) => [t, tokenizeDocs(t)])),
    },
    null,
    2,
  ) + "\n",
);
