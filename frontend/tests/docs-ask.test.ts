import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { buildChunks } from "../scripts/docs-chunks.mjs";
import { readAskStream, docsEndpoint, makeTabId } from "../src/docs/ask-api.ts";

test("heading chunks preserve code and duplicate anchors; fence headings are not sources", () => {
  const dir = mkdtempSync(join(tmpdir(), "docs-ask-"));
  try {
    writeFileSync(
      join(dir, "index.mdx"),
      "---\ntitle: Start\n---\nIntro\n## Run\n```sh\n# not a heading\nyunshu serve\n```\n## Run\nAgain",
    );
    const { chunks } = buildChunks("en", dir);
    assert.deepEqual(
      chunks.map((c) => c.anchor),
      ["", "run", "run-1"],
    );
    assert.match(chunks[1].text, /```sh\n# not a heading/);
    assert.equal(chunks[2].url, "/docs#run-1");
    assert.equal(buildChunks("zh-TW", dir).chunks.length, 3);
  } finally {
    rmSync(dir, { recursive: true });
  }
});

test("SSE handles byte-split Chinese, sources and terminal done", async () => {
  const wire =
    'event: sources\ndata: {"id":"a","sources":[]}\n\nevent: token\ndata: {"text":"快取"}\n\nevent: done\ndata: {"id":"a"}\n\n';
  const bytes = new TextEncoder().encode(wire);
  const response = new Response(
    new ReadableStream({
      start(controller) {
        for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
        controller.close();
      },
    }),
  );
  const events: unknown[] = [];
  await readAskStream(response, (e) => events.push(e));
  assert.deepEqual(
    events.map((e) => (e as { type: string }).type),
    ["sources", "token", "done"],
  );
  assert.equal((events[1] as { text: string }).text, "快取");
});

test("SSE disconnect is not reported as a finished answer", async () => {
  await assert.rejects(
    () =>
      readAskStream(
        new Response('event: token\ndata: {"text":"partial"}\n\n'),
        () => {},
      ),
    /stream_incomplete/,
  );
  await assert.rejects(
    () =>
      readAskStream(
        new Response('event: error\ndata: {"code":"measurement_paused"}\n\n'),
        () => {},
      ),
    /measurement_paused/,
  );
});

test("docs requests target the attached console with or without v1 suffix", () => {
  for (const baseUrl of [
    "http://localhost:8100",
    "http://localhost:8100/",
    "http://localhost:8100/v1",
  ])
    assert.equal(
      docsEndpoint({ baseUrl, token: "" }, "ask"),
      "http://localhost:8100/console/api/docs/ask",
    );
});

test("pinned Orama fixture stays in sync with the docs tokenizer and ranking", async () => {
  const { readFileSync } = await import("node:fs");
  const { create, insertMultiple, search } = await import("@orama/orama");
  const { tokenizeDocs } = await import("../src/docs/orama-tokenizer.ts");
  const fixture = JSON.parse(
    readFileSync(
      new URL("../../tests/fixtures/docsask/orama.json", import.meta.url),
      "utf8",
    ),
  );
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
  await insertMultiple(db, fixture.chunks);
  for (const [query, expected] of Object.entries(fixture.expected)) {
    const result = await search(db, {
      term: query,
      properties: ["title", "heading", "description", "searchText"],
      boost: { title: 6, heading: 3, description: 2 },
      threshold: 1,
      limit: 10,
    });
    assert.deepEqual(
      result.hits.map((h) => h.document.id),
      expected,
    );
  }
});

test("SSE normalizes CRLF even when each byte arrives separately", async () => {
  const wire =
    'event: token\r\ndata: {"text":"ok"}\r\n\r\nevent: done\r\ndata: {"id":"a"}\r\n\r\n';
  const bytes = new TextEncoder().encode(wire);
  const response = new Response(
    new ReadableStream({
      start(c) {
        for (const byte of bytes) c.enqueue(new Uint8Array([byte]));
        c.close();
      },
    }),
  );
  const events: unknown[] = [];
  await readAskStream(response, (e) => events.push(e));
  assert.equal(events.length, 2);
});

test("tab ids do not depend on secure-context randomUUID", () => {
  const id = makeTabId();
  assert.match(id, /^[a-f0-9]{32}$/);
  assert.notEqual(makeTabId(), id);
});
