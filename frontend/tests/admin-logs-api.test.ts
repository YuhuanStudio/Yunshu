import assert from "node:assert/strict";
import test from "node:test";
import {
  createSseParser,
  cursorStale,
  resumeCursor,
  formatLine,
  parseRecord,
} from "../src/admin-logs-api.ts";

test("the SSE parser joins split chunks and skips keepalive comments", () => {
  const got: string[] = [];
  const feed = createSseParser((d) => got.push(d));
  feed(': keepalive\n\nid: 1\ndata: {"a"');
  assert.deepEqual(got, []);
  feed(':1}\n\nid: 2\r\ndata: {"a":2}\r\n\r\n');
  assert.deepEqual(got, ['{"a":1}', '{"a":2}']);
});

test("malformed records are dropped, levels are upper-cased", () => {
  assert.equal(parseRecord({ id: "x" }), null);
  assert.equal(parseRecord(null), null);
  const r = parseRecord({
    id: 3,
    t: 1,
    level: "warning",
    logger: "l",
    msg: "m",
  });
  assert.equal(r?.level, "WARNING");
  assert.match(formatLine(r!), /WARNING l m$/);
});

test("a cursor above the ring's newest id means the engine restarted", () => {
  assert.equal(cursorStale(3000, 12), true);
  assert.equal(cursorStale(3000, 3000), false);
  assert.equal(cursorStale(3000, 3100), false);
  assert.equal(cursorStale(0, 0), false);
});

test("resumeCursor falls back to 0 after a restart and keeps the cursor otherwise", async () => {
  const real = globalThis.fetch;
  let nextId = 12;
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify({
        records: [],
        next_id: nextId,
        dropped: 0,
        capacity: 2000,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    )) as typeof fetch;
  try {
    const c = { baseUrl: "http://localhost:8000", token: "" };
    assert.equal(await resumeCursor(c, 3000), 0);
    nextId = 3000;
    assert.equal(await resumeCursor(c, 3000), 3000);
    assert.equal(await resumeCursor(c, 0), 0);
  } finally {
    globalThis.fetch = real;
  }
});
