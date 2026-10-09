import assert from "node:assert/strict";
import test from "node:test";
import {
  createSseParser,
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
