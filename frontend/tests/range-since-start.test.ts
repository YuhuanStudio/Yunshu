import assert from "node:assert/strict";
import test from "node:test";
import { rangeSeconds, sinceStartId } from "../src/useRangeHistory.ts";

test("since engine start is a window as long as the uptime, rounded up to 5 minutes, at least 15", () => {
  assert.equal(sinceStartId(0), "s900");
  assert.equal(sinceStartId(1000), "s1200");
  assert.equal(rangeSeconds(sinceStartId(7300)), 7500);
  assert.equal(rangeSeconds("1h"), 3600);
  assert.equal(rangeSeconds("nonsense"), 900);
});
