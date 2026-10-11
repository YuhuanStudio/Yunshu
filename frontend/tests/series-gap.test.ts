import assert from "node:assert/strict";
import test from "node:test";
import { chartRows, gapPoint, type SeriesPoint } from "../src/series.ts";

const pt = (at: number, decode: number): SeriesPoint => ({
  at,
  decode,
  prefill: null,
  active: 1,
  queued: 0,
  prefillRequests: 0,
  decodeRequests: 1,
  memActive: 40,
  memCache: 2,
});

test("a gap marker is an all-null row, never a value", () => {
  const rows = chartRows([pt(1000, 50), gapPoint(4000), pt(30000, 60)]);
  assert.equal(rows.length, 3);
  assert.deepEqual(Object.values(rows[1].values), Array(8).fill(null));
});
test("downsampling keeps the marker as its own row between the two sides", () => {
  const points: SeriesPoint[] = [];
  for (let i = 0; i < 10; i++) points.push(pt(i * 1000, 100));
  points.push(gapPoint(10_000));
  for (let i = 0; i < 10; i++) points.push(pt(60_000 + i * 1000, 10));
  const rows = chartRows(points, 5); // groups of 4 points
  const at = rows.findIndex((r) => r.values.decode === null);
  assert.ok(at > 0 && at < rows.length - 1);
  assert.equal(rows[at - 1].values.decode, 100);
  assert.equal(rows[at + 1].values.decode, 10);
  assert.equal(rows.filter((r) => r.values.decode === null).length, 1);
  for (let i = 1; i < rows.length; i++) assert.ok(rows[i].x >= rows[i - 1].x);
});
