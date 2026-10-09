import assert from "node:assert/strict";
import test from "node:test";
import { parseServerHistory } from "../src/history-api.ts";
import { gapPoint, mergeBackfill, type SeriesPoint } from "../src/series.ts";

const pt = (
  at: number,
  decode: number | null,
  extra: Partial<SeriesPoint> = {},
): SeriesPoint => ({
  at,
  decode,
  prefill: null,
  active: null,
  queued: null,
  prefillRequests: null,
  decodeRequests: null,
  memActive: null,
  memCache: null,
  ...extra,
});

test("the metrics payload parses with its gaps in milliseconds", () => {
  const h = parseServerHistory(
    {
      object: "yunshu.metrics_history",
      resolution_s: 10,
      series: {
        t: [100, 110, 400],
        decode_tps: [1, null, 3],
        active_gb: [20, 20, 21],
      },
      gaps: [
        [120, 400],
        ["x", 1],
        [5, 3],
      ],
    },
    true,
  );
  assert.ok(h);
  assert.equal(h.points.length, 3);
  assert.equal(h.points[0].at, 100_000);
  assert.equal(h.points[1].decode, null);
  assert.equal(h.points[2].memActive, 21);
  assert.deepEqual(h.gaps, [[120_000, 400_000]]);
  assert.equal(h.intervalS, 10);
});

test("merge: live points win, recorded points fill what the console missed", () => {
  const live = [pt(10_000, 5), pt(12_500, 6)];
  const recorded = [
    pt(1_000, 1),
    pt(5_000, 2),
    pt(10_200, 99),
    pt(12_400, 99),
    pt(15_000, 7),
  ];
  const out = mergeBackfill(live, recorded);
  assert.deepEqual(
    out.map((p) => [p.at, p.decode]),
    [
      [1000, 1],
      [5000, 2],
      [10000, 5],
      [12500, 6],
      [15000, 7],
    ],
  );
  assert.ok(out[0].backfilled && !out[2].backfilled);
});

test("merge: an engine outage becomes a gap marker, not an interpolated line", () => {
  const recorded = [pt(1_000, 1), pt(2_000, 2), pt(300_000, 3)];
  const out = mergeBackfill([], recorded, [[3_000, 300_000]]);
  assert.deepEqual(
    out.map((p) => (p.gap ? "gap" : p.decode)),
    [1, 2, "gap", 3],
  );
});

test("merge: markers are not doubled and the result stays sorted", () => {
  const live = [pt(1_000, 1), gapPoint(5_000), pt(9_000, 2)];
  const out = mergeBackfill(live, [pt(3_000, 9)], [[5_000, 9_000]]);
  assert.deepEqual(
    out.map((p) => p.at),
    [1_000, 3_000, 5_000, 9_000],
  );
  assert.equal(out.filter((p) => p.gap).length, 1);
});

test("merge: nothing recorded leaves the live series alone", () => {
  const live = [pt(1_000, 1), pt(3_500, 2)];
  assert.deepEqual(mergeBackfill(live, []), live);
});

test("idle is a measurement: a recorded sample with request counts and no rate is 0, a gap is not", () => {
  const h = parseServerHistory(
    {
      series: {
        t: [1, 2, 3],
        decode_tps: [null, 40, null],
        prefill_tps: [null, null, null],
        requests_active: [0, 1, null],
      },
      gaps: [],
    },
    true,
  );
  assert.ok(h);
  assert.deepEqual(
    h.points.map((p) => p.decode),
    [0, 40, null],
  );
  assert.deepEqual(
    h.points.map((p) => p.prefill),
    [0, 0, null],
  );
});
