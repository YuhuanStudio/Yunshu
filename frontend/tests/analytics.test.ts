import assert from "node:assert/strict";
import test from "node:test";
import {
  phaseDistribution,
  activityHeatmap,
  latencyDistribution,
  observationCsv,
  observedRequests,
  percentile,
  rollingMedian,
  trendDelta,
} from "../src/analytics.ts";
import type { EngineStatus } from "../src/api.ts";
const snapshot = (at: number, last: EngineStatus["last"] = null) => ({
  at,
  status: {
    object: "yunshu.status",
    version: "test",
    state: "running",
    uptime_s: at,
    load_error: null,
    models: [],
    memory: { active_gb: 10 },
    requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
    last,
    throughput: {
      window_s: 60,
      requests: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      live_decode_tps: null,
      mean_decode_tps: null,
      mean_prefill_tps: 700,
    },
  } as EngineStatus,
});
const ended = (id: string, ttft: number | null) => ({
  request_id: id,
  t: 1,
  prompt_tokens: 100,
  cached_tokens: 25,
  completion_tokens: 20,
  ttft_ms: ttft,
  prefill_tps: 700,
  decode_tps: 40,
});
test("repeated last-request polls are not counted as new completions", () => {
  const rows = observedRequests([
    snapshot(1, ended("a", 250)),
    snapshot(2, ended("a", 250)),
    snapshot(3, ended("b", 500)),
  ]);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].firstObservedAt, 1);
  const bins = latencyDistribution(rows);
  assert.equal(
    bins.reduce((n, b) => n + b.value, 0),
    2,
  );
  assert.equal(bins[1].value, 1);
  assert.equal(bins[2].value, 1);
});
test("latency percentiles exclude missing/invalid observations", () => {
  assert.equal(percentile([null, NaN, -1], 0.95), null);
  assert.equal(percentile([100, 200, 400, null], 0.5), 200);
  assert.equal(percentile([100, 200, 400], 0.95), 400);
  assert.deepEqual(
    latencyDistribution(observedRequests([snapshot(1, ended("x", null))])),
    [],
  );
});
test("heatmap distinguishes observed zero, missing intervals, and bucket peaks", () => {
  const pt = (at: number, active: number | null) => ({
    at,
    decode: null,
    prefill: null,
    active,
    queued: 0,
    prefillRequests: null,
    decodeRequests: null,
    memActive: null,
    memCache: null,
  });
  const heat = activityHeatmap([pt(10, 0), pt(60, 2), pt(65, 4)], 0, 100, 4);
  assert.deepEqual(heat.data[0], [0, null, 4, null]);
  assert.deepEqual(heat.coverage, [1, 0, 2, 0]);
  assert.equal(
    heat.data[2][0],
    null,
    "unreported per-phase counts stay unknown",
  );
});
test("CSV names the engine window and never says 300s", () => {
  const rows = [snapshot(1000)];
  const csv = observationCsv(rows);
  assert.ok(csv.includes("mean_decode_tps_window"));
  assert.ok(csv.includes("window_s"));
  assert.ok(!csv.includes("300s"));
  assert.ok(
    csv.includes('"1970-01-01T00:00:01.000Z","","","700","60","0","10",""'),
  );
  assert.ok(!csv.includes("NaN"));
});

test("unknown request phases cannot collide with object prototypes", () => {
  const sample = snapshot(1);
  sample.status.requests.items = ["__proto__", "constructor", "toString"].map(
    (phase) => ({ request_id: phase, phase, elapsed_s: 1 }),
  );
  const result = phaseDistribution(sample.status);
  assert.deepEqual(
    result.map((row) => [row.label, row.value, row.tone]),
    [
      ["__proto__", 1, "neutral"],
      ["constructor", 1, "neutral"],
      ["toString", 1, "neutral"],
    ],
  );
});

const ramp = (a: number, b: number, n = 10) => [
  ...Array(n).fill(a),
  ...Array(n).fill(b),
];

test("trendDelta needs enough samples on both halves and a stable base", () => {
  assert.equal(trendDelta([10, 10]), null);
  assert.equal(trendDelta(ramp(10, 15, 9)), null);
  assert.equal(trendDelta(ramp(10, 10)), null);
  assert.equal(trendDelta(ramp(0, 5)), null);
  assert.equal(trendDelta([...ramp(10, 15).slice(0, 19), Number.NaN]), null);
  const bursty = [
    ...Array(5).fill(1),
    ...Array(5).fill(100),
    ...Array(10).fill(50),
  ];
  assert.equal(trendDelta(bursty), null);
});

test("trendDelta increase: +50% points up, neutral, labelled", () => {
  const up = trendDelta(ramp(10, 15))!;
  assert.equal(Math.round(up.value), 50);
  assert.equal(up.direction, "up");
  assert.equal(up.positive, true);
  assert.equal(up.neutral, true);
  assert.equal(up.label, "50%");
});

test("trendDelta decrease: -50% points down and never exceeds 100%", () => {
  const down = trendDelta(ramp(10, 5))!;
  assert.equal(Math.round(down.value), -50);
  assert.equal(down.direction, "down");
  assert.equal(down.positive, false);
  assert.equal(trendDelta(ramp(10, 0.001))!.label, "100%");
});

test("trendDelta lower-is-better keeps direction apart from good/bad", () => {
  const slower = trendDelta(ramp(100, 250), { lowerIsBetter: true })!;
  assert.equal(slower.direction, "up");
  assert.equal(slower.positive, false);
  assert.equal(slower.label, ">100%");
  const faster = trendDelta(ramp(100, 50), { lowerIsBetter: true })!;
  assert.equal(faster.direction, "down");
  assert.equal(faster.positive, true);
});

test("trendDelta caps large rises as >100%", () => {
  assert.equal(trendDelta(ramp(10, 50))!.label, ">100%");
});

test("rollingMedian damps single-request spikes without inventing points", () => {
  assert.deepEqual(
    rollingMedian([100, 100, 900, 100, 100], 3),
    [100, 100, 100, 100, 100],
  );
  assert.deepEqual(rollingMedian([1, 3], 5), [1, 2]);
  assert.deepEqual(rollingMedian([]), []);
});
