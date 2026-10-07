import assert from "node:assert/strict";
import test from "node:test";
import {
  phaseDistribution,
  activityHeatmap,
  latencyDistribution,
  observationCsv,
  observedRequests,
  percentile,
  timeSeries,
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
  const history = [snapshot(10), snapshot(60), snapshot(65)];
  history[1].status.requests.active = 2;
  history[2].status.requests.active = 4;
  const heat = activityHeatmap(history, 0, 100, 4);
  assert.deepEqual(heat.data[0], [0, null, 4, null]);
  assert.deepEqual(heat.coverage, [1, 0, 2, 0]);
});
test("time series and CSV preserve unavailable values rather than fabricate zero", () => {
  const rows = [snapshot(1000)];
  assert.equal(timeSeries(rows)[0].values.decode, null);
  assert.equal(timeSeries(rows)[0].values.cache, null);
  const csv = observationCsv(rows);
  assert.ok(csv.includes("mean_decode_tps_300s"));
  assert.ok(csv.includes('"1970-01-01T00:00:01.000Z","","700","0","10",""'));
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

test("trendDelta compares halves honestly and stays null without evidence", () => {
  assert.equal(trendDelta([10, 10]), null);
  assert.equal(trendDelta([10, 10, 10, 10, 10, 10]), null);
  assert.equal(trendDelta([0, 0, 0, 5, 5, 5]), null);
  const up = trendDelta([10, 10, 10, 15, 15, 15])!;
  assert.equal(Math.round(up.value), 50);
  assert.equal(up.positive, true);
  const slower = trendDelta([100, 100, 100, 150, 150, 150], {
    lowerIsBetter: true,
  })!;
  assert.equal(slower.positive, false);
  assert.equal(trendDelta([10, 10, 10, Number.NaN, 10, 10]), null);
});
