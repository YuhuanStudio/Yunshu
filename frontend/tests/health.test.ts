import assert from "node:assert/strict";
import test from "node:test";
import { parseEngineStatus } from "../src/api.ts";
import { healthVerdict, type FinishedFact } from "../src/health.ts";

const NOW = 1_000_000_000;
const mk = (
  over: {
    memory?: object;
    items?: object[];
    queued?: number;
    load_error?: string;
  } = {},
) =>
  parseEngineStatus({
    object: "yunshu.status",
    version: "t",
    state: "running",
    uptime_s: 100,
    load_error: over.load_error ?? null,
    models: [],
    memory: { active_gb: 40, total_gb: 128, ...over.memory },
    requests: {
      active: 0,
      queued: over.queued ?? 0,
      prefill: 0,
      decode: 0,
      items: over.items ?? [],
    },
    last: null,
    throughput: {
      window_s: 60,
      requests: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      live_decode_tps: null,
      mean_prefill_tps: null,
      mean_decode_tps: null,
    },
  });
const run = (
  over: Parameters<typeof mk>[0] = {},
  extra: {
    finished?: FinishedFact[];
    ledger?: {
      pressureLevel: string | null;
      swapUsedGb: number | null;
      swapGrowthGb?: number | null;
    };
  } = {},
) => healthVerdict({ phase: "online", status: mk(over), now: NOW, ...extra });
const codes = (v: ReturnType<typeof run>) => v.reasons.map((r) => r.code);
const done = (n: number, over: Partial<FinishedFact> = {}): FinishedFact[] =>
  Array.from({ length: n }, (_, i) => ({
    at: NOW - 1000 * i,
    outcome: "completed",
    statusCode: 200,
    ...over,
  }));

test("a quiet engine is ok with no reasons", () => {
  assert.deepEqual(run(), { level: "ok", reasons: [] });
});
test("memory: watch at 80 %, bad at 92 %", () => {
  assert.equal(run({ memory: { active_gb: 105 } }).level, "watch");
  const bad = run({ memory: { active_gb: 121 } });
  assert.equal(bad.level, "bad");
  assert.equal(bad.reasons[0].vars.pct, 95);
});
test("missing memory numbers add no reason", () => {
  const v = healthVerdict({
    phase: "online",
    status: mk({ memory: { active_gb: undefined, total_gb: undefined } }),
    now: NOW,
  });
  assert.equal(v.level, "ok");
});
test("queue depth and oldest wait escalate separately", () => {
  const queuedRow = (s: number) => ({
    request_id: "r" + s,
    elapsed_s: s,
    phase: "queued",
  });
  const watch = run({ queued: 1, items: [queuedRow(12)] });
  assert.deepEqual(codes(watch), ["oldestWait"]);
  assert.equal(watch.level, "watch");
  const bad = run({ queued: 4, items: [queuedRow(43), queuedRow(2)] });
  assert.equal(bad.level, "bad");
  assert.deepEqual(codes(bad).sort(), ["oldestWait", "queueDepth"]);
  assert.equal(run({ queued: 1, items: [queuedRow(3)] }).level, "ok");
});
test("a decoding request's elapsed time is not a queue wait", () => {
  const v = run({
    items: [{ request_id: "a", elapsed_s: 900, phase: "decode" }],
  });
  assert.equal(v.level, "ok");
});
test("error rate needs 10 requests; 5xx counts from one", () => {
  const few = [...done(3), ...done(3, { outcome: "error", statusCode: 400 })];
  assert.equal(run({}, { finished: few }).level, "ok");
  const many = [...done(17), ...done(3, { outcome: "error", statusCode: 400 })];
  assert.deepEqual(codes(run({}, { finished: many })), ["errorRate"]);
  const five = [...done(30), ...done(1, { outcome: "error", statusCode: 503 })];
  assert.deepEqual(codes(run({}, { finished: five })), ["fivexx"]);
  const burst = [
    ...done(30),
    ...done(3, { outcome: "error", statusCode: 500 }),
  ];
  assert.equal(run({}, { finished: burst }).level, "bad");
});
test("cancelled requests are not errors; old ones fall out of the window", () => {
  const cancelled = done(20, { outcome: "cancelled" });
  assert.equal(run({}, { finished: cancelled }).level, "ok");
  const old = done(20, { outcome: "error", statusCode: 500 }).map((r) => ({
    ...r,
    at: NOW - 10 * 60_000,
  }));
  assert.equal(run({}, { finished: old }).level, "ok");
});
test("swap and host pressure", () => {
  assert.equal(
    run({}, { ledger: { pressureLevel: "normal", swapUsedGb: 0 } }).level,
    "ok",
  );
  assert.equal(
    run({}, { ledger: { pressureLevel: "warn", swapUsedGb: null } }).level,
    "watch",
  );
  assert.equal(
    run({}, { ledger: { pressureLevel: null, swapUsedGb: 9.4 } }).level,
    "ok",
  ); // a steady 9.4 GB of old swap is not a fault
  assert.equal(
    run(
      {},
      {
        ledger: {
          pressureLevel: null,
          swapUsedGb: 9.4,
          swapGrowthGb: 3.2,
        },
      },
    ).level,
    "bad",
  );
});
test("offline and load error are bad; a refused token is a watch", () => {
  assert.equal(
    healthVerdict({ phase: "offline", status: null, now: NOW }).level,
    "bad",
  );
  assert.equal(
    healthVerdict({ phase: "unauthorized", status: null, now: NOW }).level,
    "watch",
  );
  assert.deepEqual(codes(run({ load_error: "boom" })), ["loadError"]);
  assert.equal(
    healthVerdict({ phase: "connecting", status: null, now: NOW }).level,
    "ok",
  );
});
test("reasons are ordered worst first", () => {
  const v = run({ memory: { active_gb: 105 }, queued: 9 });
  assert.equal(v.reasons[0].level, "bad");
  assert.equal(v.reasons.at(-1)?.level, "watch");
});

import { factsFromRows } from "../src/health.ts";
test("ring rows become facts: seconds to ms, rows without time or outcome dropped", () => {
  const facts = factsFromRows([
    { t: 1700000000, outcome: "error", status_code: 503 },
    { t: 1700000000500, outcome: "completed", status_code: 200 },
    { outcome: "completed" },
    { t: 5 },
  ]);
  assert.deepEqual(facts, [
    { at: 1700000000000, outcome: "error", statusCode: 503 },
    { at: 1700000000500, outcome: "completed", statusCode: 200 },
  ]);
});
