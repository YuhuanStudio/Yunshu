import assert from "node:assert/strict";
import test from "node:test";
import type { EngineStatus } from "../src/api.ts";
import { parseEngineStatus } from "../src/api.ts";
import {
  activity,
  decodeHeadline,
  livePill,
  tabTitle,
  totalsFrom,
  windowLabel,
} from "../src/engineView.ts";
import { percentileWhenEnough } from "../src/analytics.ts";
import { parseServerHistory } from "../src/history-api.ts";
import {
  chartRows,
  mergeSeries,
  windowPoints,
  type SeriesPoint,
} from "../src/series.ts";

const base = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
  object: "yunshu.status",
  version: "t",
  state: "running",
  uptime_s: 10,
  load_error: null,
  models: [],
  memory: { active_gb: 10 },
  requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
  last: null,
  throughput: {
    window_s: 60,
    requests: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    live_decode_tps: null,
    mean_decode_tps: 48,
    mean_prefill_tps: 700,
  },
  ...over,
});
const status = (over?: Record<string, unknown>) =>
  parseEngineStatus(base(over)) as EngineStatus;
const last = {
  request_id: "r1",
  prompt_tokens: 100,
  completion_tokens: 20,
  cached_tokens: 50,
  prefill_tps: 700,
  decode_tps: 25,
  ttft_ms: 100,
  t: 1,
};

test("the window label comes from throughput.window_s", () => {
  assert.equal(windowLabel(status()), "近 60 秒");
  const s = status();
  s.throughput.window_s = 30;
  assert.equal(windowLabel(s), "近 30 秒");
});

test("a request in prefill is never called idle", () => {
  const s = status({
    requests: {
      active: 1,
      queued: 0,
      prefill: 1,
      decode: 0,
      items: [
        {
          request_id: "a",
          elapsed_s: 1,
          phase: "prefill",
          prompt_tokens: 1000,
          processed_tokens: 250,
        },
      ],
    },
  });
  const a = activity(s);
  assert.equal(a.phase, "prefill");
  assert.equal(a.progress, 25);
  const h = decodeHeadline(s);
  assert.ok(h.note.includes("預填中"));
  assert.ok(!h.note.includes("閒置"));
  assert.equal(livePill("online", s).phase, "預填");
});

test("idle: the top bar says 閒置, the card labels the last request", () => {
  const s = status({ last });
  assert.equal(livePill("online", s).phase, "閒置");
  assert.equal(livePill("online", s).detail, "");
  const h = decodeHeadline(s);
  assert.equal(h.label, "最近一筆");
  assert.equal(h.value, 25);
  assert.ok(!tabTitle(s, "x").includes("tok/s"));
});

test("live decode is labelled as a live aggregate", () => {
  const s = status({
    requests: { active: 2, queued: 0, prefill: 0, decode: 2, items: [] },
    throughput: { ...(base().throughput as object), live_decode_tps: 80 },
  });
  const h = decodeHeadline(s);
  assert.equal(h.label, "即時合計");
  assert.equal(h.value, 80);
  assert.equal(livePill("online", s).detail, "80.0 tok/s");
});

test("null or malformed optional fields never fail the status", () => {
  const s = status({
    models: [
      {
        id: "a",
        type: "llm",
        loaded: true,
        loading: false,
        pinned: false,
        size_gb: null,
        idle_s: "x",
      },
      { id: "b", type: "llm", loaded: false, loading: false, pinned: false },
    ],
    memory: { active_gb: null, total_gb: "n/a", cache_gb: 2 },
  });
  assert.equal(s.models[0].size_gb, undefined);
  assert.equal(s.memory.active_gb, undefined);
  assert.equal(s.memory.cache_gb, 2);
});

test("percentiles need 20 samples", () => {
  const few = Array.from({ length: 19 }, (_, i) => i);
  assert.equal(percentileWhenEnough(few, 0.95), null);
  assert.equal(percentileWhenEnough([...few, 19], 0.5), 9);
});

const point = (at: number, decode: number | null = null): SeriesPoint => ({
  at,
  decode,
  prefill: null,
  active: 0,
  queued: 0,
  prefillRequests: 0,
  decodeRequests: 0,
  memActive: 1,
  memCache: 0,
});

test("server history parses, tolerates gaps, and rejects other shapes", () => {
  const h = parseServerHistory({
    enabled: true,
    interval_s: 5,
    series: {
      t: [100, 105, 105, 110],
      decode_tps: [null, 30, 31, 40],
      prefill_tps: [null, null, null, null],
      requests_active: [0, 1, 1, 1],
      queued: [0, 0, 0, 0],
      active_gb: [1, 1, 1, 1],
      cache_gb: [0, 0, 0, 0],
    },
  });
  assert.equal(h?.points.length, 3, "duplicate timestamps are dropped");
  assert.equal(h?.points[0].at, 100_000);
  assert.equal(h?.points[1].decode, 30);
  assert.equal(h?.points[1].prefillRequests, null);
  assert.equal(
    parseServerHistory({ enabled: false, series: { t: [1] } }),
    null,
  );
  assert.equal(parseServerHistory({ object: "x" }), null);
  assert.equal(parseServerHistory(null), null);
});

test("merge puts engine rows first and drops overlap with live samples", () => {
  const merged = mergeSeries(
    [point(1000), point(2000), point(3000)],
    [point(2500), point(5500)],
    6000,
  );
  assert.deepEqual(
    merged.map((p) => p.at),
    [1000, 2000, 2500, 5500],
  );
});

test("window slice is exact and chart rows are bounded", () => {
  const pts = Array.from({ length: 1200 }, (_, i) => point(i * 3000, i));
  assert.equal(windowPoints(pts, 3000, 9000).length, 3);
  const rows = chartRows(pts, 300);
  assert.ok(rows.length <= 300);
  assert.equal(chartRows(pts.slice(0, 10), 300).length, 10);
  assert.equal(
    chartRows(pts.slice(0, 10))[3],
    chartRows(pts.slice(0, 10))[3],
    "rows are cached",
  );
});

test("one append is O(1): per-point work does not grow with history", () => {
  const time = (n: number) => {
    const pts = Array.from({ length: n }, (_, i) => point(i * 3000, i));
    chartRows(pts); // warm the per-point cache
    pts.push(point(n * 3000, 1));
    const t0 = performance.now();
    for (let k = 0; k < 50; k++) {
      windowPoints(pts, pts[0].at, pts[pts.length - 1].at);
      chartRows(pts);
    }
    return (performance.now() - t0) / 50;
  };
  time(1200);
  assert.ok(time(1200) < 3, "1200 points: under 3 ms per poll of analytics");
});

test("totals are token-weighted and counted only from what was seen", () => {
  assert.equal(totalsFrom([]), null);
  const t = totalsFrom([
    {
      prompt_tokens: 1000,
      cached_tokens: 0,
      completion_tokens: 100,
      prefill_tps: 1000,
      decode_tps: 50,
      firstObservedAt: 5,
    },
    {
      prompt_tokens: 1000,
      cached_tokens: 900,
      completion_tokens: 100,
      prefill_tps: 100,
      decode_tps: null,
      firstObservedAt: 9,
    },
  ])!;
  assert.equal(t.requests, 2);
  assert.equal(t.since, 5);
  assert.equal(Math.round(t.prefillTps!), 550);
  assert.equal(t.decodeTps, 50);
});

test("a loading model shows as loading, never idle", () => {
  const status = parseEngineStatus({
    object: "yunshu.status",
    version: "t",
    state: "running",
    uptime_s: 10,
    load_error: null,
    models: [
      {
        id: "/m/Qwen3.5-9B",
        type: "LLM",
        loaded: false,
        loading: true,
        pinned: false,
      },
    ],
    memory: {},
    requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
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
  const pill = livePill("online", status);
  assert.equal(pill.phase, "載入中");
  assert.equal(pill.detail, "Qwen3.5-9B");
  assert.equal(pill.tone, "away");
});
