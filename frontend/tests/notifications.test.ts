import assert from "node:assert/strict";
import test from "node:test";
import { parseEngineStatus } from "../src/api.ts";
import {
  addEvents,
  initialTracker,
  loadNotifications,
  markAllRead,
  observe,
  unreadCount,
  type DownloadFact,
  type Observation,
  type TrackerState,
} from "../src/notifications.ts";
import type { FinishedFact } from "../src/health.ts";

type M = { id: string; loaded: boolean; loading?: boolean; error?: string };
const status = (
  o: { uptime?: number; models?: M[]; used?: number; load_error?: string } = {},
) =>
  parseEngineStatus({
    object: "yunshu.status",
    version: "t",
    state: "running",
    uptime_s: o.uptime ?? 1000,
    load_error: o.load_error ?? null,
    models: (o.models ?? []).map((m) => ({
      type: "LLM",
      pinned: false,
      loading: false,
      ...m,
    })),
    memory: { active_gb: o.used ?? 40, total_gb: 128 },
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
let clock = 1_000_000;
const obs = (
  over: Partial<Observation> & { s?: Parameters<typeof status>[0] } = {},
): Observation => ({
  at: (clock += 3000),
  phase: "online",
  status: over.status === undefined ? status(over.s) : over.status,
  finished: [],
  downloads: null,
  ...over,
});
const feed = (state: TrackerState, ...list: Observation[]) => {
  const all = [];
  for (const o of list) {
    const r = observe(state, o);
    state = r.state;
    all.push(...r.events);
  }
  return { state, events: all };
};
const kinds = (e: { kind: string }[]) => e.map((x) => x.kind);

test("the first observation is a baseline, not an event", () => {
  const r = feed(initialTracker(), obs({ s: { used: 120 } }));
  assert.deepEqual(r.events, []);
});
test("a model finishing its load is announced once, with its name", () => {
  const loading = obs({
    s: { models: [{ id: "/m/Qwen-9B", loaded: false, loading: true }] },
  });
  const loaded = obs({ s: { models: [{ id: "/m/Qwen-9B", loaded: true }] } });
  const r = feed(
    initialTracker(),
    obs({ s: { models: [{ id: "/m/Qwen-9B", loaded: false }] } }),
    loading,
    loaded,
    obs({ s: { models: [{ id: "/m/Qwen-9B", loaded: true }] } }),
  );
  assert.deepEqual(kinds(r.events), ["model.loaded"]);
  assert.equal(r.events[0].vars.model, "Qwen-9B");
});
test("a load that ends unloaded is a failure", () => {
  const r = feed(
    initialTracker(),
    obs({ s: { models: [{ id: "/m/X", loaded: false, loading: true }] } }),
    obs({ s: { models: [{ id: "/m/X", loaded: false }] } }),
  );
  assert.deepEqual(kinds(r.events), ["model.failed"]);
});
test("restart is detected from uptime and replaces the plain recovery", () => {
  const r = feed(
    initialTracker(),
    obs({ s: { uptime: 7200 } }),
    obs({ phase: "offline", status: null }),
    obs({ s: { uptime: 12 } }),
  );
  assert.deepEqual(kinds(r.events), ["engine.offline", "engine.restart"]);
});
test("recovery without a restart says the engine is back", () => {
  const r = feed(
    initialTracker(),
    obs({ s: { uptime: 100 } }),
    obs({ phase: "offline", status: null }),
    obs({ s: { uptime: 130 } }),
    obs({ s: { uptime: 133 } }),
  );
  assert.deepEqual(kinds(r.events), ["engine.offline", "engine.back"]);
});
test("memory warning has hysteresis and does not repeat", () => {
  const r = feed(
    initialTracker(),
    obs({ s: { used: 60 } }),
    obs({ s: { used: 108 } }),
    obs({ s: { used: 104 } }),
    obs({ s: { used: 110 } }),
    obs({ s: { used: 98 } }), // below 80 % but above the clear line (75 %)
    obs({ s: { used: 106 } }),
    obs({ s: { used: 70 } }),
    obs({ s: { used: 108 } }),
  );
  assert.deepEqual(kinds(r.events), ["memory.warn", "memory.warn"]);
  assert.equal(r.events[0].vars.pct, 84);
});
test("memory already high on the first look is remembered, not announced", () => {
  const r = feed(
    initialTracker(),
    obs({ s: { used: 120 } }),
    obs({ s: { used: 121 } }),
  );
  assert.deepEqual(r.events, []);
});
test("5xx burst fires once per cooldown", () => {
  const five = (at: number): FinishedFact => ({
    at,
    outcome: "error",
    statusCode: 503,
  });
  const a = obs();
  const b = obs();
  b.finished = [five(b.at - 1000), five(b.at - 2000), five(b.at - 3000)];
  const c = obs();
  c.finished = [five(c.at - 1000), five(c.at - 2000), five(c.at - 3000)];
  const r = feed(initialTracker(), a, b, c);
  assert.deepEqual(kinds(r.events), ["fivexx.burst"]);
  assert.equal(r.events[0].vars.n, 3);
  const two = obs();
  two.finished = [five(two.at - 1000), five(two.at - 2000)];
  assert.deepEqual(feed(initialTracker(), a, two).events, []);
});
test("downloads: only a job seen running is announced when it ends", () => {
  const dl = (state: string, id = "j1"): DownloadFact[] => [
    { id, repo: "org/Model-4bit", state, error: null },
  ];
  const r = feed(
    initialTracker(),
    obs({ downloads: [...dl("done", "old"), ...dl("running")] }),
    obs({ downloads: [...dl("done", "old"), ...dl("done")] }),
    obs({ downloads: [...dl("done", "old"), ...dl("done")] }),
  );
  assert.deepEqual(kinds(r.events), ["download.done"]);
  assert.equal(r.events[0].vars.model, "Model-4bit");
  const f = feed(
    initialTracker(),
    obs({ downloads: dl("queued") }),
    obs({ downloads: dl("failed") }),
  );
  assert.deepEqual(kinds(f.events), ["download.failed"]);
  const gone = feed(
    initialTracker(),
    obs({ downloads: dl("running") }),
    obs({ downloads: null }),
  );
  assert.deepEqual(gone.events, []);
});
test("the list is newest first, capped at 50, deduped, with read state", () => {
  const ev = (i: number) => ({
    id: "e" + i,
    kind: "engine.back" as const,
    tone: "info" as const,
    at: i,
    vars: {},
  });
  let list = addEvents([], [ev(1), ev(2)]);
  assert.deepEqual(
    list.map((n) => n.id),
    ["e2", "e1"],
  );
  list = addEvents(list, [ev(2), ev(3)]);
  assert.deepEqual(
    list.map((n) => n.id),
    ["e3", "e2", "e1"],
  );
  assert.equal(unreadCount(list), 3);
  list = markAllRead(list);
  assert.equal(unreadCount(list), 0);
  const big = addEvents(
    [],
    Array.from({ length: 80 }, (_, i) => ev(i)),
  );
  assert.equal(big.length, 50);
  assert.equal(big[0].id, "e79");
});
test("a corrupt saved list loads as empty or drops bad rows", () => {
  assert.deepEqual(loadNotifications(null), []);
  assert.deepEqual(loadNotifications("{oops"), []);
  assert.deepEqual(loadNotifications('{"a":1}'), []);
  const good = {
    id: "x",
    kind: "engine.back",
    tone: "info",
    at: 1,
    vars: {},
    read: false,
  };
  const bad = { id: "y", kind: "nope", at: 1, vars: {}, read: false };
  assert.equal(loadNotifications(JSON.stringify([good, bad])).length, 1);
});
