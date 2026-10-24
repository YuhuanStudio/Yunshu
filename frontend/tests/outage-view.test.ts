import assert from "node:assert/strict";
import test from "node:test";
import type { EngineList } from "../src/api.ts";
import { outageView } from "../src/outage-view.ts";

const engine = (port: number, state: string, selected = false) => ({
  url: `http://127.0.0.1:${port}`,
  port,
  model: "m",
  uptime_seconds: 5,
  state,
  selected,
  pinned: false,
});
const list = (data: EngineList["data"]): EngineList => ({
  mode: "auto",
  selected: data.find((e) => e.selected)?.url ?? null,
  pinned: null,
  waiting: false,
  data,
});
const base = {
  phase: "offline" as const,
  offlineSince: 1_000_000,
  updatedAt: 990_000,
  status: { uptime_s: 600 },
  usualS: 25,
};

test("a short outage with no live engine is a calm restart with was-up and usual time", () => {
  const v = outageView({ ...base, list: list([]), now: 1_012_000 });
  assert.equal(v.kind, "restarting");
  if (v.kind === "restarting") {
    assert.equal(Math.round(v.downS), 12);
    assert.equal(Math.round(v.wasUpS ?? -1), 610);
    assert.equal(v.usualS, 25);
  }
});

test("another running engine means switching, not an error", () => {
  const v = outageView({
    ...base,
    list: list([engine(18991, "running", true)]),
    now: 1_090_000,
  });
  assert.equal(v.kind, "switching");
});

test("only a sustained outage with no live engine escalates, and carries the engine list", () => {
  const dead = [engine(18990, "down")];
  assert.equal(
    outageView({ ...base, list: list(dead), now: 1_059_000 }).kind,
    "restarting",
  );
  const v = outageView({ ...base, list: list(dead), now: 1_061_000 });
  assert.equal(v.kind, "down");
  if (v.kind === "down") assert.equal(v.engines.length, 1);
});

test("fixed mode, no console list, or an online engine is never calm-wrapped", () => {
  assert.equal(
    outageView({ ...base, list: null, now: 1_005_000 }).kind,
    "none",
  );
  assert.equal(
    outageView({ ...base, phase: "online", list: list([]), now: 1_005_000 })
      .kind,
    "none",
  );
});

test("a measurement pause is calm and says why, however long it lasts", () => {
  const paused: EngineList = {
    ...list([]),
    pause: { reason: "speed job", until: null },
  };
  for (const now of [1_005_000, 1_900_000]) {
    const v = outageView({ ...base, list: paused, now });
    assert.equal(v.kind, "paused");
    if (v.kind === "paused") assert.equal(v.reason, "speed job");
  }
});
