import assert from "node:assert/strict";
import { test } from "node:test";
import { parseHostTelemetry } from "../src/host-api.ts";

test("the documented sample parses; missing counters stay null, never 0", () => {
  const t = parseHostTelemetry({
    telemetry: {
      state: "partial",
      sampled_at: 1791400000.0,
      interval_s: 1.0,
      watts: { cpu: 5.0, gpu: 30.0, ane: 0.0, dram: null, package: 37.0 },
      gpu: { frequency_mhz: 900.0, active_ratio: 0.75 },
      temperature: {
        state: "ok",
        die_max_c: 70.0,
        die_mean_c: null,
        battery_c: null,
      },
      reasons: { "watts.dram": "no counter", bad: 3 },
    },
  })!;
  assert.equal(t.state, "partial");
  assert.equal(t.watts.ane, 0); // a reported zero is data
  assert.equal(t.watts.dram, null);
  assert.equal(t.temperature.dieMeanC, null);
  assert.deepEqual(t.reasons, { "watts.dram": "no counter" });
});

test("an engine without the field is unsupported (null); junk and unknown states degrade", () => {
  assert.equal(parseHostTelemetry({ thermal: {} }), null);
  assert.equal(parseHostTelemetry("x"), null);
  const t = parseHostTelemetry({
    telemetry: { state: "weird", reason: "disabled", watts: { gpu: "NaN" } },
  })!;
  assert.equal(t.state, "unknown");
  assert.equal(t.reason, "disabled");
  assert.equal(t.watts.gpu, null);
});

import {
  appendSample,
  parseHostSystem,
  type HostSample,
} from "../src/host-api.ts";

const sample = (at: number, state: "ok" | "unknown" = "ok") =>
  parseHostTelemetry({
    telemetry: {
      state,
      sampled_at: at,
      watts: { gpu: 10 + at, package: 20 },
      gpu: { frequency_mhz: 900, active_ratio: 0.5 },
      temperature: { state: "ok", die_max_c: 60 },
    },
  })!;

test("history dedupes by the engine's sampled_at, drops unknown samples and stays bounded", () => {
  let h: HostSample[] = [];
  h = appendSample(h, sample(1));
  h = appendSample(h, sample(1));
  h = appendSample(h, sample(0));
  assert.equal(h.length, 1);
  h = appendSample(h, sample(2, "unknown"));
  assert.equal(h.length, 1);
  for (let i = 2; i < 200; i++) h = appendSample(h, sample(i));
  assert.equal(h.length, 60);
  assert.equal(h.at(-1)!.at, 199);
});

test("OS readings: unknown stays unknown, a pmset limit is throttled with its percent", () => {
  const s = parseHostSystem({
    thermal: { state: "throttled", cpu_speed_limit_percent: 70 },
    memory_pressure: { state: "warning", level: 2 },
    power: { state: "battery", battery_percent: 55 },
  });
  assert.equal(s.thermal.state, "throttled");
  assert.equal(s.thermal.speedLimitPercent, 70);
  assert.equal(s.pressure.state, "warning");
  assert.equal(s.power.source, "battery");
  const u = parseHostSystem({ thermal: { state: "unknown", reason: "x" } });
  assert.equal(u.thermal.state, "unknown");
  assert.equal(u.thermal.speedLimitPercent, null);
  assert.equal(u.pressure.state, "unknown");
});
