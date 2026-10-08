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
