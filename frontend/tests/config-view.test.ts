import assert from "node:assert/strict";
import test from "node:test";
import {
  filterConfig,
  formatConfigValue,
  isChanged,
  parseConfig,
} from "../src/config-view.ts";

const payload = {
  settings: [
    {
      name: "YUNSHU_PORT",
      value: 9000,
      default: 8000,
      source: "env",
      stability: "stable",
      category: "server",
      description: "Port",
    },
    {
      name: "YUNSHU_LOG_LEVEL",
      value: "INFO",
      default: "INFO",
      source: "default",
      stability: "stable",
      category: "observability",
      description: "Level",
    },
    {
      name: "YUNSHU_AUTH_TOKEN",
      value: "hunter2",
      default: null,
      source: "env",
      stability: "stable",
      category: "auth",
      description: "Token",
    },
  ],
  warnings: ["w"],
  experimental_count: 2,
  experimental_max: 8,
};

test("parseConfig masks secrets even when the server did not", () => {
  const parsed = parseConfig(payload);
  assert.ok(parsed);
  const token = parsed.rows.find((r) => r.name === "YUNSHU_AUTH_TOKEN");
  assert.equal(token?.value, "***");
  assert.equal(parsed.experimentalCount, 2);
});

test("changed rows are the ones that differ from the default", () => {
  const rows = parseConfig(payload)!.rows;
  assert.deepEqual(
    rows.filter(isChanged).map((r) => r.name),
    ["YUNSHU_PORT", "YUNSHU_AUTH_TOKEN"],
  );
});

test("search and changed-only filters", () => {
  const rows = parseConfig(payload)!.rows;
  assert.equal(filterConfig(rows, "log", false).length, 1);
  assert.equal(filterConfig(rows, "", true).length, 2);
  assert.equal(filterConfig(rows, "observability", true).length, 0);
});

test("garbage payloads are rejected, values formatted", () => {
  assert.equal(parseConfig({ nope: 1 }), null);
  assert.equal(formatConfigValue(null), "未設定");
  assert.equal(formatConfigValue(false), "false");
});
