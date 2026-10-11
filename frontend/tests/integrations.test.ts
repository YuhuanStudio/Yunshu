import assert from "node:assert/strict";
import { test } from "node:test";
import { buildIntegrations, serviceRoot } from "../src/integrations.ts";

test("every integration carries the base URL and model", () => {
  const rows = buildIntegrations("http://10.0.0.2:8000/", "Qwen3.8-27B");
  assert.deepEqual(
    rows.map((r) => r.id),
    ["claude-code", "codex", "opencode", "openai", "anthropic", "curl"],
  );
  for (const row of rows) {
    assert.ok(row.code.includes("http://10.0.0.2:8000"), row.id);
    assert.ok(row.code.includes("Qwen3.8-27B"), row.id);
    assert.ok(!row.code.includes("8000//"), row.id);
  }
});

test("Anthropic endpoints omit /v1 and OpenAI endpoints include it", () => {
  const by = Object.fromEntries(
    buildIntegrations("http://h:8000/v1", "m").map((r) => [r.id, r.code]),
  );
  assert.ok(by["claude-code"].includes("ANTHROPIC_BASE_URL=http://h:8000\n"));
  assert.ok(by.anthropic.includes('base_url="http://h:8000"'));
  assert.ok(by.codex.includes('base_url = "http://h:8000/v1"'));
  assert.ok(by.openai.includes('"http://h:8000/v1"'));
  assert.equal(serviceRoot("http://h:8000/v1/"), "http://h:8000");
});

test("tokens are referenced by environment variable, never embedded", () => {
  for (const row of buildIntegrations("http://h:8000", "")) {
    assert.ok(row.code.includes("local") || row.code.includes("YUNSHU_"));
  }
});
