import assert from "node:assert/strict";
import test from "node:test";
import {
  normaliseOrigin,
  parsePatchResult,
  summarizePatch,
  validationErrors,
  AdminError,
  findReloadRoute,
} from "../src/admin-settings-api.ts";
import { dailySeries, quotaFill, cleanQuotas } from "../src/admin-keys-api.ts";
import { fieldKind, withDraft, type ConfigRow } from "../src/config-view.ts";
import {
  rememberedToken,
  setRememberedToken,
  isTokenRemembered,
} from "../src/token-store.ts";

const row = (o: Partial<ConfigRow>): ConfigRow => ({
  name: "YUNSHU_X",
  value: 1,
  default: 1,
  source: "default",
  stability: "stable",
  category: "",
  description: "",
  secret: false,
  type: "int",
  choices: [],
  applies: "live",
  minimum: null,
  ...o,
});

test("control kind follows type, choices and secrecy", () => {
  assert.equal(fieldKind(row({})), "number");
  assert.equal(fieldKind(row({ type: "bool" })), "bool");
  assert.equal(fieldKind(row({ type: "enum", choices: ["a"] })), "enum");
  assert.equal(fieldKind(row({ type: "str", secret: true })), "secret");
  assert.equal(fieldKind(row({ type: "path" })), "text");
});

test("a draft equal to the current value is not a change; null resets", () => {
  const r = row({});
  assert.deepEqual(withDraft({}, r, 1), {});
  assert.deepEqual(withDraft({}, r, 2), { YUNSHU_X: 2 });
  assert.deepEqual(withDraft({ YUNSHU_X: 2 }, r, 1), {});
  assert.deepEqual(withDraft({}, r, null), { YUNSHU_X: null });
  assert.deepEqual(withDraft({}, row({ type: "str", value: null }), ""), {});
});

test("patch results are summarised by when they apply", () => {
  const res = parsePatchResult({
    dry_run: false,
    restart_required: true,
    results: {
      A: { status: "applied", applies: "live", source: "file", reset: false },
      B: { status: "needs_reload", applies: "reload", source: "file" },
      C: { status: "needs_restart", applies: "restart", source: "file" },
      D: { status: "overridden", applies: "live", source: "env" },
    },
    restart: { available: false, cli: null, manual: "run it" },
  });
  assert.ok(res);
  const s = summarizePatch(res);
  assert.deepEqual(s.applied, ["A"]);
  assert.deepEqual(s.needsReload, ["B"]);
  assert.deepEqual(s.needsRestart, ["C"]);
  assert.deepEqual(s.overridden, [{ name: "D", source: "env" }]);
  assert.equal(res.restart?.manual, "run it");
});

test("422 per-name errors are surfaced", () => {
  const e = new AdminError(422, "invalid_settings", "x", {
    errors: { A: "bad", B: 3 },
  });
  assert.deepEqual(validationErrors(e), { A: "bad" });
  assert.deepEqual(validationErrors(new Error("x")), {});
});

test("origins are validated like the server", () => {
  assert.equal(
    normaliseOrigin("https://a.example.com/"),
    "https://a.example.com",
  );
  assert.equal(
    normaliseOrigin("http://localhost:3000"),
    "http://localhost:3000",
  );
  assert.equal(normaliseOrigin("*"), "*");
  assert.equal(normaliseOrigin("ftp://a"), null);
  assert.equal(normaliseOrigin("https://a.com/path"), null);
  assert.equal(normaliseOrigin("https://u:p@a.com"), null);
  assert.equal(normaliseOrigin("nope"), null);
});

test("reload route is detected only when the engine has it", () => {
  assert.equal(findReloadRoute({ paths: {} }), false);
  assert.equal(
    findReloadRoute({
      paths: { "/v1/yunshu/models/{model_id}/reload": { post: {} } },
    }),
    true,
  );
});

test("usage series is zero-filled per day and quota fill is null when unlimited", () => {
  const now = Date.UTC(2026, 9, 7, 12);
  const s = dailySeries(
    [
      {
        key: "k",
        name: "n",
        day: "2026-10-07",
        requests: 3,
        promptTokens: 5,
        completionTokens: 7,
        cachedTokens: 0,
        errors: 0,
      },
    ],
    "k",
    3,
    now,
  );
  assert.deepEqual(
    s.map((p) => [p.day, p.requests, p.tokens]),
    [
      ["2026-10-05", 0, 0],
      ["2026-10-06", 0, 0],
      ["2026-10-07", 3, 12],
    ],
  );
  assert.equal(quotaFill(5, null), null);
  assert.equal(quotaFill(5, 10), 0.5);
  assert.equal(
    cleanQuotas({
      requests_per_day: 0,
      tokens_per_day: 5,
      max_concurrent: null,
    }).requests_per_day,
    null,
  );
});

test("remembered token is opt-in, bound to its address, and storage failure is safe", () => {
  const mem = new Map<string, string>();
  (globalThis as { localStorage?: unknown }).localStorage = {
    getItem: (k: string) => mem.get(k) ?? null,
    setItem: (k: string, v: string) => void mem.set(k, v),
    removeItem: (k: string) => void mem.delete(k),
  };
  assert.equal(isTokenRemembered(), false);
  assert.equal(rememberedToken("http://h:1"), "");
  assert.equal(setRememberedToken("http://h:1/", "tok"), true);
  assert.equal(rememberedToken("http://h:1"), "tok");
  assert.equal(rememberedToken("http://other:1"), "");
  setRememberedToken("http://h:1", null);
  assert.equal(isTokenRemembered(), false);
  (globalThis as { localStorage?: unknown }).localStorage = {
    getItem: () => {
      throw new Error("blocked");
    },
    setItem: () => {
      throw new Error("blocked");
    },
    removeItem: () => {},
  };
  assert.equal(rememberedToken("http://h:1"), "");
  assert.equal(setRememberedToken("http://h:1", "x"), false);
});
