import assert from "node:assert/strict";
import test from "node:test";
import { ApiError, requestJson } from "../src/api.ts";
import { requestServerJson } from "../src/management-api.ts";
import { detailText } from "../src/errors.ts";

const connection = { baseUrl: "http://127.0.0.1:8000", token: "" };
const cases: [number, RegExp][] = [
  [401, /驗證失敗/],
  [403, /權限/],
  [404, /找不到/],
  [409, /衝突/],
  [429, /頻繁|忙碌/],
  [500, /引擎內部錯誤/],
  [503, /引擎內部錯誤/],
  [502, /無法連線到引擎/],
  [504, /無法連線到引擎/],
];

async function withFetch(
  fn: typeof fetch,
  run: () => Promise<void>,
): Promise<void> {
  const original = globalThis.fetch;
  globalThis.fetch = fn;
  try {
    await run();
  } finally {
    globalThis.fetch = original;
  }
}

for (const [status, pattern] of cases) {
  test(`HTTP ${status} becomes zh-TW copy and keeps the raw detail`, async () => {
    await withFetch(
      async () =>
        new Response(JSON.stringify({ detail: "raw english detail" }), {
          status,
        }),
      async () => {
        for (const call of [
          () => requestJson(connection, "/yunshu/status"),
          () => requestServerJson(connection, "/debug/system"),
        ]) {
          await assert.rejects(call(), (error: unknown) => {
            assert.ok(error instanceof ApiError);
            assert.match(error.publicMessage, pattern);
            assert.doesNotMatch(error.publicMessage, /[A-Za-z]{6,}/);
            assert.match(detailText(error) ?? "", /raw english detail/);
            assert.match(detailText(error) ?? "", new RegExp(`HTTP ${status}`));
            return true;
          });
        }
      },
    );
  });
}

test("5xx is an engine error, never 'cannot connect'", async () => {
  await withFetch(
    async () => new Response("boom", { status: 500 }),
    async () => {
      await assert.rejects(
        requestJson(connection, "/yunshu/status"),
        (error: unknown) =>
          error instanceof ApiError && !/無法連線/.test(error.publicMessage),
      );
    },
  );
});

test("network failure, bad JSON and empty body are zh-TW", async () => {
  await withFetch(
    async () => {
      throw new TypeError("Failed to fetch");
    },
    async () => {
      await assert.rejects(
        requestJson(connection, "/yunshu/status"),
        (error: unknown) =>
          error instanceof ApiError &&
          /無法連線/.test(error.publicMessage) &&
          /Failed to fetch/.test(detailText(error) ?? ""),
      );
    },
  );
  await withFetch(
    async () => new Response("<html>", { status: 200 }),
    async () => {
      await assert.rejects(
        requestJson(connection, "/yunshu/status"),
        (e: unknown) => e instanceof ApiError && /JSON/.test(e.publicMessage),
      );
    },
  );
  await withFetch(
    async () => new Response("", { status: 200 }),
    async () => {
      await assert.rejects(
        requestJson(connection, "/yunshu/status"),
        (e: unknown) => e instanceof ApiError && /空/.test(e.publicMessage),
      );
    },
  );
});

test("a proxy 502/504 reads as unreachable, a real 5xx stays an engine fault", async () => {
  const { offlineCause } = await import("../src/errors.ts");
  assert.equal(offlineCause("offline", 502).short, "無法連線");
  assert.equal(offlineCause("offline", 504).short, "無法連線");
  assert.notEqual(offlineCause("offline", 500).short, "無法連線");
});
