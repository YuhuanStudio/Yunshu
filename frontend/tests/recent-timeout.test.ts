import assert from "node:assert/strict";
import test from "node:test";
import { getRecent } from "../src/recentRequests.ts";

test("a /requests/recent response that never arrives is abandoned after the timeout", async () => {
  const real = globalThis.fetch;
  const keepAlive = setTimeout(() => {}, 1000); // AbortSignal.timeout does not hold the loop open
  // a wedged engine: the request settles only when its signal aborts
  globalThis.fetch = ((_url: unknown, init?: RequestInit) =>
    new Promise((_, reject) =>
      init?.signal?.addEventListener("abort", () =>
        reject(init.signal?.reason),
      ),
    )) as typeof fetch;
  try {
    await assert.rejects(
      getRecent(
        { baseUrl: "http://localhost:8000", token: "" },
        new AbortController().signal,
        30,
      ),
    );
  } finally {
    globalThis.fetch = real;
    clearTimeout(keepAlive);
  }
});
