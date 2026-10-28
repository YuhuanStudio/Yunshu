import assert from "node:assert/strict";
import test from "node:test";
import {
  createHistoryTracker,
  getRecent,
  historyStale,
} from "../src/recentRequests.ts";

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

test("the recorded history is read again after an outage or an emptied ring, not on a steady poll", () => {
  assert.equal(historyStale(null, 40), false);
  assert.equal(historyStale({ failed: false, ringRows: 40 }, 40), false);
  assert.equal(historyStale({ failed: false, ringRows: 40 }, 45), false);
  assert.equal(historyStale({ failed: true, ringRows: 40 }, 40), true);
  assert.equal(historyStale({ failed: false, ringRows: 40 }, 0), true);
});

test("a first poll that fails while the first history load is still on its way does not read it twice", () => {
  const t = createHistoryTracker();
  t.pollFailed(); // the engine is away when the page opens
  assert.equal(t.pollOk(40), false, "the initial load already covers this");
  t.loaded();
  assert.equal(t.pollOk(40), false);
});

test("an outage after the snapshot, a shrunk ring or a failed load each ask for a re-read", () => {
  const t = createHistoryTracker();
  t.loaded();
  assert.equal(t.pollOk(40), false);
  t.pollFailed();
  assert.equal(t.pollOk(40), true);
  t.started();
  t.loaded();
  assert.equal(t.pollOk(0), true, "the ring was emptied");
  const f = createHistoryTracker();
  f.loadFailed();
  assert.equal(f.pollOk(5), true);
});

test("a history load finishing during an outage covers it", () => {
  const t = createHistoryTracker();
  t.loaded();
  t.pollOk(10);
  t.pollFailed();
  t.started();
  t.loaded();
  assert.equal(t.pollOk(10), false);
});
