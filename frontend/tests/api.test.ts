import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import {
  ApiError,
  cancelRequest,
  fetchStatus,
  getModel,
  loadModel,
  parseEngineStatus,
  requestJson,
  unloadModel,
  warmupModel,
  type Connection,
} from "../src/api.ts";

const connection: Connection = {
  baseUrl: "http://127.0.0.1:8000",
  token: "local-secret",
};
const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function validStatus() {
  return {
    object: "yunshu.status",
    version: "0.1-test",
    state: "running",
    uptime_s: 42,
    load_error: null,
    models: [
      {
        id: "org/model",
        type: "LLM",
        loaded: true,
        loading: false,
        pinned: true,
        idle_s: null,
        keep_alive_s: null,
        expires_in_s: null,
      },
    ],
    memory: {
      active_gb: 10.2,
      cache_gb: 2.1,
      peak_gb: 12.8,
      total_gb: 64,
      pressure: 0.159,
    },
    requests: {
      active: 1,
      queued: 0,
      prefill: 1,
      decode: 0,
      items: [
        {
          request_id: "request-1",
          elapsed_s: 4.1,
          phase: "prefill",
          prompt_tokens: 4096,
          cached_tokens: 1024,
          processed_tokens: 2500,
          percent: 48.2,
          tokens_per_second: 811.4,
          eta_s: 1.9,
          model: "org/model",
        },
      ],
    },
    last: {
      request_id: "request-0",
      prompt_tokens: 100,
      completion_tokens: 20,
      cached_tokens: 25,
      prefill_tps: 600,
      decode_tps: 35,
      ttft_ms: 420,
      t: 1_800_000_000,
    },
    gpu: {},
    throughput: {
      window_s: 60,
      requests: 3,
      prompt_tokens: 5500,
      completion_tokens: 240,
      live_decode_tps: null,
      mean_prefill_tps: 710.2,
      mean_decode_tps: 32.1,
    },
  };
}

async function withFetch(mock: typeof fetch, run: () => Promise<void>) {
  globalThis.fetch = mock;
  try {
    await run();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test("requestJson normalizes root and /v1 base URLs and sends the token only as a bearer header", async () => {
  const calls: { url: string; init?: RequestInit }[] = [];
  await withFetch(
    async (input, init) => {
      calls.push({ url: String(input), init });
      return jsonResponse({ ok: true });
    },
    async () => {
      assert.deepEqual(await requestJson(connection, "/yunshu/status"), {
        ok: true,
      });
      assert.deepEqual(
        await requestJson(
          { ...connection, baseUrl: "http://127.0.0.1:8000/v1/" },
          "/yunshu/status",
        ),
        { ok: true },
      );
    },
  );

  assert.equal(new URL(calls[0].url).pathname, "/v1/yunshu/status");
  assert.equal(new URL(calls[1].url).pathname, "/v1/yunshu/status");
  for (const call of calls) {
    const headers = new Headers(call.init?.headers);
    assert.equal(headers.get("Authorization"), "Bearer local-secret");
    assert.equal(new URL(call.url).search, "");
    assert.equal(new URL(call.url).hash, "");
  }
});

test("connection and path validation reject credentials, query, fragment and path escape", async () => {
  await assert.rejects(
    requestJson(
      { ...connection, baseUrl: "http://user:pass@localhost:8000" },
      "/yunshu/status",
    ),
    ApiError,
  );
  await assert.rejects(
    requestJson(
      { ...connection, baseUrl: "http://localhost:8000/?token=x" },
      "/yunshu/status",
    ),
    ApiError,
  );
  await assert.rejects(
    requestJson(
      { ...connection, baseUrl: "https://localhost:8000/#hash" },
      "/yunshu/status",
    ),
    ApiError,
  );
  await assert.rejects(
    requestJson(connection, "//attacker.test/path"),
    ApiError,
  );
  await assert.rejects(requestJson(connection, "/../admin"), ApiError);
});

test("requestJson parses backend detail, status, invalid JSON and timeout errors", async () => {
  await withFetch(
    async () => jsonResponse({ detail: "Model is in use" }, 409),
    async () => {
      await assert.rejects(
        requestJson(connection, "/models/load", {
          method: "POST",
          body: { model: "org/model" },
        }),
        (error: unknown) => {
          assert.ok(error instanceof ApiError);
          assert.equal(error.status, 409);
          assert.equal(error.publicMessage, "Model is in use");
          return true;
        },
      );
    },
  );

  await withFetch(
    async () => new Response("not json", { status: 200 }),
    async () => {
      await assert.rejects(
        requestJson(connection, "/yunshu/status"),
        (error: unknown) =>
          error instanceof ApiError && /invalid JSON/.test(error.message),
      );
    },
  );

  await withFetch(
    (_input, init) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener(
          "abort",
          () => reject(init.signal?.reason),
          { once: true },
        );
      }),
    async () => {
      await assert.rejects(
        requestJson(connection, "/yunshu/status", { timeoutMs: 5 }),
        (error: unknown) =>
          error instanceof ApiError && /timed out/.test(error.publicMessage),
      );
    },
  );
});

test("fetchStatus validates server payload and permits optional model fields", async () => {
  let headers: Headers | undefined;
  await withFetch(
    async (_input, init) => {
      headers = new Headers(init?.headers);
      return jsonResponse(validStatus());
    },
    async () => {
      const status = await fetchStatus(connection);
      assert.equal(status.models[0]?.id, "org/model");
      assert.equal(status.requests.items[0]?.phase, "prefill");
      assert.equal(status.requests.items[0]?.percent, 48.2);
      assert.equal(status.throughput.live_decode_tps, null);
    },
  );
  assert.equal(headers?.get("Authorization"), "Bearer local-secret");

  const malformed = validStatus();
  malformed.requests.active = Number.NaN;
  assert.throws(() => parseEngineStatus(malformed), ApiError);
});

test("operation helpers use the documented methods, encoded IDs, and JSON bodies", async () => {
  const calls: { url: string; method: string; body?: unknown }[] = [];
  await withFetch(
    async (input, init) => {
      calls.push({
        url: String(input),
        method: init?.method ?? "GET",
        body: init?.body ? JSON.parse(String(init.body)) : undefined,
      });
      return jsonResponse({ status: "ok" });
    },
    async () => {
      await getModel(connection, "org/model");
      await loadModel(connection, "org/model", { pin: true });
      await unloadModel(connection, "org/model");
      await warmupModel(connection, {
        model: "org/model",
        max_tokens: 1,
        keep_alive: "10m",
      });
      await cancelRequest(connection, "request/with:slash");
    },
  );
  assert.deepEqual(
    calls.map(({ url, method }) => [new URL(url).pathname, method]),
    [
      ["/v1/models/org%2Fmodel", "GET"],
      ["/v1/models/load", "POST"],
      ["/v1/models/unload/org%2Fmodel", "POST"],
      ["/v1/yunshu/warmup", "POST"],
      ["/v1/requests/request%2Fwith%3Aslash", "DELETE"],
    ],
  );
  assert.deepEqual(calls[1]?.body, { model: "org/model", pin: true });
  assert.deepEqual(calls[3]?.body, {
    model: "org/model",
    max_tokens: 1,
    keep_alive: "10m",
  });
});
