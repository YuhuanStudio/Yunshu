import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { ApiError, type Connection } from "../src/api.ts";
import {
  copyModel,
  deleteModel,
  pullModel,
  requestServerJson,
} from "../src/management-api.ts";

const connection: Connection = {
  baseUrl: "http://127.0.0.1:8000/v1/",
  token: "secret-token",
};
const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

test("server management paths normalize root and /v1 bases and send auth only in a header", async () => {
  const urls: string[] = [];
  globalThis.fetch = async (input, init) => {
    urls.push(String(input));
    assert.equal(
      new Headers(init?.headers).get("Authorization"),
      "Bearer secret-token",
    );
    assert.equal(new URL(String(input)).search, "");
    assert.equal(new URL(String(input)).hash, "");
    return new Response(JSON.stringify({ ready: true }), { status: 200 });
  };

  assert.deepEqual(await requestServerJson(connection, "/debug/engine"), {
    ready: true,
  });
  assert.deepEqual(
    await requestServerJson(
      { ...connection, baseUrl: "http://127.0.0.1:8000" },
      "/debug/system",
    ),
    { ready: true },
  );
  assert.equal(new URL(urls[0]).pathname, "/debug/engine");
  assert.equal(new URL(urls[1]).pathname, "/debug/system");
});

test("Ollama pull/copy/delete send the exact backend methods and payloads and accept empty 200", async () => {
  const calls: { path: string; method?: string; body?: string }[] = [];
  globalThis.fetch = async (input, init) => {
    calls.push({
      path: new URL(String(input)).pathname,
      method: init?.method,
      body: init?.body as string | undefined,
    });
    return new Response(null, { status: 200 });
  };

  await pullModel(connection, "mlx-community/model name");
  await copyModel(connection, "mlx-community/model name", "team/alias");
  await deleteModel(connection, "team/alias");
  assert.deepEqual(calls, [
    {
      path: "/api/pull",
      method: "POST",
      body: JSON.stringify({
        model: "mlx-community/model name",
        stream: false,
      }),
    },
    {
      path: "/api/copy",
      method: "POST",
      body: JSON.stringify({
        source: "mlx-community/model name",
        destination: "team/alias",
      }),
    },
    {
      path: "/api/delete",
      method: "DELETE",
      body: JSON.stringify({ model: "team/alias" }),
    },
  ]);
});

test("debug responses require JSON while management responses permit empty success", async () => {
  globalThis.fetch = async () => new Response(null, { status: 200 });
  await assert.rejects(
    requestServerJson(connection, "/debug/engine"),
    (error: unknown) =>
      error instanceof ApiError &&
      error.status === 200 &&
      /empty response/.test(error.message),
  );
  await assert.doesNotReject(copyModel(connection, "source", "destination"));
});

test("server operations preserve auth and expose backend 401/409 errors", async () => {
  globalThis.fetch = async (_input, init) => {
    assert.equal(
      new Headers(init?.headers).get("Authorization"),
      "Bearer secret-token",
    );
    return new Response(JSON.stringify({ detail: "model is in use" }), {
      status: 409,
    });
  };
  await assert.rejects(
    deleteModel(connection, "org/model"),
    (error: unknown) =>
      error instanceof ApiError &&
      error.status === 409 &&
      error.publicMessage === "model is in use",
  );

  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: "unauthorized" }), { status: 401 });
  await assert.rejects(
    requestServerJson(connection, "/debug/requests"),
    (error: unknown) =>
      error instanceof ApiError &&
      error.status === 401 &&
      /Authentication failed/.test(error.publicMessage),
  );
});

test("unsafe server URLs and paths are rejected before fetch", async () => {
  let called = false;
  globalThis.fetch = async () => {
    called = true;
    return new Response("{}", { status: 200 });
  };
  await assert.rejects(
    requestServerJson(
      { ...connection, baseUrl: "http://user:pass@localhost" },
      "/debug/engine",
    ),
    ApiError,
  );
  await assert.rejects(
    requestServerJson(connection, "/debug/../api/delete"),
    ApiError,
  );
  await assert.rejects(
    requestServerJson(connection, "/debug/engine?token=x"),
    ApiError,
  );
  assert.equal(called, false);
});
