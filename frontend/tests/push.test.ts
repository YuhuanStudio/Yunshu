import assert from "node:assert/strict";
import test from "node:test";
import {
  createEventParser,
  pushLive,
  pushedPayload,
  resetPushForTests,
  subscribePush,
  type SseEvent,
} from "../src/push.ts";

const conn = { baseUrl: "http://x.test", token: "tok" };
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** A fetch whose body the test feeds by hand. */
function fakeStream() {
  const enc = new TextEncoder();
  let ctl!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({ start: (c) => (ctl = c) });
  const calls: { url: string; auth: string | null }[] = [];
  globalThis.fetch = (async (url: URL, init: RequestInit) => {
    calls.push({
      url: String(url),
      auth: new Headers(init.headers).get("Authorization"),
    });
    return new Response(body, {
      status: 200,
      headers: { "x-yunshu-console": "1" },
    });
  }) as typeof fetch;
  return {
    calls,
    send: (event: string, data: unknown) =>
      ctl.enqueue(
        enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
      ),
    end: () => ctl.close(),
  };
}

test("parser splits events, joins data lines, ignores keepalives", () => {
  const out: SseEvent[] = [];
  const feed = createEventParser((e) => out.push(e));
  feed(': keepalive\n\nevent: status\nid: 4\ndata: {"a"');
  feed(":1}\n\nevent: host\r\ndata: 1\r\ndata: 2\r\n\r\n");
  assert.deepEqual(out, [
    { event: "status", data: '{"a":1}' },
    { event: "host", data: "1\n2" },
  ]);
});

test("a healthy stream answers polls from the pushed payload and wakes listeners", async () => {
  resetPushForTests();
  const s = fakeStream();
  let woke = 0;
  const off = subscribePush(conn, "status", () => woke++);
  await sleep(80);
  assert.equal(s.calls.length, 1);
  assert.match(s.calls[0].url, /\/v1\/yunshu\/events\?topics=status$/);
  assert.equal(s.calls[0].auth, "Bearer tok");
  assert.equal(pushLive(conn, "status"), false); // no snapshot yet: poll for real
  s.send("hello", { boot_id: "B", topics: ["status"] });
  s.send("status", { v: 1, skipped: 0, d: { state: "running" } });
  await sleep(30);
  assert.equal(woke, 1);
  assert.equal(pushLive(conn, "status"), true);
  const hit = pushedPayload(conn, "/yunshu/status");
  assert.deepEqual(hit?.data, { state: "running" });
  assert.equal(hit?.headers.get("x-yunshu-console"), "1");
  assert.equal(pushedPayload(conn, "/yunshu/host"), undefined);
  off();
  resetPushForTests();
});

test("the fallback is immediate: a stream that ends is not live, polls resume", async () => {
  resetPushForTests();
  const s = fakeStream();
  subscribePush(conn, "status", () => undefined);
  await sleep(60);
  s.send("hello", { boot_id: "B" });
  s.send("status", { v: 1, skipped: 0, d: { n: 1 } });
  await sleep(30);
  assert.equal(pushLive(conn, "status"), true);
  s.end();
  await sleep(30);
  assert.equal(pushLive(conn, "status"), false);
  assert.equal(pushedPayload(conn, "/yunshu/status"), undefined);
  resetPushForTests();
});

test("a refusing server (old engine) leaves everything polling", async () => {
  resetPushForTests();
  globalThis.fetch = (async () =>
    new Response("nope", { status: 404 })) as typeof fetch;
  subscribePush(conn, "status", () => undefined);
  await sleep(80);
  assert.equal(pushLive(conn, "status"), false);
  resetPushForTests();
});

test("a new boot id drops what the old engine process held", async () => {
  resetPushForTests();
  const s = fakeStream();
  subscribePush(conn, "host", () => undefined);
  await sleep(60);
  s.send("hello", { boot_id: "A" });
  s.send("host", { v: 1, skipped: 0, d: { n: 1 } });
  await sleep(20);
  assert.ok(pushedPayload(conn, "/yunshu/host"));
  s.send("hello", { boot_id: "B" });
  await sleep(20);
  assert.equal(pushedPayload(conn, "/yunshu/host"), undefined);
  resetPushForTests();
});

test("topics joining within the debounce share one stream", async () => {
  resetPushForTests();
  const s = fakeStream();
  subscribePush(conn, "status", () => undefined);
  subscribePush(conn, "memory", () => undefined);
  subscribePush(conn, "host", () => undefined);
  await sleep(100);
  assert.equal(s.calls.length, 1);
  assert.match(s.calls[0].url, /topics=host%2Cmemory%2Cstatus$/);
  resetPushForTests();
});
