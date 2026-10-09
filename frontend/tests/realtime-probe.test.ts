import assert from "node:assert/strict";
import test from "node:test";
import { probeRealtime, realtimeTarget } from "../src/realtime-probe.ts";

class FakeSocket {
  static script: (s: FakeSocket) => void = () => {};
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  closed = false;
  url: string;
  protocols?: string | string[];
  constructor(url: string, protocols?: string | string[]) {
    this.url = url;
    this.protocols = protocols;
    queueMicrotask(() => FakeSocket.script(this));
  }
  close() {
    this.closed = true;
  }
}
const conn = { baseUrl: "http://127.0.0.1:3984", token: "k" };
const ctor = FakeSocket as unknown as new (
  u: string,
  p?: string | string[],
) => WebSocket;

test("target: scheme follows the page, token travels as a subprotocol", () => {
  const a = realtimeTarget(conn);
  assert.equal(a.url, "ws://127.0.0.1:3984/v1/realtime");
  assert.deepEqual(a.protocols, ["realtime", "openai-insecure-api-key.k"]);
  const b = realtimeTarget({ baseUrl: "https://h.example/yunshu/", token: "" });
  assert.equal(b.url, "wss://h.example/yunshu/v1/realtime");
  assert.deepEqual(b.protocols, ["realtime"]);
});

test("probe: opens, first event measured, socket closed", async () => {
  let t = 0;
  FakeSocket.script = (s) => {
    t = 40;
    s.onopen?.();
    t = 95;
    s.onmessage?.({ data: JSON.stringify({ type: "session.created" }) });
  };
  const seen: string[] = [];
  const p = await probeRealtime(conn, (x) => seen.push(x.state), {
    WebSocketImpl: ctor,
    now: () => t,
  });
  assert.equal(p.openMs, 40);
  assert.equal(p.firstEventMs, 95);
  assert.equal(p.firstEventType, "session.created");
  assert.equal(p.state, "closed");
  assert.ok(seen.includes("open"));
});

test("probe: a refused socket is an error, never a 0 ms success", async () => {
  FakeSocket.script = (s) => s.onclose?.({ code: 1006 });
  const p = await probeRealtime(conn, () => {}, {
    WebSocketImpl: ctor,
    now: () => 0,
  });
  assert.equal(p.state, "error");
  assert.equal(p.openMs, null);
  assert.equal(p.firstEventMs, null);
  assert.match(p.error ?? "", /1006/);
});

test("probe: no event in time times out", async () => {
  FakeSocket.script = (s) => s.onopen?.();
  const p = await probeRealtime(conn, () => {}, {
    WebSocketImpl: ctor,
    timeoutMs: 20,
  });
  assert.equal(p.state, "error");
  assert.equal(p.error, "timeout");
  assert.notEqual(p.openMs, null);
});
