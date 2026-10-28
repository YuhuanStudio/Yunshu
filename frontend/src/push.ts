/**
 * The console's change feed: ONE `GET /v1/yunshu/events?topics=…` server-sent-event stream per
 * service (shared by every hook of the tab), replacing the per-hook timers.
 *
 * - `subscribePush(connection, topic, cb)` wakes a hook the moment its topic changes.
 * - `pushedPayload(connection, path)` lets `requestJson` answer a poll from the newest pushed
 *   payload while the stream is healthy, so existing parse code is unchanged and no request is sent.
 * - Anything wrong (old engine, 4xx, network, stale stream) makes `pushLive` false: the hooks'
 *   own timers poll again, exactly as before. The stream retries with backoff.
 * - Browsers allow six HTTP/1.1 connections per origin and every open stream holds one, so with
 *   Web Locks + BroadcastChannel the tabs of one browser elect a single leader that owns the stream
 *   and relays every event; the other tabs listen on the channel (and take over when the leader
 *   closes). Without those APIs each tab keeps its own stream.
 * - The stream closes after HIDDEN_CLOSE_MS with no visible tab; a visible tab reopens it (snapshot first).
 */
import { retryDelayMs } from "./backoff.ts";

export type PushTopic = "status" | "host" | "memory" | "requests" | "downloads";
type Conn = { baseUrl: string; token: string };

/** Polled paths a pushed payload may answer (same JSON as the endpoint). */
const PATH_TOPIC: Record<string, PushTopic> = {
  "/yunshu/status": "status",
  "/yunshu/host": "host",
  "/yunshu/memory": "memory",
  "/yunshu/downloads": "downloads",
  "/yunshu/requests/recent": "requests",
};
export const topicOfPath = (path: string): PushTopic | undefined =>
  PATH_TOPIC[path];

/** The engine re-sends every topic at least every 10 s; a stream silent for longer is not live. */
export const LIVE_WITHIN_MS = 13_000;
export const HIDDEN_CLOSE_MS = 20_000;
const OPEN_DEBOUNCE_MS = 30;
const MIN_HEALTHY_MS = 2_000;
const VIS_PING_MS = 5_000;
/** A refusing server (old engine, auth) is asked again only this rarely. */
const REFUSED_RETRY_MS = 60_000;

export interface PushStats {
  opens: number;
  events: number;
  failures: number;
  live: boolean;
}

export interface SseEvent {
  event: string;
  data: string;
}

/** Feed text chunks in; get complete SSE events out (comments / keepalives are dropped). */
export function createEventParser(onEvent: (e: SseEvent) => void) {
  let buffer = "";
  return (chunk: string) => {
    buffer += chunk.replace(/\r\n?/g, "\n");
    let at: number;
    while ((at = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, at);
      buffer = buffer.slice(at + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:"))
          data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) onEvent({ event, data: data.join("\n") });
    }
  };
}

function eventsUrl(connection: Conn, topics: string[]): URL {
  const url = new URL(connection.baseUrl.trim());
  const base = url.pathname.replace(/\/+$/, "");
  url.pathname = `${base.endsWith("/v1") ? base : `${base}/v1`}/yunshu/events`;
  url.hash = "";
  url.search = "";
  url.searchParams.set("topics", topics.join(","));
  return url;
}

type Listener = () => void;
interface Entry {
  data: unknown;
  at: number;
}

type Msg =
  | { t: "sync" }
  | { t: "vis" }
  | { t: "conn"; c: boolean; console?: boolean }
  | { t: "hello"; boot: string | null }
  | { t: "event"; topic: PushTopic; d: unknown }
  | {
      t: "snap";
      c: boolean;
      console: boolean;
      boot: string | null;
      entries: [string, unknown][];
    };

/** A short stable name for a channel / lock: the token never goes into either. */
function hashKey(key: string): string {
  let h = 5381;
  for (let i = 0; i < key.length; i++)
    h = ((h << 5) + h + key.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

class Feed {
  readonly listeners = new Map<PushTopic, Set<Listener>>();
  readonly entries = new Map<PushTopic, Entry>();
  headers = new Headers();
  connected = false;
  lastMessageAt = 0;
  bootId: string | null = null;
  stats: PushStats = { opens: 0, events: 0, failures: 0, live: false };
  private controller: AbortController | null = null;
  private running = false;
  private openTimer: ReturnType<typeof setTimeout> | undefined;
  private wake: (() => void) | null = null;
  private failures = 0;
  private opened = "";
  /** solo: this tab owns its stream (no election APIs); leader: owns the shared one; follower: listens. */
  role: "solo" | "leader" | "follower" = "solo";
  private chan: BroadcastChannel | null = null;
  private competing = false;
  private release: (() => void) | null = null;
  private lastVisibleAt = Date.now();
  private viaConsole = false;
  private tick: ReturnType<typeof setInterval> | undefined;

  readonly connection: Conn;
  constructor(connection: Conn) {
    this.connection = connection;
    const nav = globalThis.navigator as
      | { locks?: { request: (...a: unknown[]) => Promise<unknown> } }
      | undefined;
    if (nav?.locks && typeof BroadcastChannel !== "undefined") {
      this.role = "follower";
      this.chan = new BroadcastChannel(
        `yunshu-push:${hashKey(keyOf(connection))}`,
      );
      this.chan.onmessage = (m) => this.onMessage(m.data as Msg);
      this.chan.postMessage({ t: "sync" } satisfies Msg);
    }
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", this.onVisibility);
      addEventListener("pagehide", this.onPageHide);
    }
    this.tick = setInterval(() => this.housekeeping(), VIS_PING_MS);
  }

  private ownVisible = () =>
    typeof document === "undefined" || document.visibilityState === "visible";

  /** Is any tab of this browser looking? (the leader asks; followers ping while visible) */
  private anyVisible = () =>
    this.ownVisible() || Date.now() - this.lastVisibleAt < HIDDEN_CLOSE_MS;

  private housekeeping() {
    if (this.ownVisible()) {
      this.lastVisibleAt = Date.now();
      if (this.role === "follower")
        this.chan?.postMessage({ t: "vis" } satisfies Msg);
    }
    // the stream's owner closes it when nobody has looked for a while (the server stops sampling)
    if (this.role !== "follower" && this.connected && !this.anyVisible()) {
      this.controller?.abort();
      this.connected = false;
    }
  }

  private onVisibility = () => {
    if (this.ownVisible()) {
      this.lastVisibleAt = Date.now();
      if (this.role === "follower")
        this.chan?.postMessage({ t: "vis" } satisfies Msg);
      else if (!this.connected) this.kick();
    }
  };

  private onPageHide = () => {
    // a closing leader says so, so followers stop trusting a stream nobody feeds
    if (this.role === "leader")
      this.chan?.postMessage({ t: "conn", c: false } satisfies Msg);
  };

  private onMessage(m: Msg) {
    if (m.t === "sync") {
      if (this.role !== "follower" && this.chan && this.role === "leader")
        this.chan.postMessage({
          t: "snap",
          c: this.connected,
          console: this.viaConsole,
          boot: this.bootId,
          entries: [...this.entries].map(([k, v]) => [k, v.data]),
        } satisfies Msg);
      return;
    }
    if (this.role === "leader" || this.role === "solo") {
      if (m.t === "vis") {
        this.lastVisibleAt = Date.now();
        if (!this.connected) this.kick();
      }
      return;
    }
    // follower
    this.lastMessageAt = Date.now();
    if (m.t === "conn") this.setConn(m.c, m.console);
    else if (m.t === "hello") this.applyHello(m.boot);
    else if (m.t === "snap") {
      this.applyHello(m.boot);
      this.setConn(m.c, m.console);
      for (const [k, d] of m.entries) this.applyEvent(k as PushTopic, d);
    } else if (m.t === "event") this.applyEvent(m.topic, m.d);
  }

  private setConn(c: boolean, viaConsole?: boolean) {
    this.connected = c;
    this.stats.live = c;
    if (viaConsole !== undefined)
      this.headers = new Headers(viaConsole ? { "x-yunshu-console": "1" } : {});
  }

  /** Enter the election: the lock's holder owns the stream until this tab has no listener left. */
  private compete() {
    if (this.competing || !this.chan) return;
    this.competing = true;
    const locks = (
      globalThis.navigator as unknown as {
        locks: {
          request: (name: string, cb: () => Promise<void>) => Promise<unknown>;
        };
      }
    ).locks;
    void locks
      .request(`yunshu-push:${hashKey(keyOf(this.connection))}`, async () => {
        this.role = "leader";
        this.chan?.postMessage({ t: "conn", c: false } satisfies Msg);
        this.kick();
        await new Promise<void>((r) => (this.release = r));
        this.release = null;
        this.controller?.abort();
        this.connected = false;
        this.role = "follower";
      })
      .finally(() => {
        this.competing = false;
        if (this.topics().length) this.compete();
      });
  }

  /**
   * One stream carries every topic as soon as any hook listens: hooks mount a page at a time, and
   * reopening the stream for each new topic cost three connections on the overview alone. The extra
   * samplers are cheap (host and memory every 2 s, requests 0.5 s) and stop when the tab goes away.
   */
  private topics(): PushTopic[] {
    const any = [...this.listeners.values()].some((set) => set.size > 0);
    return any ? (Object.keys(TOPICS).sort() as PushTopic[]) : [];
  }

  live(topic: PushTopic): boolean {
    return (
      this.connected &&
      this.entries.has(topic) &&
      Date.now() - this.lastMessageAt < LIVE_WITHIN_MS
    );
  }

  subscribe(topic: PushTopic, cb: Listener): () => void {
    let set = this.listeners.get(topic);
    if (!set) this.listeners.set(topic, (set = new Set()));
    set.add(cb);
    this.scheduleOpen();
    return () => {
      set.delete(cb);
      if (!set.size) this.listeners.delete(topic);
      this.scheduleOpen();
    };
  }

  /** (Re)open when the wanted topic set differs from the open stream's; close when nobody listens. */
  private scheduleOpen() {
    clearTimeout(this.openTimer);
    this.openTimer = setTimeout(() => {
      const want = this.topics().join(",");
      if (this.chan) {
        if (want) this.compete();
        else this.release?.();
        if (this.role !== "leader") return;
      }
      if (want === this.opened && (this.connected || this.running)) return;
      this.controller?.abort();
      if (!want) {
        this.opened = "";
        this.connected = false;
        return;
      }
      this.kick();
    }, OPEN_DEBOUNCE_MS);
  }

  private kick() {
    if (this.role === "follower") return;
    if (this.wake) this.wake();
    else if (!this.running) void this.run();
  }

  private isFollower = () => this.role === "follower";

  private async run() {
    this.running = true;
    try {
      while (this.topics().length && !this.isFollower()) {
        if (!this.anyVisible()) {
          await new Promise<void>((r) => (this.wake = r));
          this.wake = null;
          continue;
        }
        const topics = this.topics();
        this.opened = topics.join(",");
        let refused = false;
        const openedAt = Date.now();
        const eventsBefore = this.stats.events;
        try {
          await this.stream(topics);
          // A stream that gave nothing, or died within seconds, is a failure (a mock or a proxy that
          // answers 200 with a plain body must not turn this loop into a reconnect spin).
          if (
            this.stats.events === eventsBefore ||
            Date.now() - openedAt < MIN_HEALTHY_MS
          )
            this.failures++;
          else this.failures = 0;
        } catch (e) {
          this.stats.failures++;
          this.failures++;
          const status = (e as { status?: number })?.status;
          refused = status != null && status >= 400 && status < 500;
          if ((e as { name?: string })?.name === "AbortError")
            this.failures = Math.max(0, this.failures - 1);
        }
        this.connected = false;
        this.stats.live = false;
        this.chan?.postMessage({ t: "conn", c: false } satisfies Msg);
        if (this.opened !== this.topics().join(",")) continue; // topic set changed: reopen now
        const delay = refused
          ? REFUSED_RETRY_MS
          : this.failures
            ? retryDelayMs(this.failures)
            : 250;
        await new Promise<void>((r) => {
          const t = setTimeout(r, delay);
          this.wake = () => {
            clearTimeout(t);
            r();
          };
        });
        this.wake = null;
      }
    } finally {
      this.running = false;
      this.wake = null;
    }
  }

  private async stream(topics: PushTopic[]) {
    const controller = new AbortController();
    this.controller = controller;
    const headers = new Headers({ Accept: "text/event-stream" });
    if (this.connection.token.trim())
      headers.set("Authorization", `Bearer ${this.connection.token.trim()}`);
    this.stats.opens++;
    const response = await fetch(eventsUrl(this.connection, topics), {
      headers,
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      throw Object.assign(new Error(`HTTP ${response.status}`), {
        status: response.status,
      });
    }
    this.headers = response.headers;
    this.connected = true;
    this.lastMessageAt = Date.now();
    this.stats.live = true;
    this.viaConsole = response.headers.get("x-yunshu-console") === "1";
    this.chan?.postMessage({
      t: "conn",
      c: true,
      console: this.viaConsole,
    } satisfies Msg);
    const parse = createEventParser((e) => this.onEvent(e));
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) return;
        this.lastMessageAt = Date.now();
        parse(decoder.decode(value, { stream: true }));
      }
    } finally {
      reader.cancel().catch(() => undefined);
    }
  }

  private onEvent(e: SseEvent) {
    let body: { boot_id?: string; d?: unknown };
    try {
      body = JSON.parse(e.data);
    } catch {
      return;
    }
    this.stats.events++;
    if (e.event === "hello") {
      this.applyHello(body.boot_id ?? null);
      this.chan?.postMessage({
        t: "hello",
        boot: body.boot_id ?? null,
      } satisfies Msg);
      return;
    }
    if (!(e.event in TOPICS)) return;
    const topic = e.event as PushTopic;
    this.applyEvent(topic, body.d);
    this.chan?.postMessage({ t: "event", topic, d: body.d } satisfies Msg);
  }

  /** A new engine process starts every sequence over: nothing held from the old one is valid. */
  private applyHello(boot: string | null) {
    if (this.bootId && boot !== this.bootId) this.entries.clear();
    this.bootId = boot;
  }

  private applyEvent(topic: PushTopic, d: unknown) {
    this.entries.set(topic, { data: d, at: Date.now() });
    for (const cb of [...(this.listeners.get(topic) ?? [])]) {
      try {
        cb();
      } catch {
        /* a listener must not break the stream */
      }
    }
  }

  dispose() {
    this.listeners.clear();
    this.controller?.abort();
    this.wake?.();
    this.release?.();
    clearTimeout(this.openTimer);
    clearInterval(this.tick);
    const chan = this.chan;
    this.chan = null; // later posts (a lock handover finishing) go nowhere
    chan?.close();
    if (typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", this.onVisibility);
      removeEventListener("pagehide", this.onPageHide);
    }
  }
}

const TOPICS: Record<PushTopic, true> = {
  status: true,
  host: true,
  memory: true,
  requests: true,
  downloads: true,
};

const feeds = new Map<string, Feed>();
const keyOf = (c: Conn) => `${c.baseUrl.trim()}\n${c.token.trim()}`;
function feedFor(c: Conn): Feed {
  const key = keyOf(c);
  let f = feeds.get(key);
  if (!f)
    feeds.set(key, (f = new Feed({ baseUrl: c.baseUrl, token: c.token })));
  return f;
}

/** Disabled where the page runs without fetch streaming (tests set this to opt out). */
let disabled = false;
export const pushEnabled = () => !disabled;
export const setPushEnabled = (on: boolean) => {
  disabled = !on;
};

export function subscribePush(
  connection: Conn,
  topic: PushTopic,
  cb: () => void,
): () => void {
  if (disabled || typeof fetch === "undefined") return () => undefined;
  return feedFor(connection).subscribe(topic, cb);
}

/** True while the stream is healthy and has delivered this topic: the hook's timer may stay idle. */
export function pushLive(connection: Conn, topic: PushTopic): boolean {
  if (disabled) return false;
  return feeds.get(keyOf(connection))?.live(topic) ?? false;
}

/** The newest pushed payload for a polled path, or undefined (poll for real). */
export function pushedPayload(
  connection: Conn,
  path: string,
): { data: unknown; headers: Headers } | undefined {
  const topic = topicOfPath(path);
  if (!topic || disabled) return undefined;
  const feed = feeds.get(keyOf(connection));
  if (!feed?.live(topic)) return undefined;
  const entry = feed.entries.get(topic);
  return entry ? { data: entry.data, headers: feed.headers } : undefined;
}

export function pushStats(connection: Conn): PushStats | null {
  const f = feeds.get(keyOf(connection));
  return f ? { ...f.stats, live: f.connected } : null;
}

/** For the browser tests and the measurement scripts: every feed's counters on `window`. */
export function exposePushStats() {
  (globalThis as { __yunshuPush?: () => unknown }).__yunshuPush = () =>
    [...feeds.values()].map((f) => ({
      topics: [...f.listeners.keys()],
      ...f.stats,
      live: f.connected,
    }));
}

/** Two tabs of one browser are two Feeds on one channel: tests build them directly. */
export const newFeedForTests = (c: Conn) => new Feed(c);

export function resetPushForTests() {
  for (const f of feeds.values()) f.dispose();
  feeds.clear();
}
