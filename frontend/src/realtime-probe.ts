import type { Connection } from "./api.ts";

/** What one Realtime connection test measured, all of it by this browser. */
export interface RealtimeProbe {
  state: "idle" | "connecting" | "open" | "closed" | "error";
  /** Socket opened, ms after the attempt started; null when it never opened. */
  openMs: number | null;
  /** First server event (the engine sends `session.created`), ms after the attempt started. */
  firstEventMs: number | null;
  firstEventType: string | null;
  /** The URL scheme actually used, so a plain ws on a LAN address is visible. */
  scheme: "ws" | "wss" | null;
  error: string | null;
}

export const IDLE_PROBE: RealtimeProbe = {
  state: "idle",
  openMs: null,
  firstEventMs: null,
  firstEventType: null,
  scheme: null,
  error: null,
};

/** `http(s)://host[/base]` to the realtime socket URL, and the subprotocols that carry the token. */
export function realtimeTarget(connection: Connection): {
  url: string;
  protocols: string[];
  scheme: "ws" | "wss";
} {
  const base = new URL(connection.baseUrl || globalThis.location?.origin);
  const scheme = base.protocol === "https:" ? "wss" : "ws";
  const path = base.pathname.replace(/\/+$/, "");
  const token = connection.token.trim();
  return {
    url: `${scheme}://${base.host}${path}/v1/realtime`,
    protocols: token
      ? ["realtime", `openai-insecure-api-key.${token}`]
      : ["realtime"],
    scheme,
  };
}

type SocketCtor = new (url: string, protocols?: string | string[]) => WebSocket;

/**
 * Opens the realtime socket, waits for the first server event and closes it. Nothing is sent, so
 * nothing runs on the GPU. `onUpdate` is called with every state change; the promise resolves
 * with the final probe and never rejects.
 */
export function probeRealtime(
  connection: Connection,
  onUpdate: (p: RealtimeProbe) => void,
  options: {
    timeoutMs?: number;
    WebSocketImpl?: SocketCtor;
    now?: () => number;
  } = {},
): Promise<RealtimeProbe> {
  const Impl = options.WebSocketImpl ?? globalThis.WebSocket;
  const now = options.now ?? (() => performance.now());
  const timeoutMs = options.timeoutMs ?? 8000;
  const target = realtimeTarget(connection);
  let probe: RealtimeProbe = {
    ...IDLE_PROBE,
    state: "connecting",
    scheme: target.scheme,
  };
  const set = (patch: Partial<RealtimeProbe>) => {
    probe = { ...probe, ...patch };
    onUpdate(probe);
  };
  return new Promise((resolve) => {
    const started = now();
    let done = false;
    let socket: WebSocket;
    const finish = (patch: Partial<RealtimeProbe>) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      try {
        socket?.close();
      } catch {
        /* already closed */
      }
      set(patch);
      resolve(probe);
    };
    const timer = setTimeout(
      () => finish({ state: "error", error: "timeout" }),
      timeoutMs,
    );
    set({});
    try {
      socket = new Impl(target.url, target.protocols);
    } catch (e) {
      finish({
        state: "error",
        error: e instanceof Error ? e.message : "open",
      });
      return;
    }
    socket.onopen = () => set({ state: "open", openMs: now() - started });
    socket.onmessage = (ev) => {
      let type: string | null = null;
      try {
        const parsed = JSON.parse(String(ev.data)) as { type?: unknown };
        type = typeof parsed.type === "string" ? parsed.type : null;
      } catch {
        /* not JSON: still an event */
      }
      finish({
        state: "closed",
        firstEventMs: now() - started,
        firstEventType: type,
      });
    };
    socket.onerror = () => finish({ state: "error", error: "socket" });
    socket.onclose = (ev) =>
      finish({
        state: probe.firstEventMs == null ? "error" : "closed",
        error: probe.firstEventMs == null ? `close ${ev.code}` : null,
      });
  });
}
