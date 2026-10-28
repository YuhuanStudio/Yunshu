import { connectionScope, useScopedState } from "./scoped-state.ts";
import { createSingleFlight } from "./single-flight.ts";
import { useCallback, useEffect, useRef, useState } from "react";
import { needsToken, type Connection } from "./api.ts";
import {
  UNSUPPORTED,
  isActive,
  listDownloads,
  listLocalModels,
  type DownloadList,
  type LocalInventory,
  type Unsupported,
} from "./admin-models-api.ts";
import { pushLive, subscribePush, type PushTopic } from "./push.ts";
import { getCache, type CacheOverview } from "./admin-cache-api.ts";

export type Polled<T> = {
  data: T | null;
  unsupported: boolean;
  error: string;
  /** Fetch now, outside the schedule. */
  refresh: () => void;
};

/**
 * Poll `load` while the tab is visible. A server without the route ("unsupported")
 * stops the polling for good; a failed poll keeps the last data and the error text.
 * `intervalMs` may depend on the data (fast while a download runs).
 */
export function usePolled<T>(
  connection: Connection,
  enabled: boolean,
  load: (
    connection: Connection,
    signal: AbortSignal,
  ) => Promise<T | Unsupported>,
  intervalMs: (data: T | null) => number,
  key = "",
  topic?: PushTopic,
): Polled<T> {
  const [state, setState] = useScopedState<Omit<Polled<T>, "refresh">>(
    connectionScope(connection),
    { data: null, unsupported: false, error: "" },
  );
  const kick = useRef<() => void>(() => undefined);
  const loadRef = useRef(load),
    intervalRef = useRef(intervalMs);
  loadRef.current = load;
  intervalRef.current = intervalMs;
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined,
      last: T | null = null,
      stop = false,
      // set by a push wake-up: that poll must run even while the stream is live
      viaPush = false;
    // A refresh asked for while a poll is in flight queues one more load (the poll in flight may
    // have been issued before the change the caller is refreshing for).
    const once = createSingleFlight(async () => {
      if (
        document.visibilityState === "visible" &&
        (viaPush || !topic || !pushLive(connection, topic))
      ) {
        try {
          const v = await loadRef.current(connection, controller.signal);
          if (controller.signal.aborted) return;
          if (v === UNSUPPORTED) {
            setState({ data: null, unsupported: true, error: "" });
            stop = true;
          } else {
            last = v;
            setState({ data: v, unsupported: false, error: "" });
          }
        } catch (e) {
          if (controller.signal.aborted) return;
          setState((s) => ({
            ...s,
            error: e instanceof Error ? e.message : String(e),
          }));
          if (needsToken(e)) stop = true;
        }
      }
    });
    const tick = async (fromPush = false) => {
      if (timer) clearTimeout(timer);
      viaPush = fromPush;
      await once();
      if (!stop && !controller.signal.aborted) {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => void tick(), intervalRef.current(last));
      }
    };
    kick.current = () => void tick();
    void tick();
    const off = topic
      ? subscribePush(connection, topic, () => void tick(true))
      : () => undefined;
    return () => {
      off();
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [connection.baseUrl, connection.token, enabled, key]);
  const refresh = useCallback(() => kick.current(), []);
  return { ...state, refresh };
}

// ── concrete polls ─────────────────────────────────────────────────────
/** Fast while anything is downloading (1.5 s), slow when idle (8 s). */
export const useDownloads = (c: Connection, enabled: boolean) =>
  usePolled<DownloadList>(
    c,
    enabled,
    (conn, signal) => listDownloads(conn, signal),
    (d) => (d && d.jobs.some(isActive) ? 1500 : 8000),
    "",
    "downloads",
  );

/** The disk scan is expensive server-side (cached 15 s), so poll lazily. */
export const useLocalInventory = (c: Connection, enabled: boolean) =>
  usePolled<LocalInventory>(
    c,
    enabled,
    (conn, signal) => listLocalModels(conn, false, signal),
    () => 20_000,
  );

export const useCache = (c: Connection, enabled: boolean) =>
  usePolled<CacheOverview>(
    c,
    enabled,
    (conn, signal) => getCache(conn, signal),
    () => 5000,
  );
