import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  fetchStatus,
  type Connection,
  type EngineStatus,
} from "./api";

export type EngineConnectionPhase =
  "connecting" | "online" | "offline" | "unauthorized";
export interface EngineHistoryPoint {
  at: number;
  status: EngineStatus;
}

export interface UseEngineResult {
  status: EngineStatus | null;
  phase: EngineConnectionPhase;
  error: string | null;
  updatedAt: number | null;
  history: readonly EngineHistoryPoint[];
  refresh: () => Promise<void>;
  polling: boolean;
  setPolling: (enabled: boolean) => void;
}

const POLL_INTERVAL_MS = 3_000;
const MAX_HISTORY_POINTS = 1_200;

interface EngineViewState {
  connectionKey: symbol;
  status: EngineStatus | null;
  phase: EngineConnectionPhase;
  error: string | null;
  updatedAt: number | null;
  history: EngineHistoryPoint[];
}

function initialState(connectionKey: symbol): EngineViewState {
  return {
    connectionKey,
    status: null,
    phase: "connecting",
    error: null,
    updatedAt: null,
    history: [],
  };
}

/**
 * Read-only engine status and bounded history. Operations such as load, warmup,
 * and cancellation remain explicit host actions through `api.ts`.
 */
export function useEngine(connection: Connection): UseEngineResult {
  // Use an opaque identity so bearer credentials aren't copied into state.
  const connectionKey = useMemo(
    () => Symbol("engine-connection"),
    [connection.baseUrl, connection.token],
  );
  const apiConnection = useMemo<Connection>(
    () => ({ baseUrl: connection.baseUrl, token: connection.token }),
    [connectionKey],
  );
  const connectionKeyRef = useRef(connectionKey);
  connectionKeyRef.current = connectionKey;

  const [state, setState] = useState<EngineViewState>(() =>
    initialState(connectionKey),
  );
  const stateForConnection =
    state.connectionKey === connectionKey ? state : initialState(connectionKey);
  const [polling, setPolling] = useState(true);
  const generationRef = useRef(0);
  const activeControllerRef = useRef<AbortController | null>(null);
  const inFlightRef = useRef<{
    connectionKey: symbol;
    promise: Promise<void>;
  } | null>(null);

  useEffect(() => {
    const generation = ++generationRef.current;
    activeControllerRef.current?.abort();
    activeControllerRef.current = null;
    inFlightRef.current = null;
    setState(initialState(connectionKey));
    setPolling(true);
    return () => {
      if (generationRef.current === generation) generationRef.current += 1;
      activeControllerRef.current?.abort();
      activeControllerRef.current = null;
      inFlightRef.current = null;
    };
  }, [connectionKey]);

  const refresh = useCallback((): Promise<void> => {
    const existing = inFlightRef.current;
    if (existing?.connectionKey === connectionKey) return existing.promise;

    const generation = generationRef.current;
    const controller = new AbortController();
    activeControllerRef.current?.abort();
    activeControllerRef.current = controller;

    let promise: Promise<void> = Promise.resolve();
    promise = (async () => {
      try {
        const status = await fetchStatus(apiConnection, {
          signal: controller.signal,
        });
        if (
          controller.signal.aborted ||
          generation !== generationRef.current ||
          connectionKey !== connectionKeyRef.current
        )
          return;
        const at = Date.now();
        setState((current) => {
          const previous =
            current.connectionKey === connectionKey
              ? current
              : initialState(connectionKey);
          const restarted =
            previous.status !== null &&
            status.uptime_s < previous.status.uptime_s;
          const samples = restarted ? [] : previous.history;
          return {
            connectionKey,
            status,
            phase: "online",
            error: null,
            updatedAt: at,
            history: [...samples, { at, status }].slice(-MAX_HISTORY_POINTS),
          };
        });
      } catch (error) {
        if (
          controller.signal.aborted ||
          generation !== generationRef.current ||
          connectionKey !== connectionKeyRef.current
        )
          return;
        const apiError = error instanceof ApiError ? error : null;
        setState((current) => {
          const previous =
            current.connectionKey === connectionKey
              ? current
              : initialState(connectionKey);
          return {
            ...previous,
            connectionKey,
            phase:
              apiError?.status === 401 || apiError?.status === 403
                ? "unauthorized"
                : "offline",
            error:
              apiError?.publicMessage ?? "Could not reach the Yunshu service.",
          };
        });
      } finally {
        if (activeControllerRef.current === controller)
          activeControllerRef.current = null;
        if (inFlightRef.current?.promise === promise)
          inFlightRef.current = null;
      }
    })();

    inFlightRef.current = { connectionKey, promise };
    return promise;
  }, [apiConnection, connectionKey]);

  useEffect(() => {
    if (!polling) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const clearTimer = () => {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
    };
    const schedule = () => {
      clearTimer();
      if (!disposed && document.visibilityState === "visible") {
        timer = setTimeout(() => {
          void poll();
        }, POLL_INTERVAL_MS);
      }
    };
    const poll = async () => {
      if (disposed || document.visibilityState !== "visible") return;
      await refresh();
      schedule();
    };
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") void poll();
      else clearTimer();
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    if (document.visibilityState === "visible") void poll();
    return () => {
      disposed = true;
      clearTimer();
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [polling, refresh]);

  return {
    status: stateForConnection.status,
    phase: stateForConnection.phase,
    error: stateForConnection.error,
    updatedAt: stateForConnection.updatedAt,
    history: stateForConnection.history,
    refresh,
    polling,
    setPolling,
  };
}
