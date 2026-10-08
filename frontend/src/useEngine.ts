import { activity, type ActivityPhase } from "./engineView";
import {
  emaStep,
  initialStablePhase,
  nextStablePhase,
  type StablePhaseState,
} from "./phase-stability";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  fetchStatus,
  type Connection,
  type EngineStatus,
} from "./api";
import type { ObservedRequest } from "./analytics";
import { fetchServerHistory } from "./history-api";
import {
  gapPoint,
  mergeSeries,
  pointFromStatus,
  type SeriesPoint,
} from "./series";

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
  /** HTTP status of the failed poll, or null when the engine was unreachable. */
  errorStatus: number | null;
  updatedAt: number | null;
  history: readonly EngineHistoryPoint[];
  /** Slim chart samples: the engine's own history first, then live polls. */
  series: readonly SeriesPoint[];
  /** Finished requests seen since the page opened, oldest first, no repeats. */
  finished: readonly ObservedRequest[];
  /** Epoch ms of the oldest engine-side history row, or null when there is none. */
  historyFrom: number | null;
  /** One or two polls failed in a row: the page shows the last good numbers and says so. */
  retrying: boolean;
  /** The phase to show: forward moves at once, steps back (a brief prefill, a gap) must hold. */
  livePhase: ActivityPhase;
  /** Live decode tok/s smoothed over about a second; null when nothing is decoding. */
  liveDecodeTps: number | null;
  refresh: () => Promise<void>;
  polling: boolean;
  setPolling: (enabled: boolean) => void;
}

/** Idle: a calm poll. */
const POLL_INTERVAL_MS = 3_000;
/**
 * While a request runs (and for a moment after), the status is read at about 4 Hz so tok/s, phase and
 * prefill progress move smoothly. The engine builds /v1/yunshu/status from live counters on every call, so
 * there is no 1 Hz sampling step in the way; it is simply asked more often, and only while it is busy.
 */
export const BUSY_POLL_INTERVAL_MS = 250;
const BUSY_LINGER_MS = 2_000;
/** The history and chart series keep their old density however often the status is read. */
const SAMPLE_EVERY_MS = 2_500;
/** Offline is shown only after this many consecutive failed polls. */
const OFFLINE_AFTER_FAILURES = 3;
const MAX_HISTORY_POINTS = 1_200;
const MAX_SERIES_POINTS = 2_400;
const MAX_FINISHED = 1_000;

interface EngineViewState {
  connectionKey: symbol;
  status: EngineStatus | null;
  phase: EngineConnectionPhase;
  error: string | null;
  /** HTTP status of the failed poll, or null when the engine was unreachable. */
  errorStatus: number | null;
  updatedAt: number | null;
  history: EngineHistoryPoint[];
  series: SeriesPoint[];
  finished: ObservedRequest[];
  historyFrom: number | null;
  /** A poll failed but fewer than OFFLINE_AFTER_FAILURES in a row: the numbers are the last good ones. */
  retrying: boolean;
}

/** The series with an outage marker at its end, unless it already ends in one. */
function withGap(series: SeriesPoint[], at: number): SeriesPoint[] {
  const last = series[series.length - 1];
  return !last || last.gap ? series : [...series, gapPoint(at)];
}

function initialState(connectionKey: symbol): EngineViewState {
  return {
    connectionKey,
    status: null,
    phase: "connecting",
    error: null,
    errorStatus: null,
    updatedAt: null,
    history: [],
    series: [],
    finished: [],
    historyFrom: null,
    retrying: false,
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
  const failuresRef = useRef(0);
  /** The connection whose engine-side history was already requested. */
  const historyAskedRef = useRef<symbol | null>(null);
  const activeControllerRef = useRef<AbortController | null>(null);
  const stableRef = useRef<StablePhaseState>(initialStablePhase("idle"));
  const emaRef = useRef<{ v: number | null; at: number }>({ v: null, at: 0 });
  const lastBusyRef = useRef(0);
  const lastSampleRef = useRef(0);
  const memEmaRef = useRef<{ v: number | null; at: number }>({
    v: null,
    at: 0,
  });
  const [live, setLive] = useState<{
    phase: ActivityPhase;
    tps: number | null;
    /** Metal active memory smoothed over about 1.5 s: it moves with every allocation. */
    mem: number | null;
  }>({ phase: "idle", tps: null, mem: null });
  const inFlightRef = useRef<{
    connectionKey: symbol;
    promise: Promise<void>;
  } | null>(null);

  useEffect(() => {
    const generation = ++generationRef.current;
    activeControllerRef.current?.abort();
    activeControllerRef.current = null;
    inFlightRef.current = null;
    // Failures counted against the previous service say nothing about this one.
    failuresRef.current = 0;
    // Smoothing carries nothing over from another service.
    stableRef.current = initialStablePhase("idle");
    emaRef.current = { v: null, at: 0 };
    memEmaRef.current = { v: null, at: 0 };
    lastBusyRef.current = 0;
    setLive({ phase: "idle", tps: null, mem: null });
    setState(initialState(connectionKey));
    setPolling(true);
    return () => {
      if (generationRef.current === generation) generationRef.current += 1;
      activeControllerRef.current?.abort();
      activeControllerRef.current = null;
      inFlightRef.current = null;
    };
  }, [connectionKey]);

  const refresh = useCallback(
    (force = false): Promise<void> => {
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
          failuresRef.current = 0;
          // Steady phase and smoothed speed, from the freshest sample (see phase-stability.ts).
          const raw = activity(status);
          if (raw.phase !== "idle") lastBusyRef.current = at;
          stableRef.current = nextStablePhase(stableRef.current, raw.phase, at);
          const prevEma = emaRef.current;
          emaRef.current = {
            v: emaStep(prevEma.v, raw.decodeNow, at - prevEma.at),
            at,
          };
          const rawMem =
            typeof status.memory.active_gb === "number"
              ? status.memory.active_gb
              : null;
          memEmaRef.current = {
            v: emaStep(
              memEmaRef.current.v,
              rawMem,
              at - memEmaRef.current.at,
              1500,
            ),
            at,
          };
          setLive((cur) =>
            cur.phase === stableRef.current.phase &&
            cur.tps === emaRef.current.v &&
            cur.mem === memEmaRef.current.v
              ? cur
              : {
                  phase: stableRef.current.phase,
                  tps: emaRef.current.v,
                  mem: memEmaRef.current.v,
                },
          );
          // A reading the user asked for (the 更新 button) is always a sample; background polls keep the density.
          const sampleDue =
            force || at - lastSampleRef.current >= SAMPLE_EVERY_MS;
          if (sampleDue) lastSampleRef.current = at;
          setState((current) => {
            const previous =
              current.connectionKey === connectionKey
                ? current
                : initialState(connectionKey);
            const restarted =
              previous.status !== null &&
              status.uptime_s < previous.status.uptime_s;
            if (restarted) historyAskedRef.current = null;
            const samples = restarted ? [] : previous.history;
            // A restart keeps the chart history but marks the break, so the line
            // never runs straight through the time the engine was down.
            const series = restarted
              ? withGap(previous.series, at - 1)
              : previous.series;
            const known = restarted ? [] : previous.finished;
            const last = status.last;
            // The engine reports only its latest finished request; keep each once.
            const isNew =
              last != null &&
              !known.some((r) => r.request_id === last.request_id);
            return {
              connectionKey,
              status,
              phase: "online",
              error: null,
              errorStatus: null,
              updatedAt: at,
              // Between samples only the latest status moves; the history keeps its density.
              history: sampleDue
                ? [...samples, { at, status }].slice(-MAX_HISTORY_POINTS)
                : samples,
              series: sampleDue
                ? [...series, pointFromStatus(at, status)].slice(
                    -MAX_SERIES_POINTS,
                  )
                : series,
              finished: isNew
                ? [...known, { ...last, firstObservedAt: at }].slice(
                    -MAX_FINISHED,
                  )
                : known,
              historyFrom: restarted ? null : previous.historyFrom,
              retrying: false,
            };
          });
          // Backfill the charts from the engine's own history, once per
          // connection (and again after an engine restart). A server without
          // the route just means the charts start from live polls.
          if (historyAskedRef.current !== connectionKey) {
            historyAskedRef.current = connectionKey;
            void fetchServerHistory(apiConnection, {
              signal: controller.signal,
              // A newer engine sends exact bytes in /status and binary GB everywhere.
              binary:
                (status.memory as { total_bytes?: unknown }).total_bytes !=
                null,
            }).then(
              (loaded) => {
                if (!loaded || generation !== generationRef.current) return;
                setState((current) =>
                  current.connectionKey !== connectionKey
                    ? current
                    : {
                        ...current,
                        series: mergeSeries(
                          loaded.points,
                          current.series.filter((p) => !p.backfilled),
                          Date.now(),
                        ),
                        historyFrom: loaded.points[0].at,
                      },
                );
              },
              () => undefined,
            );
          }
        } catch (error) {
          if (
            controller.signal.aborted ||
            generation !== generationRef.current ||
            connectionKey !== connectionKeyRef.current
          )
            return;
          const apiError = error instanceof ApiError ? error : null;
          const refused = apiError?.status === 401 || apiError?.status === 403;
          if (refused) setPolling(false);
          failuresRef.current += 1;
          // A restart or one slow poll must not flash the offline banner.
          if (!refused && failuresRef.current < OFFLINE_AFTER_FAILURES) {
            // Keep the last numbers, but say so: they are no longer live.
            setState((current) =>
              current.connectionKey === connectionKey
                ? { ...current, retrying: true }
                : current,
            );
            return;
          }
          setState((current) => {
            const previous =
              current.connectionKey === connectionKey
                ? current
                : initialState(connectionKey);
            return {
              ...previous,
              series: withGap(previous.series, Date.now()),
              connectionKey,
              phase:
                apiError?.status === 401 || apiError?.status === 403
                  ? "unauthorized"
                  : "offline",
              error:
                apiError?.publicMessage ??
                "Could not reach the Yunshu service.",
              errorStatus: apiError?.status ?? null,
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
    },
    [apiConnection, connectionKey],
  );

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
        const busy = Date.now() - lastBusyRef.current < BUSY_LINGER_MS;
        timer = setTimeout(
          () => {
            void poll();
          },
          busy ? BUSY_POLL_INTERVAL_MS : POLL_INTERVAL_MS,
        );
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

  const manualRefresh = useCallback(() => refresh(true), [refresh]);
  // The status handed to the pages carries the steady phase and smoothed speed (see phase-stability.ts).
  const rawStatus = stateForConnection.status;
  const status = useMemo(
    () =>
      rawStatus
        ? {
            ...rawStatus,
            memory:
              live.mem != null
                ? { ...rawStatus.memory, active_gb: live.mem }
                : rawStatus.memory,
            console_live: { phase: live.phase, tps: live.tps },
          }
        : null,
    [rawStatus, live.phase, live.tps, live.mem],
  );
  return {
    status,
    phase: stateForConnection.phase,
    error: stateForConnection.error,
    errorStatus: stateForConnection.errorStatus,
    updatedAt: stateForConnection.updatedAt,
    history: stateForConnection.history,
    series: stateForConnection.series,
    finished: stateForConnection.finished,
    historyFrom: stateForConnection.historyFrom,
    livePhase: live.phase,
    liveDecodeTps: live.tps,
    retrying: stateForConnection.retrying,
    refresh: manualRefresh,
    polling,
    setPolling,
  };
}
