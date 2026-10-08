import { connectionScope, useScopedState } from "./scoped-state.ts";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "@yuhuanowo/yunui";
import { ApiError, requestJson, type Connection } from "./api.ts";
import { factsFromRows, healthVerdict, type Verdict } from "./health.ts";
import { useMemoryLedger, type MemoryLedgerData } from "./memory-api.ts";
import {
  STORAGE_KEY,
  TOAST_TONES,
  addEvents,
  initialTracker,
  loadNotifications,
  markAllRead,
  observe,
  unreadCount,
  type DownloadFact,
  type NotifRecord,
} from "./notifications.ts";
import { describeNotification } from "./notification-text.ts";
import { useRecentRequests } from "./recentRequests.ts";
import type { Engine } from "./ui.tsx";

/**
 * What the shell derives from polling and shares with every page: the host
 * memory ledger, the health verdict and the notification list. One poller
 * each, here, so the status band, the overview and the bell cannot disagree.
 */
export interface Signals {
  ledger: MemoryLedgerData | null;
  verdict: Verdict;
  notifications: readonly NotifRecord[];
  unread: number;
  markRead: () => void;
  clearAll: () => void;
}

const quiet: Signals = {
  ledger: null,
  verdict: { level: "ok", reasons: [] },
  notifications: [],
  unread: 0,
  markRead: () => undefined,
  clearAll: () => undefined,
};
const SignalsContext = createContext<Signals>(quiet);
export const SignalsProvider = SignalsContext.Provider;
export const useSignals = () => useContext(SignalsContext);

const DOWNLOAD_POLL_ACTIVE_MS = 4_000;
const DOWNLOAD_POLL_IDLE_MS = 20_000;

/** Download jobs, light enough to poll: id, repo, state. null when the engine has no route. */
function useDownloadFacts(
  connection: Connection,
  enabled: boolean,
): readonly DownloadFact[] | null {
  const [facts, setFacts] = useScopedState<readonly DownloadFact[] | null>(
    connectionScope(connection),
    null,
  );
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      let next = DOWNLOAD_POLL_IDLE_MS;
      if (document.visibilityState === "visible") {
        try {
          const body = await requestJson<{ downloads?: unknown }>(
            connection,
            "/yunshu/downloads",
            { signal: controller.signal },
          );
          if (controller.signal.aborted) return;
          const rows = Array.isArray(body.downloads) ? body.downloads : [];
          const list = rows.flatMap((r: Record<string, unknown>) =>
            typeof r?.id === "string" && typeof r.state === "string"
              ? [
                  {
                    id: r.id,
                    repo: typeof r.repo === "string" ? r.repo : r.id,
                    state: r.state,
                    error: typeof r.error === "string" ? r.error : null,
                  },
                ]
              : [],
          );
          setFacts(list);
          if (list.some((d) => d.state === "queued" || d.state === "running"))
            next = DOWNLOAD_POLL_ACTIVE_MS;
        } catch (e) {
          if (controller.signal.aborted) return;
          // An older engine has no download route: stop asking.
          if (e instanceof ApiError && (e.status === 404 || e.status === 405)) {
            setFacts(null);
            return;
          }
          if (e instanceof ApiError && (e.status === 401 || e.status === 403))
            return;
        }
      }
      timer = setTimeout(() => void tick(), next);
    };
    void tick();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [connection.baseUrl, connection.token, enabled]);
  return facts;
}

/** The session's notifications live per service address (never per token: the token is not stored). */
const storageKey = (c: Connection) => `${STORAGE_KEY}:${c.baseUrl}`;
function readStored(c: Connection): NotifRecord[] {
  try {
    return loadNotifications(sessionStorage.getItem(storageKey(c)));
  } catch {
    return [];
  }
}

export function useShellSignals(
  engine: Engine,
  connection: Connection,
): Signals {
  const online = engine.phase === "online";
  const ledger = useMemoryLedger(connection, online, 5000).data;
  const recent = useRecentRequests(
    connection,
    online ? engine.status?.last?.request_id : null,
  );
  const downloads = useDownloadFacts(connection, online);
  const facts = useMemo(() => factsFromRows(recent.rows), [recent.rows]);
  const verdict = useMemo(
    () =>
      healthVerdict({
        phase: engine.phase,
        status: engine.status,
        ledger: ledger
          ? {
              pressureLevel: ledger.host.pressure_level,
              swapUsedGb: ledger.host.swap_used_gb,
            }
          : null,
        finished: facts,
        now: Date.now(),
      }),
    [engine.phase, engine.status, ledger, facts],
  );

  // The list belongs to one connection identity: switching address or token shows that
  // service's own history (empty for a token change, restored for a known address) and the
  // change tracker starts from a fresh baseline, so service B never announces service A's events.
  const scope = connectionScope(connection);
  const [held, setHeld] = useState<{ scope: string; list: NotifRecord[] }>(
    () => ({ scope, list: readStored(connection) }),
  );
  const notifications =
    held.scope === scope ? held.list : readStored(connection);
  const setNotifications = useCallback(
    (update: (list: NotifRecord[]) => NotifRecord[]) =>
      setHeld((h) => ({
        scope,
        list: update(h.scope === scope ? h.list : readStored(connection)),
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [scope],
  );
  const tracker = useRef(initialTracker());
  const trackerScope = useRef(scope);
  useEffect(() => {
    // A different service is a fresh baseline, not a restart or a recovery
    // (reset after the approach of consolereview 89403c99).
    if (trackerScope.current !== scope) {
      tracker.current = initialTracker();
      trackerScope.current = scope;
    }
    if (engine.phase === "connecting") return;
    const { state, events } = observe(tracker.current, {
      at: Date.now(),
      phase: engine.phase,
      status: engine.status,
      finished: facts,
      downloads,
    });
    tracker.current = state;
    if (!events.length) return;
    setNotifications((list) => addEvents(list, events));
    for (const e of events) {
      if (!TOAST_TONES.includes(e.tone)) continue;
      const { title, body } = describeNotification(e);
      (e.tone === "error" ? toast.error : toast.warning)(title, body);
    }
  }, [scope, engine.phase, engine.status, facts, downloads]);
  useEffect(() => {
    try {
      sessionStorage.setItem(
        storageKey(connection),
        JSON.stringify(notifications),
      );
    } catch {
      /* storage may be unavailable */
    }
  }, [notifications, connection.baseUrl]);

  const markRead = useCallback(
    () => setNotifications((list) => markAllRead(list)),
    [setNotifications],
  );
  const clearAll = useCallback(
    () => setNotifications(() => []),
    [setNotifications],
  );
  return useMemo(
    () => ({
      ledger,
      verdict,
      notifications,
      unread: unreadCount(notifications),
      markRead,
      clearAll,
    }),
    [ledger, verdict, notifications, markRead, clearAll],
  );
}
