import type { EngineStatus } from "./api.ts";
import { HEALTH, type FinishedFact } from "./health.ts";
import type { EngineConnectionPhase } from "./useEngine.ts";

/**
 * Client-side notifications, derived from what the console already polls.
 * `observe` is a pure state machine: feed it each new observation, get back the
 * events that appeared since the last one. Calm by construction:
 *   - the first observation never produces events (it is not a change);
 *   - memory pressure has hysteresis (raise at the watch line, clear 5 points lower);
 *   - a 5xx burst (3 or more within a minute) fires once per 5 minutes;
 *   - downloads already finished when the page opened are never announced.
 */
export type NotifKind =
  | "model.loaded"
  | "model.failed"
  | "download.done"
  | "download.failed"
  | "engine.restart"
  | "engine.offline"
  | "engine.back"
  | "memory.warn"
  | "fivexx.burst";
export type NotifTone = "success" | "info" | "warn" | "error";

/** Where the event's row leads (a console hash route). */
export const NOTIF_HREF: Record<NotifKind, string> = {
  "model.loaded": "#/models",
  "model.failed": "#/models",
  "download.done": "#/downloads",
  "download.failed": "#/downloads",
  "engine.restart": "#/diagnostics",
  "engine.offline": "#/diagnostics",
  "engine.back": "#/overview",
  "memory.warn": "#/models",
  "fivexx.burst": "#/requests",
};
const TONE: Record<NotifKind, NotifTone> = {
  "model.loaded": "success",
  "model.failed": "error",
  "download.done": "success",
  "download.failed": "error",
  "engine.restart": "warn",
  "engine.offline": "error",
  "engine.back": "info",
  "memory.warn": "warn",
  "fivexx.burst": "error",
};
/** Only these interrupt with a toast; the rest wait in the center. */
export const TOAST_TONES: readonly NotifTone[] = ["warn", "error"];

export interface NotifEvent {
  id: string;
  kind: NotifKind;
  tone: NotifTone;
  /** Epoch ms. */
  at: number;
  /** Raw values for the sentence (model name, percent, count); formatted when shown. */
  vars: Record<string, string | number>;
}
export interface DownloadFact {
  id: string;
  repo: string;
  state: string;
  error: string | null;
}
export interface Observation {
  at: number;
  phase: EngineConnectionPhase;
  status: EngineStatus | null;
  finished: readonly FinishedFact[];
  /** null: the engine has no download route (or it was not asked). */
  downloads: readonly DownloadFact[] | null;
}
export interface TrackerState {
  prev: Observation | null;
  memoryWarned: boolean;
  lastBurstAt: number;
  /** Last status seen while online: the baseline for restart and model changes. */
  lastStatus: EngineStatus | null;
  offlineAnnounced: boolean;
}
export const initialTracker = (): TrackerState => ({
  prev: null,
  memoryWarned: false,
  lastBurstAt: 0,
  lastStatus: null,
  offlineAnnounced: false,
});

const BURST_COUNT = 3;
const BURST_WINDOW_MS = 60_000;
const BURST_COOLDOWN_MS = 5 * 60_000;
const MEMORY_CLEAR_GAP = 0.05;
const ACTIVE_DOWNLOAD = ["queued", "running"];

const leaf = (id: string) => id.split("/").filter(Boolean).at(-1) ?? id;

export function observe(
  state: TrackerState,
  next: Observation,
): { state: TrackerState; events: NotifEvent[] } {
  const events: NotifEvent[] = [];
  const push = (kind: NotifKind, key: string, vars: NotifEvent["vars"] = {}) =>
    events.push({
      id: `${kind}:${key}:${next.at}`,
      kind,
      tone: TONE[kind],
      at: next.at,
      vars,
    });
  const prev = state.prev;
  const before = state.lastStatus;
  const after = next.phase === "online" ? next.status : null;
  let { memoryWarned, lastBurstAt, offlineAnnounced } = state;

  if (prev) {
    if (prev.phase === "online" && next.phase === "offline") {
      push("engine.offline", "off");
      offlineAnnounced = true;
    }
    if (after && before) {
      const restarted = after.uptime_s < before.uptime_s;
      if (restarted) push("engine.restart", "restart");
      else if (offlineAnnounced) push("engine.back", "back");
      offlineAnnounced = false;
      if (!restarted) {
        const was = new Map(before.models.map((m) => [m.id, m]));
        for (const m of after.models) {
          const old = was.get(m.id);
          if (!old) continue;
          const name = leaf(m.id);
          if (m.loaded && !old.loaded)
            push("model.loaded", m.id, { model: name });
          else if (
            (m.error && !old.error) ||
            (old.loading && !m.loading && !m.loaded)
          )
            push("model.failed", m.id, { model: name });
        }
        if (after.load_error && !before.load_error)
          push("model.failed", "engine", { model: after.load_error });
      }
    }
  }
  if (after) {
    const { active_gb: used, total_gb: total } = after.memory;
    if (typeof used === "number" && typeof total === "number" && total > 0) {
      const usage = used / total;
      if (!memoryWarned && usage >= HEALTH.memory.watch) {
        if (prev) push("memory.warn", "mem", { pct: Math.round(usage * 100) });
        memoryWarned = true;
      } else if (memoryWarned && usage < HEALTH.memory.watch - MEMORY_CLEAR_GAP)
        memoryWarned = false;
    }
    const burst = next.finished.filter(
      (r) =>
        r.statusCode != null &&
        r.statusCode >= 500 &&
        next.at - r.at <= BURST_WINDOW_MS,
    ).length;
    if (burst >= BURST_COUNT && next.at - lastBurstAt >= BURST_COOLDOWN_MS) {
      if (prev) push("fivexx.burst", "5xx", { n: burst });
      lastBurstAt = next.at;
    }
  }
  if (prev?.downloads && next.downloads) {
    const was = new Map(prev.downloads.map((d) => [d.id, d]));
    for (const d of next.downloads) {
      const old = was.get(d.id);
      if (!old || !ACTIVE_DOWNLOAD.includes(old.state)) continue;
      if (d.state === "done")
        push("download.done", d.id, { model: leaf(d.repo) });
      else if (d.state === "failed")
        push("download.failed", d.id, { model: leaf(d.repo) });
    }
  }
  return {
    state: {
      prev: next,
      memoryWarned,
      lastBurstAt,
      lastStatus: after ?? state.lastStatus,
      offlineAnnounced,
    },
    events,
  };
}

/** The saved list of one session: newest first, at most 50, with read state. */
export interface NotifRecord extends NotifEvent {
  read: boolean;
}
export const MAX_NOTIFICATIONS = 50;
export const STORAGE_KEY = "yunshu.console.notifications";

export function addEvents(
  list: readonly NotifRecord[],
  events: readonly NotifEvent[],
): NotifRecord[] {
  const known = new Set(list.map((n) => n.id));
  const fresh = events
    .filter((e) => !known.has(e.id))
    .map((e) => ({ ...e, read: false }))
    .reverse();
  return [...fresh, ...list].slice(0, MAX_NOTIFICATIONS);
}
export const markAllRead = (list: readonly NotifRecord[]): NotifRecord[] =>
  list.map((n) => (n.read ? n : { ...n, read: true }));
export const unreadCount = (list: readonly NotifRecord[]) =>
  list.reduce((n, r) => n + (r.read ? 0 : 1), 0);

const KINDS = new Set<string>(Object.keys(NOTIF_HREF));
/** Tolerant load: anything malformed is dropped, never thrown. */
export function loadNotifications(raw: string | null): NotifRecord[] {
  try {
    const data: unknown = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(data)) return [];
    return data
      .filter(
        (r): r is NotifRecord =>
          !!r &&
          typeof r.id === "string" &&
          KINDS.has(r.kind) &&
          typeof r.at === "number" &&
          typeof r.read === "boolean" &&
          !!r.vars &&
          typeof r.vars === "object",
      )
      .map((r) => ({ ...r, tone: TONE[r.kind as NotifKind] }))
      .slice(0, MAX_NOTIFICATIONS);
  } catch {
    return [];
  }
}
