import type { EngineStatus } from "./api.ts";
import type { EngineConnectionPhase } from "./useEngine.ts";

/**
 * The overview health verdict: one of ok / watch / bad, plus the reasons.
 * Pure and i18n-free; the page turns each reason `code` into a sentence.
 *
 * Thresholds (each is "watch at or above A, bad at or above B"):
 *   memory in use      active / total Metal memory      80 %      92 %
 *                      (the host's own pressure level `warn` / `critical` counts too)
 *   swap used          host swap, GB                    0.05 GB   2 GB
 *   queue depth        requests waiting                 4         8
 *   oldest wait        longest-waiting queued request   10 s      30 s
 *   error rate         failed share of finished         5 %       20 %
 *                      requests in the last 5 minutes, only from 10 requests up
 *   5xx                server errors in the last 5 minutes   1    3
 * An engine that cannot be reached, or reports a model load error, is bad.
 * A refused token is "watch": the engine is fine, the console cannot see it.
 * Nothing here invents a number: a missing input adds no reason.
 */
export const HEALTH = {
  memory: { watch: 0.8, bad: 0.92 },
  swapGb: { watch: 0.05, bad: 2 },
  queueDepth: { watch: 4, bad: 8 },
  oldestWaitS: { watch: 10, bad: 30 },
  errorRate: { watch: 0.05, bad: 0.2, minRequests: 10 },
  fivexx: { watch: 1, bad: 3 },
  windowMs: 5 * 60_000,
} as const;

export type HealthLevel = "ok" | "watch" | "bad";
export type HealthCode =
  | "offline"
  | "unauthorized"
  | "loadError"
  | "memory"
  | "pressure"
  | "swap"
  | "queueDepth"
  | "oldestWait"
  | "errorRate"
  | "fivexx";
export interface HealthReason {
  code: HealthCode;
  level: Exclude<HealthLevel, "ok">;
  /** Numbers for the sentence, already rounded for display. */
  vars: Record<string, number | string>;
}
export interface Verdict {
  level: HealthLevel;
  reasons: HealthReason[];
}
/** The finished-request facts the verdict needs; `at` is epoch ms. */
export interface FinishedFact {
  at: number;
  /** "error" counts toward the error rate; cancelled and completed do not. */
  outcome: "completed" | "cancelled" | "error";
  statusCode: number | null;
}
export interface HealthInput {
  phase: EngineConnectionPhase;
  status: EngineStatus | null;
  /** Host memory ledger, when the engine has one. */
  ledger?: {
    pressureLevel: string | null;
    swapUsedGb: number | null;
  } | null;
  finished?: readonly FinishedFact[];
  now: number;
}

const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);
const grade = (v: number, t: { watch: number; bad: number }) =>
  v >= t.bad ? "bad" : v >= t.watch ? "watch" : null;
const rank = { bad: 0, watch: 1 } as const;

export function healthVerdict(input: HealthInput): Verdict {
  const { phase, status, now } = input;
  const reasons: HealthReason[] = [];
  const add = (
    code: HealthCode,
    level: HealthReason["level"] | null,
    vars: HealthReason["vars"] = {},
  ) => {
    if (level) reasons.push({ code, level, vars });
  };
  if (phase === "offline") add("offline", "bad");
  else if (phase === "unauthorized") add("unauthorized", "watch");
  if (status && phase === "online") {
    if (status.load_error)
      add("loadError", "bad", { error: status.load_error });
    const { active_gb: used, total_gb: total } = status.memory;
    if (finite(used) && finite(total) && total > 0)
      add("memory", grade(used / total, HEALTH.memory), {
        pct: Math.round((used / total) * 100),
        used: Math.round(used * 10) / 10,
        total: Math.round(total),
      });
    const level = input.ledger?.pressureLevel;
    if (level === "critical") add("pressure", "bad");
    else if (level === "warn" || level === "warning") add("pressure", "watch");
    const swap = input.ledger?.swapUsedGb;
    if (finite(swap))
      add("swap", grade(swap, HEALTH.swapGb), {
        gb: Math.round(swap * 10) / 10,
      });
    const queued = status.requests.items.filter((r) => r.phase === "queued");
    const depth = Math.max(status.requests.queued, queued.length);
    add("queueDepth", grade(depth, HEALTH.queueDepth), { n: depth });
    const oldest = queued.reduce(
      (m, r) => (finite(r.elapsed_s) ? Math.max(m, r.elapsed_s) : m),
      0,
    );
    add("oldestWait", grade(oldest, HEALTH.oldestWaitS), {
      s: Math.round(oldest),
    });
    const recent = (input.finished ?? []).filter(
      (r) => now - r.at <= HEALTH.windowMs && r.at <= now + 60_000,
    );
    const failed = recent.filter((r) => r.outcome === "error").length;
    if (recent.length >= HEALTH.errorRate.minRequests)
      add("errorRate", grade(failed / recent.length, HEALTH.errorRate), {
        pct: Math.round((failed / recent.length) * 100),
        n: recent.length,
      });
    const fivexx = recent.filter(
      (r) => r.statusCode != null && r.statusCode >= 500,
    ).length;
    add("fivexx", grade(fivexx, HEALTH.fivexx), { n: fivexx });
  }
  reasons.sort((a, b) => rank[a.level] - rank[b.level]);
  const level: HealthLevel = reasons.some((r) => r.level === "bad")
    ? "bad"
    : reasons.length
      ? "watch"
      : "ok";
  return { level, reasons };
}

/**
 * Finished-request facts from the server's recent-request ring. `t` is the
 * finish time, in seconds on the engine; a row without a time or an outcome
 * says nothing, so it is dropped rather than guessed.
 */
export function factsFromRows(
  rows: readonly {
    t?: number;
    outcome?: FinishedFact["outcome"];
    status_code?: number | null;
  }[],
): FinishedFact[] {
  const facts: FinishedFact[] = [];
  for (const r of rows) {
    if (!finite(r.t) || !r.outcome) continue;
    facts.push({
      at: r.t > 1e11 ? r.t : r.t * 1000,
      outcome: r.outcome,
      statusCode: finite(r.status_code) ? r.status_code : null,
    });
  }
  return facts;
}

/** The page that deals with each reason; the verdict line links to the worst one. */
export const HEALTH_TARGET: Record<HealthCode, string> = {
  offline: "settings",
  unauthorized: "settings",
  loadError: "models",
  memory: "models",
  pressure: "models",
  swap: "diagnostics",
  queueDepth: "requests",
  oldestWait: "requests",
  errorRate: "logs",
  fivexx: "logs",
};
