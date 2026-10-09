import { BYTES_PER_GB, binaryGb, readGb } from "./byte-format.ts";
import { connectionScope, useScopedState } from "./scoped-state.ts";
import { useEffect, useState } from "react";
import { fixed } from "./i18n/format.ts";
import { t } from "./i18n/index.ts";
import { ApiError, requestJson, type Connection } from "./api.ts";

export type MemoryOwner = {
  kind: string;
  id: string | null;
  bytes: number | null;
  gb: number | null;
  reclaimable: boolean;
  estimated: boolean;
  source: string | null;
};

export type MemoryLedgerData = {
  total_gb: number | null;
  free_gb: number | null;
  host: {
    pressure_level: string | null;
    swap_used_gb: number | null;
    swap_total_gb: number | null;
    wired_limit_gb: number | null;
  };
  mlx: {
    active_gb: number | null;
    cache_gb: number | null;
    peak_gb: number | null;
    recommended_working_set_gb: number | null;
  };
  owners: MemoryOwner[];
  attribution_overshoot_gb: number | null;
  limits: {
    apc_max_gb: number | null;
    apc_warm_max_gb: number | null;
    guard_margin_pct: number | null;
  };
};

const rec = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
/** null for anything that is not a finite number: unknown stays unknown. */
const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
/** A `*_gb` field: the engine reports decimal GB, the console shows binary ("GB" as macOS does). */
const gbNum = (v: unknown): number | null => {
  const n = num(v);
  return n == null ? null : binaryGb(n);
};

/** Tolerant parse: a missing or mistyped field becomes null, never 0. */
export function parseMemory(raw: unknown): MemoryLedgerData {
  const r = rec(raw),
    host = rec(r.host),
    mlx = rec(r.mlx),
    limits = rec(r.limits);
  return {
    total_gb: readGb(r, "total"),
    free_gb: readGb(r, "free"),
    host: {
      pressure_level: str(host.pressure_level),
      swap_used_gb: readGb(host, "swap_used"),
      swap_total_gb: readGb(host, "swap_total"),
      wired_limit_gb: readGb(host, "wired_limit"),
    },
    mlx: {
      active_gb: readGb(mlx, "active"),
      cache_gb: readGb(mlx, "cache"),
      peak_gb: readGb(mlx, "peak"),
      recommended_working_set_gb: readGb(mlx, "recommended_working_set"),
    },
    owners: (Array.isArray(r.owners) ? r.owners : []).map((o) => {
      const x = rec(o);
      return {
        kind: str(x.kind) ?? "other",
        id: str(x.id),
        bytes: num(x.bytes),
        gb:
          num(x.bytes) != null
            ? (x.bytes as number) / BYTES_PER_GB
            : gbNum(x.gb),
        reclaimable: x.reclaimable === true,
        estimated: x.estimated === true,
        source: str(x.source),
      };
    }),
    attribution_overshoot_gb: readGb(r, "attribution_overshoot"),
    limits: {
      apc_max_gb: num(limits.apc_max_gb),
      apc_warm_max_gb: num(limits.apc_warm_max_gb),
      guard_margin_pct: num(limits.guard_margin_pct),
    },
  };
}

/** The ledger, or "unsupported" on a server without the route (404/405). */
export async function getMemory(
  connection: Connection,
  signal?: AbortSignal,
): Promise<MemoryLedgerData | "unsupported"> {
  try {
    return parseMemory(
      await requestJson<unknown>(connection, "/yunshu/memory", { signal }),
    );
  } catch (e) {
    if (e instanceof ApiError && (e.status === 404 || e.status === 405))
      return "unsupported";
    throw e;
  }
}

export type LedgerState = {
  data: MemoryLedgerData | null;
  unsupported: boolean;
  error: string;
};

/** Polls the ledger every `intervalMs` while visible; stops polling once unsupported. */
export function useMemoryLedger(
  connection: Connection,
  enabled: boolean,
  intervalMs = 5000,
): LedgerState {
  const [state, setState] = useScopedState<LedgerState>(
    connectionScope(connection),
    { data: null, unsupported: false, error: "" },
  );
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      if (document.visibilityState === "visible") {
        try {
          const v = await getMemory(connection, controller.signal);
          if (controller.signal.aborted) return;
          if (v === "unsupported") {
            setState({ data: null, unsupported: true, error: "" });
            return;
          }
          setState({ data: v, unsupported: false, error: "" });
        } catch (e) {
          if (controller.signal.aborted) return;
          setState((s) => ({
            ...s,
            error: e instanceof Error ? e.message : String(e),
          }));
        }
      }
      timer = setTimeout(() => void tick(), intervalMs);
    };
    void tick();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [connection.baseUrl, connection.token, enabled, intervalMs]);
  return state;
}

export type Fit =
  | { verdict: "fits"; text: string }
  | { verdict: "tight"; text: string }
  | { verdict: "no"; text: string }
  | { verdict: "unknown"; text: string };

/** Honest fit: size vs the ledger's free_gb. Informs only; callers never block loading. */
export function fitVerdict(
  sizeGb: number | null | undefined,
  freeGb: number | null | undefined,
): Fit {
  if (!sizeGb || sizeGb <= 0)
    return { verdict: "unknown", text: t("overview.fit.unknownSize") };
  if (freeGb == null)
    return { verdict: "unknown", text: t("overview.fit.unknownFree") };
  const vars = { size: fixed(sizeGb), free: fixed(freeGb) };
  if (sizeGb > freeGb)
    return { verdict: "no", text: t("overview.fit.no", vars) };
  if (sizeGb > freeGb * 0.85)
    return { verdict: "tight", text: t("overview.fit.tight", vars) };
  return { verdict: "fits", text: t("overview.fit.fits", vars) };
}

/**
 * The one "available memory" figure of the Models page. Every unloaded row says how much it needs
 * against it, so the summary card must show the same number: the system's available memory from the
 * ledger. Only when the ledger is missing (an older engine) does it fall back to total minus Metal's
 * allocation, which ignores other apps, and it says so (`source: "metal"`) instead of passing it off as
 * the same thing.
 */
export function modelsFreeMemory(input: {
  ledgerFreeGb: number | null | undefined;
  totalGb: number | null | undefined;
  activeGb: number | null | undefined;
}): { gb: number | null; source: "system" | "metal" } {
  if (input.ledgerFreeGb != null)
    return { gb: input.ledgerFreeGb, source: "system" };
  if (input.totalGb != null && input.activeGb != null)
    return { gb: input.totalGb - input.activeGb, source: "metal" };
  return { gb: null, source: "metal" };
}
