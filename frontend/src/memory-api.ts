import { connectionScope, useScopedState } from "./scoped-state";
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

/** Tolerant parse: a missing or mistyped field becomes null, never 0. */
export function parseMemory(raw: unknown): MemoryLedgerData {
  const r = rec(raw),
    host = rec(r.host),
    mlx = rec(r.mlx),
    limits = rec(r.limits);
  return {
    total_gb: num(r.total_gb),
    free_gb: num(r.free_gb),
    host: {
      pressure_level: str(host.pressure_level),
      swap_used_gb: num(host.swap_used_gb),
      swap_total_gb: num(host.swap_total_gb),
      wired_limit_gb: num(host.wired_limit_gb),
    },
    mlx: {
      active_gb: num(mlx.active_gb),
      cache_gb: num(mlx.cache_gb),
      peak_gb: num(mlx.peak_gb),
      recommended_working_set_gb: num(mlx.recommended_working_set_gb),
    },
    owners: (Array.isArray(r.owners) ? r.owners : []).map((o) => {
      const x = rec(o);
      return {
        kind: str(x.kind) ?? "other",
        id: str(x.id),
        bytes: num(x.bytes),
        gb: num(x.gb),
        reclaimable: x.reclaimable === true,
        estimated: x.estimated === true,
        source: str(x.source),
      };
    }),
    attribution_overshoot_gb: num(r.attribution_overshoot_gb),
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
