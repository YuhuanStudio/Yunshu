import { ApiError, requestJson, type Connection } from "./api.ts";

/**
 * Typed client stub for the host telemetry the `telemetry` backend branch adds to
 * GET /v1/yunshu/host (contract: docs/guides/TELEMETRY.md there). Nothing renders it yet; the
 * round-9 Host card builds on `fetchHostTelemetry` + `parseHostTelemetry`. Feature detection:
 * an engine without the field (or without the route) is "unsupported", not an error, and every
 * unknown number stays null, never 0.
 */
export type HostTelemetryState = "ok" | "partial" | "unknown";

export interface HostTelemetry {
  state: HostTelemetryState;
  /** Epoch seconds of the sample; null when the engine has none. */
  sampledAt: number | null;
  intervalS: number | null;
  /** Interval energy / elapsed seconds; package is the sum of the reported channels, not wall power. */
  watts: {
    cpu: number | null;
    gpu: number | null;
    ane: number | null;
    dram: number | null;
    package: number | null;
  };
  gpu: {
    /** Weighted by active DVFS residency. */
    frequencyMhz: number | null;
    /** Includes idle residency in the denominator (time fraction, not shader occupancy). */
    activeRatio: number | null;
  };
  temperature: {
    state: HostTelemetryState;
    dieMaxC: number | null;
    dieMeanC: number | null;
    batteryC: number | null;
  };
  /** Why a value is null, keyed by field, as the engine reports it. */
  reasons: Record<string, string>;
  /** Set when state is unknown: why (disabled, stale, unavailable). */
  reason: string | null;
}

export type HostTelemetryResult =
  { kind: "ok"; telemetry: HostTelemetry } | { kind: "unsupported" };

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const state = (v: unknown): HostTelemetryState =>
  v === "ok" || v === "partial" ? v : "unknown";

/** null when the payload has no telemetry section (an older engine). */
export function parseHostTelemetry(payload: unknown): HostTelemetry | null {
  if (!isRecord(payload) || !isRecord(payload.telemetry)) return null;
  const t = payload.telemetry;
  const watts = isRecord(t.watts) ? t.watts : {};
  const gpu = isRecord(t.gpu) ? t.gpu : {};
  const temp = isRecord(t.temperature) ? t.temperature : {};
  const reasons: Record<string, string> = {};
  if (isRecord(t.reasons))
    for (const [k, v] of Object.entries(t.reasons))
      if (typeof v === "string") reasons[k] = v;
  return {
    state: state(t.state),
    sampledAt: num(t.sampled_at),
    intervalS: num(t.interval_s),
    watts: {
      cpu: num(watts.cpu),
      gpu: num(watts.gpu),
      ane: num(watts.ane),
      dram: num(watts.dram),
      package: num(watts.package),
    },
    gpu: {
      frequencyMhz: num(gpu.frequency_mhz),
      activeRatio: num(gpu.active_ratio),
    },
    temperature: {
      state: state(temp.state),
      dieMaxC: num(temp.die_max_c),
      dieMeanC: num(temp.die_mean_c),
      batteryC: num(temp.battery_c),
    },
    reasons,
    reason: typeof t.reason === "string" ? t.reason : null,
  };
}

export async function fetchHostTelemetry(
  connection: Connection,
  signal?: AbortSignal,
): Promise<HostTelemetryResult> {
  try {
    const body = await requestJson<unknown>(connection, "/yunshu/host", {
      signal,
    });
    const telemetry = parseHostTelemetry(body);
    return telemetry ? { kind: "ok", telemetry } : { kind: "unsupported" };
  } catch (e) {
    if (e instanceof ApiError && (e.status === 404 || e.status === 405))
      return { kind: "unsupported" };
    throw e;
  }
}
