import { ApiError, requestJson, type Connection } from "./api.ts";

/**
 * Per-request latency stages from `GET /v1/yunshu/requests/recent` (backend branch consolefeat):
 * `latency.milestones_ms` (relative to gateway receive) and `latency.durations_ms`. A stage the
 * engine did not observe, or fused into another, is null and stays null, never 0.
 */
export const LATENCY_STAGES = [
  { id: "model_lease", from: "model_lease_start", to: "model_lease" },
  { id: "gateway_admit", from: "gateway_receive", to: "gateway_admit" },
  { id: "engine_queue", from: "engine_submit", to: "engine_admit" },
  { id: "template_tokenize", from: "template_start", to: "template_end" },
  { id: "apc_lookup_restore", from: "apc_start", to: "apc_end" },
  { id: "prefill", from: "prefill_start", to: "prefill_end" },
  { id: "first_decode", from: "prefill_end", to: "first_decode" },
  { id: "sse_first_flush", from: "first_decode", to: "sse_first_flush" },
] as const;
export type LatencyStageId = (typeof LATENCY_STAGES)[number]["id"];

export interface RequestLatency {
  /** Stage duration in ms, null when unobserved or fused. */
  durations: Record<LatencyStageId, number | null>;
  milestones: Record<string, number>;
  /** Where each observed stage sits on the request timeline (ms since gateway receive); absent when the marks are. */
  spans: { id: LatencyStageId; start: number; end: number }[];
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;

export function parseLatency(value: unknown): RequestLatency | null {
  if (!isRecord(value)) return null;
  const d = isRecord(value.durations_ms) ? value.durations_ms : null;
  if (!d) return null;
  const m = isRecord(value.milestones_ms) ? value.milestones_ms : {};
  const milestones: Record<string, number> = {};
  for (const [k, v] of Object.entries(m)) {
    const n = num(v);
    if (n != null) milestones[k] = n;
  }
  // The gateway-receive mark is the origin of every offset.
  milestones.gateway_receive ??= 0;
  const durations = {} as Record<LatencyStageId, number | null>;
  const spans: RequestLatency["spans"] = [];
  for (const s of LATENCY_STAGES) {
    const dur = num(d[s.id]);
    durations[s.id] = dur;
    const a = milestones[s.from],
      b = milestones[s.to];
    if (dur != null && a != null && b != null && b >= a)
      spans.push({ id: s.id, start: a, end: b });
  }
  return { durations, milestones, spans };
}

export type RecentLatency =
  { kind: "ok"; byId: Map<string, RequestLatency> } | { kind: "unsupported" };

/** null latency (older engine without the field) on every row means the feature is absent. */
export function parseRecentLatency(payload: unknown): RecentLatency {
  if (!isRecord(payload) || !Array.isArray(payload.data))
    return { kind: "unsupported" };
  const byId = new Map<string, RequestLatency>();
  let sawField = false;
  for (const row of payload.data) {
    if (!isRecord(row)) continue;
    if ("latency" in row) sawField = true;
    const lat = parseLatency(row.latency);
    if (lat && typeof row.request_id === "string")
      byId.set(row.request_id, lat);
  }
  if (!sawField && payload.data.length > 0) return { kind: "unsupported" };
  return { kind: "ok", byId };
}

export async function fetchRecentLatency(
  connection: Connection,
  signal?: AbortSignal,
): Promise<RecentLatency> {
  try {
    const body = await requestJson<unknown>(
      connection,
      "/yunshu/requests/recent",
      { signal, search: { limit: "512" } },
    );
    return parseRecentLatency(body);
  } catch (e) {
    if (e instanceof ApiError && [401, 403, 404, 405].includes(e.status ?? 0))
      return { kind: "unsupported" };
    throw e;
  }
}
