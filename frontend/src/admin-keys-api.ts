import type { Connection } from "./api.ts";
import { AdminError, adminRequest, isRecord } from "./admin-settings-api.ts";
import { t } from "./i18n/index.ts";

export const SCOPES = ["infer", "admin"] as const;
export type Scope = (typeof SCOPES)[number];
export const QUOTA_FIELDS = [
  "requests_per_day",
  "tokens_per_day",
  "max_concurrent",
] as const;
export type QuotaField = (typeof QUOTA_FIELDS)[number];
export type Quotas = Record<QuotaField, number | null>;

export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  created: number;
  enabled: boolean;
  scopes: Scope[];
  /** Epoch seconds, or null for no expiry. */
  expires: number | null;
  expired: boolean;
  quotas: Quotas;
  lastUsed: number | null;
  /** Rolling 24 h window counters. */
  window: { requests: number; tokens: number; inflight: number };
}

export interface UsageDay {
  key: string;
  name: string;
  day: string;
  requests: number;
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
  errors: number;
}

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

export function parseKey(raw: unknown): ApiKey | null {
  if (!isRecord(raw) || typeof raw.id !== "string") return null;
  const quotas = isRecord(raw.quotas) ? raw.quotas : {};
  const window = isRecord(raw.window) ? raw.window : {};
  return {
    id: raw.id,
    name: typeof raw.name === "string" ? raw.name : raw.id,
    prefix: typeof raw.prefix === "string" ? raw.prefix : "",
    created: num(raw.created) ?? 0,
    enabled: raw.enabled !== false,
    scopes: Array.isArray(raw.scopes)
      ? raw.scopes.filter((s): s is Scope =>
          (SCOPES as readonly unknown[]).includes(s),
        )
      : ["infer"],
    expires: num(raw.expires),
    expired: raw.expired === true,
    quotas: {
      requests_per_day: num(quotas.requests_per_day),
      tokens_per_day: num(quotas.tokens_per_day),
      max_concurrent: num(quotas.max_concurrent),
    },
    lastUsed: num(raw.last_used),
    window: {
      requests: num(window.requests) ?? 0,
      tokens: num(window.tokens) ?? 0,
      inflight: num(window.inflight) ?? 0,
    },
  };
}

export function parseKeyList(payload: unknown): ApiKey[] | null {
  if (!isRecord(payload) || !Array.isArray(payload.data)) return null;
  return payload.data.flatMap((k) => {
    const parsed = parseKey(k);
    return parsed ? [parsed] : [];
  });
}

export function parseUsage(payload: unknown): UsageDay[] {
  if (!isRecord(payload) || !Array.isArray(payload.data)) return [];
  const out: UsageDay[] = [];
  for (const r of payload.data) {
    if (!isRecord(r) || typeof r.day !== "string" || typeof r.key !== "string")
      continue;
    out.push({
      key: r.key,
      name: typeof r.name === "string" ? r.name : r.key,
      day: r.day,
      requests: num(r.requests) ?? 0,
      promptTokens: num(r.prompt_tokens) ?? 0,
      completionTokens: num(r.completion_tokens) ?? 0,
      cachedTokens: num(r.cached_tokens) ?? 0,
      errors: num(r.errors) ?? 0,
    });
  }
  return out;
}

export async function listKeys(
  c: Connection,
  signal?: AbortSignal,
): Promise<ApiKey[]> {
  const keys = parseKeyList(
    await adminRequest<unknown>(c, "GET", "/yunshu/keys", undefined, signal),
  );
  if (!keys)
    throw new AdminError(200, "shape", t("settings.admin.error.shape"));
  return keys;
}

export async function getUsage(
  c: Connection,
  days: number,
  signal?: AbortSignal,
): Promise<UsageDay[]> {
  return parseUsage(
    await adminRequest<unknown>(
      c,
      "GET",
      `/yunshu/usage?since=${days}d&group=day`,
      undefined,
      signal,
    ),
  );
}

export interface KeyDraft {
  name: string;
  scopes: Scope[];
  quotas: Quotas;
  /** Epoch seconds; null = never. */
  expires: number | null;
}

export interface CreatedKey {
  key: ApiKey;
  /** Shown exactly once; the server stores only its hash. */
  secret: string;
}

function parseCreated(payload: unknown): CreatedKey {
  const key = parseKey(payload);
  const secret =
    isRecord(payload) && typeof payload.secret === "string"
      ? payload.secret
      : "";
  if (!key || !secret)
    throw new AdminError(200, "shape", t("settings.admin.error.shape"));
  return { key, secret };
}

/** 0 or empty means unlimited and is sent as null. */
export const cleanQuotas = (q: Quotas): Quotas => ({
  requests_per_day: q.requests_per_day || null,
  tokens_per_day: q.tokens_per_day || null,
  max_concurrent: q.max_concurrent || null,
});

export async function createKey(
  c: Connection,
  d: KeyDraft,
): Promise<CreatedKey> {
  return parseCreated(
    await adminRequest<unknown>(c, "POST", "/yunshu/keys", {
      name: d.name.trim(),
      scopes: d.scopes,
      quotas: cleanQuotas(d.quotas),
      expires: d.expires,
    }),
  );
}

export async function rotateKey(
  c: Connection,
  id: string,
): Promise<CreatedKey> {
  return parseCreated(
    await adminRequest<unknown>(
      c,
      "POST",
      `/yunshu/keys/${encodeURIComponent(id)}/rotate`,
      {},
    ),
  );
}

export async function patchKey(
  c: Connection,
  id: string,
  patch: Partial<{
    name: string;
    enabled: boolean;
    scopes: Scope[];
    quotas: Quotas;
    expires: number | null;
  }>,
): Promise<ApiKey> {
  const body = {
    ...patch,
    ...(patch.quotas ? { quotas: cleanQuotas(patch.quotas) } : {}),
  };
  const key = parseKey(
    await adminRequest<unknown>(
      c,
      "PATCH",
      `/yunshu/keys/${encodeURIComponent(id)}`,
      body,
    ),
  );
  if (!key) throw new AdminError(200, "shape", t("settings.admin.error.shape"));
  return key;
}

export async function deleteKey(c: Connection, id: string): Promise<void> {
  await adminRequest<unknown>(
    c,
    "DELETE",
    `/yunshu/keys/${encodeURIComponent(id)}`,
  );
}

/** Today's usage of a key in UTC (the server buckets days in UTC). */
export const utcDay = (now: number) => new Date(now).toISOString().slice(0, 10);

export function todayUsage(
  usage: readonly UsageDay[],
  keyId: string,
  now = Date.now(),
): { requests: number; tokens: number } {
  const day = utcDay(now);
  let requests = 0,
    tokens = 0;
  for (const r of usage)
    if (r.key === keyId && r.day === day) {
      requests += r.requests;
      tokens += r.promptTokens + r.completionTokens;
    }
  return { requests, tokens };
}

/** Quota fill, 0..1+; null when the quota is unlimited. */
export function quotaFill(used: number, quota: number | null): number | null {
  return quota ? used / quota : null;
}

export interface DayPoint {
  day: string;
  requests: number;
  tokens: number;
}

/** One zero-filled point per UTC day for the last `days` days, for one key (or all when null). */
export function dailySeries(
  usage: readonly UsageDay[],
  keyId: string | null,
  days: number,
  now = Date.now(),
): DayPoint[] {
  const byDay = new Map<string, DayPoint>();
  for (let i = days - 1; i >= 0; i--) {
    const day = utcDay(now - i * 86_400_000);
    byDay.set(day, { day, requests: 0, tokens: 0 });
  }
  for (const r of usage) {
    if (keyId && r.key !== keyId) continue;
    const p = byDay.get(r.day);
    if (!p) continue;
    p.requests += r.requests;
    p.tokens += r.promptTokens + r.completionTokens;
  }
  return [...byDay.values()];
}

/** A datetime-local value (local time) to epoch seconds, or null when empty / invalid. */
export function localToEpoch(value: string): number | null {
  if (!value) return null;
  const ms = new Date(value).getTime();
  return Number.isFinite(ms) ? Math.floor(ms / 1000) : null;
}

export function epochToLocal(epoch: number | null): string {
  if (epoch == null) return "";
  const d = new Date(epoch * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}
