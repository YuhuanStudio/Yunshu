import type { Connection } from "./api.ts";
import { t } from "./i18n/index.ts";

/** Typed client for the admin routes under /v1/yunshu (config writes, service, CORS). */
export class AdminError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: Record<string, unknown>;
  constructor(
    status: number,
    code: string,
    message: string,
    detail: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "AdminError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
  /** The server does not have this route (an older engine). */
  get unsupported() {
    return this.status === 404 || this.status === 405;
  }
  get denied() {
    return this.status === 401 || this.status === 403;
  }
}

export const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

export function adminUrl(connection: Connection, path: string): string {
  const root = connection.baseUrl
    .trim()
    .replace(/\/+$/, "")
    .replace(/\/v1$/i, "");
  return `${root}/v1${path}`;
}

export function openapiUrl(connection: Connection): string {
  const root = connection.baseUrl
    .trim()
    .replace(/\/+$/, "")
    .replace(/\/v1$/i, "");
  return `${root}/openapi.json`;
}

/** Public, translated message for a failed admin call. Server wording never reaches the UI. */
function publicMessage(status: number): string {
  if (status === 0) return t("settings.admin.error.network");
  if (status === 401 || status === 403) return t("settings.admin.error.denied");
  if (status === 404 || status === 405)
    return t("settings.admin.error.unsupported");
  if (status === 409) return t("settings.admin.error.conflict");
  if (status === 422 || status === 400)
    return t("settings.admin.error.invalid");
  return t("settings.admin.error.server", { status });
}

export async function adminRequest<T>(
  connection: Connection,
  method: "GET" | "POST" | "PATCH" | "DELETE",
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const headers = new Headers({ Accept: "application/json" });
  if (connection.token.trim())
    headers.set("Authorization", `Bearer ${connection.token.trim()}`);
  if (body !== undefined) headers.set("Content-Type", "application/json");
  let response: Response;
  try {
    response = await fetch(adminUrl(connection, path), {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new AdminError(0, "network", publicMessage(0));
  }
  let payload: unknown = null;
  try {
    const text = await response.text();
    payload = text.trim() ? JSON.parse(text) : null;
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const raw = isRecord(payload) ? payload.detail : undefined;
    const detail = isRecord(raw) ? raw : {};
    throw new AdminError(
      response.status,
      typeof detail.code === "string" ? detail.code : "",
      publicMessage(response.status),
      detail,
    );
  }
  return payload as T;
}

// ── settings writes ─────────────────────────────────────────────────

export type PatchStatus =
  "applied" | "needs_reload" | "needs_restart" | "overridden";

export interface PatchResultRow {
  status: PatchStatus;
  applies: string;
  source: string;
  reset: boolean;
  note?: string;
}

export interface RestartHint {
  available: boolean;
  cli: string | null;
  manual: string | null;
}

export interface PatchResult {
  dryRun: boolean;
  results: Record<string, PatchResultRow>;
  restartRequired: boolean;
  reloadRequired: boolean;
  restart: RestartHint | null;
}

const STATUSES: PatchStatus[] = [
  "applied",
  "needs_reload",
  "needs_restart",
  "overridden",
];

export function parsePatchResult(payload: unknown): PatchResult | null {
  if (!isRecord(payload) || !isRecord(payload.results)) return null;
  const results: Record<string, PatchResultRow> = {};
  for (const [name, row] of Object.entries(payload.results)) {
    if (!isRecord(row)) continue;
    const status = STATUSES.find((s) => s === row.status);
    if (!status) continue;
    results[name] = {
      status,
      applies: typeof row.applies === "string" ? row.applies : "",
      source: typeof row.source === "string" ? row.source : "",
      reset: row.reset === true,
      note: typeof row.note === "string" ? row.note : undefined,
    };
  }
  const restart = isRecord(payload.restart)
    ? {
        available: payload.restart.available === true,
        cli:
          typeof payload.restart.cli === "string" ? payload.restart.cli : null,
        manual:
          typeof payload.restart.manual === "string"
            ? payload.restart.manual
            : null,
      }
    : null;
  return {
    dryRun: payload.dry_run === true,
    results,
    restartRequired: payload.restart_required === true,
    reloadRequired: payload.reload_required === true,
    restart,
  };
}

export async function patchConfig(
  connection: Connection,
  settings: Record<string, unknown>,
  options: { dryRun?: boolean; confirmExperimental?: boolean } = {},
): Promise<PatchResult> {
  const payload = await adminRequest<unknown>(
    connection,
    "PATCH",
    "/yunshu/config",
    {
      settings,
      dry_run: options.dryRun === true,
      confirm_experimental: options.confirmExperimental === true,
    },
  );
  const parsed = parsePatchResult(payload);
  if (!parsed)
    throw new AdminError(200, "shape", t("settings.admin.error.shape"));
  return parsed;
}

/** The 422 per-name error map of a rejected PATCH ({} when the failure is not a validation error). */
export function validationErrors(error: unknown): Record<string, string> {
  if (!(error instanceof AdminError) || !isRecord(error.detail.errors))
    return {};
  const out: Record<string, string> = {};
  for (const [name, message] of Object.entries(error.detail.errors))
    if (typeof message === "string") out[name] = message;
  return out;
}

export interface Summary {
  applied: string[];
  needsReload: string[];
  needsRestart: string[];
  overridden: { name: string; source: string }[];
}

export function summarizePatch(result: PatchResult): Summary {
  const out: Summary = {
    applied: [],
    needsReload: [],
    needsRestart: [],
    overridden: [],
  };
  for (const [name, row] of Object.entries(result.results)) {
    if (row.status === "applied") out.applied.push(name);
    else if (row.status === "needs_reload") out.needsReload.push(name);
    else if (row.status === "needs_restart") out.needsRestart.push(name);
    else out.overridden.push({ name, source: row.source });
  }
  return out;
}

// ── service ─────────────────────────────────────────────────────────

export interface ServiceInfo {
  label: string;
  plist: string;
  installed: boolean;
  loaded: boolean;
  pid: number | null;
  state: string | null;
  lastExitCode: number | null;
  log: string;
  underLaunchd: boolean;
  version: string;
  uptimeS: number | null;
  restartNote: string | null;
  cli: { status: string; restart: string; install: string };
}

const str = (v: unknown, fallback = "") =>
  typeof v === "string" ? v : fallback;
const num = (v: unknown) =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

export function parseService(payload: unknown): ServiceInfo | null {
  if (!isRecord(payload) || typeof payload.label !== "string") return null;
  const cli = isRecord(payload.cli) ? payload.cli : {};
  return {
    label: payload.label,
    plist: str(payload.plist),
    installed: payload.installed === true,
    loaded: payload.loaded === true,
    pid: num(payload.pid),
    state: typeof payload.state === "string" ? payload.state : null,
    lastExitCode: num(payload.last_exit_code),
    log: str(payload.log),
    underLaunchd: payload.under_launchd === true,
    version: str(payload.version),
    uptimeS: num(payload.uptime_s),
    restartNote:
      typeof payload.restart_note === "string" ? payload.restart_note : null,
    cli: {
      status: str(cli.status, "yunshu service status"),
      restart: str(cli.restart, "yunshu service restart"),
      install: str(cli.install, "yunshu service install --model <model>"),
    },
  };
}

export async function getService(
  connection: Connection,
  signal?: AbortSignal,
): Promise<ServiceInfo> {
  const parsed = parseService(
    await adminRequest<unknown>(
      connection,
      "GET",
      "/yunshu/service",
      undefined,
      signal,
    ),
  );
  if (!parsed)
    throw new AdminError(200, "shape", t("settings.admin.error.shape"));
  return parsed;
}

export interface RestartAccepted {
  activeRequests: number;
  drainTimeoutS: number | null;
}

export async function restartService(
  connection: Connection,
): Promise<RestartAccepted> {
  const payload = await adminRequest<unknown>(
    connection,
    "POST",
    "/yunshu/service/restart",
    { confirm: true },
  );
  const p = isRecord(payload) ? payload : {};
  return {
    activeRequests: num(p.active_requests) ?? 0,
    drainTimeoutS: num(p.drain_timeout_s),
  };
}

/** The manual restart command out of a 409 not_under_launchd error. */
export function manualCommand(error: unknown): string | null {
  if (!(error instanceof AdminError)) return null;
  const m = error.detail.manual;
  return typeof m === "string" ? m : null;
}

// ── CORS ────────────────────────────────────────────────────────────

export interface CorsInfo {
  origins: string[];
  wildcard: boolean;
  credentials: boolean;
  source: string;
  default: string;
  warnings: string[];
  requestOrigin: string | null;
  requestOriginAllowed: boolean | null;
}

export function parseCors(payload: unknown): CorsInfo | null {
  if (!isRecord(payload) || !Array.isArray(payload.origins)) return null;
  return {
    origins: payload.origins.filter((o): o is string => typeof o === "string"),
    wildcard: payload.wildcard === true,
    credentials: payload.credentials === true,
    source: str(payload.source, "default"),
    default: str(payload.default),
    warnings: Array.isArray(payload.warnings)
      ? payload.warnings.filter((w): w is string => typeof w === "string")
      : [],
    requestOrigin:
      typeof payload.request_origin === "string"
        ? payload.request_origin
        : null,
    requestOriginAllowed:
      typeof payload.request_origin_allowed === "boolean"
        ? payload.request_origin_allowed
        : null,
  };
}

export async function getCors(
  connection: Connection,
  signal?: AbortSignal,
): Promise<CorsInfo> {
  const parsed = parseCors(
    await adminRequest<unknown>(
      connection,
      "GET",
      "/yunshu/cors",
      undefined,
      signal,
    ),
  );
  if (!parsed)
    throw new AdminError(200, "shape", t("settings.admin.error.shape"));
  return parsed;
}

/** `origins: null` resets to the default. */
export async function patchCors(
  connection: Connection,
  origins: string[] | null,
  allowAnyOrigin = false,
): Promise<CorsInfo> {
  const parsed = parseCors(
    await adminRequest<unknown>(connection, "PATCH", "/yunshu/cors", {
      origins,
      allow_any_origin: allowAnyOrigin,
    }),
  );
  if (!parsed)
    throw new AdminError(200, "shape", t("settings.admin.error.shape"));
  return parsed;
}

/**
 * Client-side origin check, mirroring the server: `scheme://host[:port]` only, or a lone `*`.
 * Returns the normalised origin, or null when it is not valid.
 */
export function normaliseOrigin(raw: string): string | null {
  const o = raw.trim();
  if (o === "*") return "*";
  try {
    const u = new URL(o);
    if (u.protocol !== "http:" && u.protocol !== "https:") return null;
    if (u.username || u.password || u.search || u.hash) return null;
    if (u.pathname !== "/" && u.pathname !== "") return null;
    if (!u.hostname) return null;
    return u.origin;
  } catch {
    return null;
  }
}

/** Locate POST /v1/yunshu/models/{id}/reload in an OpenAPI document, if the engine has it. */
export function findReloadRoute(openapi: unknown): boolean {
  if (!isRecord(openapi) || !isRecord(openapi.paths)) return false;
  return Object.entries(openapi.paths).some(
    ([path, ops]) =>
      /^\/v1\/yunshu\/models\/\{[^}]+\}\/reload$/.test(path) &&
      isRecord(ops) &&
      "post" in ops,
  );
}

export async function reloadModel(
  connection: Connection,
  modelId: string,
): Promise<void> {
  await adminRequest<unknown>(
    connection,
    "POST",
    `/yunshu/models/${encodeURIComponent(modelId)}/reload`,
    {},
  );
}
