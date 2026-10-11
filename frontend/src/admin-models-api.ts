import { ApiError, requestJson, type Connection } from "./api.ts";

/** Returned by every call below on a server that predates the route (404/405). */
export const UNSUPPORTED = "unsupported" as const;
export type Unsupported = typeof UNSUPPORTED;

const rec = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const strs = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];

export const isUnsupported = (e: unknown): boolean =>
  e instanceof ApiError && (e.status === 404 || e.status === 405);

/** Run a call; a missing route becomes "unsupported" instead of an error. */
export async function orUnsupported<T>(
  call: () => Promise<T>,
): Promise<T | Unsupported> {
  try {
    return await call();
  } catch (e) {
    if (isUnsupported(e)) return UNSUPPORTED;
    throw e;
  }
}

// ── downloads ──────────────────────────────────────────────────────────

export type DownloadState =
  "queued" | "running" | "done" | "failed" | "cancelled";

export type DownloadJob = {
  id: string;
  repo: string;
  revision: string | null;
  patterns: string[];
  state: DownloadState;
  error: string | null;
  path: string | null;
  bytesTotal: number | null;
  bytesDone: number | null;
  filesTotal: number | null;
  filesDone: number | null;
  activeFiles: string[];
  rateBps: number | null;
  etaS: number | null;
  created: number | null;
  started: number | null;
  finished: number | null;
  registered: boolean;
  alreadyPresent: boolean;
};

const STATES: readonly string[] = [
  "queued",
  "running",
  "done",
  "failed",
  "cancelled",
];
export const isActive = (j: Pick<DownloadJob, "state">) =>
  j.state === "queued" || j.state === "running";

export function parseJob(raw: unknown): DownloadJob {
  const r = rec(raw);
  const state = str(r.state) ?? "";
  return {
    id: str(r.id) ?? "",
    repo: str(r.repo) ?? "",
    revision: str(r.revision),
    patterns: strs(r.allow_patterns),
    state: (STATES.includes(state) ? state : "failed") as DownloadState,
    error: str(r.error),
    path: str(r.path),
    bytesTotal: num(r.bytes_total),
    bytesDone: num(r.bytes_done),
    filesTotal: num(r.files_total),
    filesDone: num(r.files_done),
    activeFiles: strs(r.active_files),
    rateBps: num(r.rate_bps),
    etaS: num(r.eta_s),
    created: num(r.created),
    started: num(r.started),
    finished: num(r.finished),
    registered: r.registered === true,
    alreadyPresent: r.already_present === true,
  };
}

export type DownloadList = {
  jobs: DownloadJob[];
  active: number;
  freeBytes: number | null;
  modelsDir: string | null;
};

export function parseDownloads(raw: unknown): DownloadList {
  const r = rec(raw);
  const jobs = (Array.isArray(r.downloads) ? r.downloads : []).map(parseJob);
  return {
    jobs,
    active: num(r.active) ?? jobs.filter(isActive).length,
    freeBytes: num(r.free_bytes),
    modelsDir: str(r.models_dir),
  };
}

export type DownloadRequest = {
  repo: string;
  revision?: string;
  allow_patterns?: string[];
};

export const listDownloads = (c: Connection, signal?: AbortSignal) =>
  orUnsupported(async () =>
    parseDownloads(
      await requestJson<unknown>(c, "/yunshu/downloads", { signal }),
    ),
  );

export const startDownload = (c: Connection, body: DownloadRequest) =>
  requestJson<unknown>(c, "/yunshu/downloads", {
    method: "POST",
    body,
    timeoutMs: 60_000, // the hub is asked for the file list before the 202
  }).then(parseJob);

export const cancelDownload = (c: Connection, id: string) =>
  requestJson<unknown>(c, `/yunshu/downloads/${encodeURIComponent(id)}`, {
    method: "DELETE",
  }).then(parseJob);

/** The 507 body: how much was needed, how much is free, and where. */
export type DiskShortfall = {
  needed: number | null;
  free: number | null;
  path: string | null;
};
export function diskShortfall(e: unknown): DiskShortfall | null {
  if (!(e instanceof ApiError) || e.status !== 507) return null;
  const d = rec(e.detail);
  return {
    needed: num(d.needed_bytes),
    free: num(d.free_bytes),
    path: str(d.path),
  };
}

/** Parse `a, b` or one-per-line glob patterns; empty input means the whole repo. */
export function parsePatterns(text: string): string[] {
  return text
    .split(/[\n,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export const REPO_RE = /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/;

// ── local inventory ────────────────────────────────────────────────────

export type LocalModel = {
  id: string;
  path: string;
  source: string | null;
  sizeBytes: number | null;
  modelType: string | null;
  kind: string | null;
  architecture: string | null;
  parameters: string | null;
  quantBits: number | null;
  contextLength: number | null;
  capabilities: string[];
  complete: boolean;
  completeReason: string | null;
  registeredAs: string | null;
  loaded: boolean;
};

export type LocalInventory = {
  models: LocalModel[];
  totalBytes: number | null;
  modelsDir: string | null;
  freeBytes: number | null;
};

export function parseLocal(raw: unknown): LocalInventory {
  const r = rec(raw);
  return {
    models: (Array.isArray(r.models) ? r.models : []).map((m) => {
      const x = rec(m);
      return {
        id: str(x.id) ?? str(x.path) ?? "",
        path: str(x.path) ?? "",
        source: str(x.source),
        sizeBytes: num(x.size_bytes),
        modelType: str(x.model_type),
        kind: str(x.kind),
        architecture: str(x.architecture),
        parameters: typeof x.parameters === "string" ? x.parameters : null,
        quantBits: num(rec(x.quantization).bits),
        contextLength: num(x.context_length),
        capabilities: strs(x.capabilities),
        complete: x.complete === true,
        completeReason: str(x.complete_reason),
        registeredAs: str(x.registered_as),
        loaded: x.loaded === true,
      };
    }),
    totalBytes: num(r.total_bytes),
    modelsDir: str(r.models_dir),
    freeBytes: num(r.free_bytes),
  };
}

export const listLocalModels = (
  c: Connection,
  refresh = false,
  signal?: AbortSignal,
) =>
  orUnsupported(async () =>
    parseLocal(
      await requestJson<unknown>(c, "/yunshu/models/local", {
        signal,
        timeoutMs: 30_000,
        search: refresh ? { refresh: "true" } : undefined,
      }),
    ),
  );

// ── fit ────────────────────────────────────────────────────────────────

export type FitVerdict = "fits" | "tight" | "wont_fit";
export type FitResult = {
  model: string;
  verdict: FitVerdict;
  reason: string | null;
  weightsBytes: number | null;
  kvReserveBytes: number | null;
  neededBytes: number | null;
  budgetBytes: number | null;
  usedBytes: number | null;
  freeBytes: number | null;
  freeAfterEvictBytes: number | null;
  wouldEvict: string[];
  loaded: boolean;
  estimated: boolean;
};

export function parseFit(raw: unknown): FitResult {
  const r = rec(raw);
  const v = str(r.verdict);
  return {
    model: str(r.model) ?? "",
    verdict: v === "tight" || v === "wont_fit" ? v : "fits",
    reason: str(r.reason),
    weightsBytes: num(r.weights_bytes),
    kvReserveBytes: num(r.kv_reserve_bytes),
    neededBytes: num(r.needed_bytes),
    budgetBytes: num(r.budget_bytes),
    usedBytes: num(r.used_bytes),
    freeBytes: num(r.free_bytes),
    freeAfterEvictBytes: num(r.free_bytes_after_evict),
    wouldEvict: strs(r.would_evict),
    loaded: r.loaded === true,
    estimated: rec(r.basis).estimated !== false,
  };
}

/** The dry-run load check. "unsupported" on an older server or a single-model server (400/404). */
export const getFit = (c: Connection, id: string, signal?: AbortSignal) =>
  orUnsupported(async () => {
    try {
      return parseFit(
        await requestJson<unknown>(
          c,
          `/yunshu/models/${encodeURIComponent(id)}/fit`,
          { signal },
        ),
      );
    } catch (e) {
      // 400 = single-model server without a model manager: nothing to check against.
      if (e instanceof ApiError && e.status === 400)
        throw new ApiError("", 404);
      throw e;
    }
  });
