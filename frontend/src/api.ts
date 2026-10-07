import { failureMessage, statusMessage } from "./errors.ts";
/** Small, typed client for Yunshu's same-origin `/v1` control/status routes. */
export interface Connection {
  /** Server origin or API base, with or without a trailing `/v1`. */
  baseUrl: string;
  /** Sent only as an Authorization bearer token; never persisted here. */
  token: string;
}

export type RequestPhase =
  "queued" | "starting" | "prefill" | "decode" | "running" | string;

export interface RequestRow {
  request_id: string;
  elapsed_s: number;
  phase: RequestPhase;
  queue_position?: number;
  queue_est_wait_ms?: number;
  prompt_tokens?: number;
  cached_tokens?: number;
  processed_tokens?: number;
  percent?: number;
  tokens_per_second?: number | null;
  eta_s?: number | null;
  completion_tokens?: number;
  path?: string;
  stream?: boolean;
  model?: string;
  engine_request_id?: string;
  cancelled?: boolean;
  [key: string]: unknown;
}

export interface EngineModelStatus {
  id: string;
  type: string;
  loaded: boolean;
  loading: boolean;
  pinned: boolean;
  size_gb?: number;
  idle_s?: number | null;
  keep_alive_s?: number | null;
  expires_in_s?: number | null;
  error?: string | null;
}

export interface EngineMemoryStatus {
  active_gb?: number;
  cache_gb?: number;
  peak_gb?: number;
  total_gb?: number;
  pressure?: number;
}

export interface EngineLastRequest {
  request_id: string;
  prompt_tokens: number;
  completion_tokens: number;
  cached_tokens: number;
  prefill_tps: number | null;
  decode_tps: number | null;
  ttft_ms: number | null;
  t: number;
  /** Speculative decoding of that request (x_yunshu.speculative), when it drafted. */
  speculative?: {
    mode?: string;
    acceptance_rate?: number | null;
    rounds?: number;
  } | null;
  [key: string]: unknown;
}

export interface EngineThroughput {
  window_s: number;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  live_decode_tps: number | null;
  mean_prefill_tps: number | null;
  mean_decode_tps: number | null;
}

export interface EngineStatus {
  object: "yunshu.status";
  version: string;
  state: string;
  uptime_s: number;
  load_error: string | null;
  models: EngineModelStatus[];
  memory: EngineMemoryStatus;
  requests: {
    active: number;
    queued: number;
    prefill: number;
    decode: number;
    items: RequestRow[];
  };
  last: EngineLastRequest | null;
  throughput: EngineThroughput;
  gpu?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface WarmupRequest {
  model?: string;
  prompt?: string;
  messages?: readonly Record<string, unknown>[];
  keep_alive?: string | number;
  max_tokens?: number;
}

export interface RequestOptions {
  method?: "GET" | "POST" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
  timeoutMs?: number;
  /** Query parameters, kept apart from the path so the path stays validated. */
  search?: Readonly<Record<string, string>>;
}

export class ApiError extends Error {
  readonly status?: number;
  readonly publicMessage: string;
  readonly detail?: unknown;

  constructor(
    publicMessage: string,
    status?: number,
    detail?: unknown,
    options?: ErrorOptions,
  ) {
    super(publicMessage, options);
    this.name = "ApiError";
    this.status = status;
    this.publicMessage = publicMessage;
    this.detail = detail;
  }
}

const DEFAULT_TIMEOUT_MS = 10_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function apiRoot(connection: Connection): URL {
  let url: URL;
  try {
    url = new URL(connection.baseUrl.trim());
  } catch {
    throw new ApiError("請輸入有效的引擎位址。");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new ApiError("引擎位址必須使用 HTTP 或 HTTPS。");
  }
  if (url.username || url.password)
    throw new ApiError("引擎位址不可包含帳號或密碼。");
  if (url.search || url.hash)
    throw new ApiError("引擎位址不可包含查詢參數或錨點。");

  const path = url.pathname.replace(/\/+$/, "");
  url.pathname = path.endsWith("/v1") ? `${path}/` : `${path}/v1/`;
  url.search = "";
  url.hash = "";
  return url;
}

function buildUrl(connection: Connection, path: string): URL {
  if (
    !path.startsWith("/") ||
    path.startsWith("//") ||
    path.includes("?") ||
    path.includes("#")
  ) {
    throw new ApiError("無效的 API 路徑。");
  }
  const relativePath = path.replace(/^\/+/, "");
  if (relativePath.split("/").some((part) => part === ".." || part === ".")) {
    throw new ApiError("無效的 API 路徑。");
  }
  return new URL(relativePath, apiRoot(connection));
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text.trim())
    throw new ApiError(failureMessage("empty"), response.status);
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new ApiError(failureMessage("json"), response.status);
  }
}

function backendDetail(body: unknown): unknown {
  if (!isRecord(body)) return undefined;
  return body.detail ?? body.message ?? body.error;
}

function publicHttpMessage(
  status: number,
  detail: unknown,
  statusText: string,
): string {
  void detail;
  void statusText;
  return statusMessage(status);
}

/** Fetch and parse one JSON API response with bearer auth, timeout and public errors. */
export async function requestJson<T>(
  connection: Connection,
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const url = buildUrl(connection, path);
  for (const [k, v] of Object.entries(options.search ?? {}))
    url.searchParams.set(k, v);
  const controller = new AbortController();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  let didTimeout = false;
  const timeout = globalThis.setTimeout(() => {
    didTimeout = true;
    controller.abort(new DOMException("Request timed out", "TimeoutError"));
  }, timeoutMs);
  const forwardAbort = () => controller.abort(options.signal?.reason);
  if (options.signal?.aborted) forwardAbort();
  else options.signal?.addEventListener("abort", forwardAbort, { once: true });

  const headers = new Headers({ Accept: "application/json" });
  if (connection.token.trim())
    headers.set("Authorization", `Bearer ${connection.token.trim()}`);
  let body: string | undefined;
  if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.body);
  }

  try {
    const response = await fetch(url, {
      method: options.method ?? "GET",
      headers,
      body,
      signal: controller.signal,
    });
    let payload: unknown;
    try {
      payload = await readJson(response);
    } catch (error) {
      if (!response.ok && error instanceof ApiError) {
        throw new ApiError(
          publicHttpMessage(response.status, undefined, response.statusText),
          response.status,
          undefined,
          { cause: error },
        );
      }
      throw error;
    }
    if (!response.ok) {
      const detail = backendDetail(payload);
      throw new ApiError(
        publicHttpMessage(response.status, detail, response.statusText),
        response.status,
        detail,
      );
    }
    return payload as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (options.signal?.aborted) throw options.signal.reason ?? error;
    if (didTimeout || controller.signal.aborted)
      throw new ApiError(failureMessage("timeout"), undefined, undefined, {
        cause: error,
      });
    throw new ApiError(failureMessage("network"), undefined, undefined, {
      cause: error,
    });
  } finally {
    globalThis.clearTimeout(timeout);
    options.signal?.removeEventListener("abort", forwardAbort);
  }
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string")
    throw new ApiError(failureMessage("field"), undefined, field);
  return value;
}

function requiredNumber(value: unknown, field: string): number {
  if (!isFiniteNumber(value))
    throw new ApiError(failureMessage("field"), undefined, field);
  return value;
}

function nullableNumber(value: unknown, field: string): number | null {
  if (value === null) return null;
  return requiredNumber(value, field);
}

function optionalNumber(
  record: Record<string, unknown>,
  key: string,
  field = key,
): number | null | undefined {
  const value = record[key];
  if (value === undefined) return undefined;
  return nullableNumber(value, field);
}

function validateRequestRow(value: unknown, index: number): RequestRow {
  if (!isRecord(value))
    throw new ApiError(
      `The server returned an invalid request item at index ${index}.`,
    );
  return {
    ...value,
    request_id: requiredString(
      value.request_id,
      `requests.items[${index}].request_id`,
    ),
    elapsed_s: requiredNumber(
      value.elapsed_s,
      `requests.items[${index}].elapsed_s`,
    ),
    phase: requiredString(value.phase, `requests.items[${index}].phase`),
    ...(optionalNumber(value, "queue_position") === undefined
      ? {}
      : {
          queue_position: optionalNumber(
            value,
            "queue_position",
            `requests.items[${index}].queue_position`,
          ) as number,
        }),
    ...(optionalNumber(value, "queue_est_wait_ms") === undefined
      ? {}
      : {
          queue_est_wait_ms: optionalNumber(
            value,
            "queue_est_wait_ms",
            `requests.items[${index}].queue_est_wait_ms`,
          ) as number,
        }),
    ...(optionalNumber(value, "prompt_tokens") === undefined
      ? {}
      : {
          prompt_tokens: optionalNumber(
            value,
            "prompt_tokens",
            `requests.items[${index}].prompt_tokens`,
          ) as number,
        }),
    ...(optionalNumber(value, "cached_tokens") === undefined
      ? {}
      : {
          cached_tokens: optionalNumber(
            value,
            "cached_tokens",
            `requests.items[${index}].cached_tokens`,
          ) as number,
        }),
    ...(optionalNumber(value, "processed_tokens") === undefined
      ? {}
      : {
          processed_tokens: optionalNumber(
            value,
            "processed_tokens",
            `requests.items[${index}].processed_tokens`,
          ) as number,
        }),
    ...(optionalNumber(value, "percent") === undefined
      ? {}
      : {
          percent: optionalNumber(
            value,
            "percent",
            `requests.items[${index}].percent`,
          ) as number,
        }),
    ...(optionalNumber(value, "tokens_per_second") === undefined
      ? {}
      : {
          tokens_per_second: optionalNumber(
            value,
            "tokens_per_second",
            `requests.items[${index}].tokens_per_second`,
          ),
        }),
    ...(optionalNumber(value, "eta_s") === undefined
      ? {}
      : {
          eta_s: optionalNumber(
            value,
            "eta_s",
            `requests.items[${index}].eta_s`,
          ),
        }),
    ...(optionalNumber(value, "completion_tokens") === undefined
      ? {}
      : {
          completion_tokens: optionalNumber(
            value,
            "completion_tokens",
            `requests.items[${index}].completion_tokens`,
          ) as number,
        }),
  } as RequestRow;
}

function validateMemory(value: Record<string, unknown>): EngineMemoryStatus {
  const fields = [
    "active_gb",
    "cache_gb",
    "peak_gb",
    "total_gb",
    "pressure",
  ] as const;
  // Memory figures are display-only: an absent or malformed one is unknown.
  const memory: Record<string, unknown> = { ...value };
  for (const field of fields) {
    if (!isFiniteNumber(value[field])) delete memory[field];
  }
  return memory as EngineMemoryStatus;
}

/** Validate the exact required status envelope; backend additions remain preserved. */
export function parseEngineStatus(value: unknown): EngineStatus {
  if (!isRecord(value) || value.object !== "yunshu.status")
    throw new ApiError(failureMessage("field"));
  if (
    !Array.isArray(value.models) ||
    !isRecord(value.memory) ||
    !isRecord(value.requests) ||
    !isRecord(value.throughput)
  ) {
    throw new ApiError(failureMessage("field"));
  }
  const models = value.models.map((model, index): EngineModelStatus => {
    if (!isRecord(model))
      throw new ApiError(
        `The server returned an invalid model at index ${index}.`,
      );
    if (
      typeof model.loaded !== "boolean" ||
      typeof model.loading !== "boolean" ||
      typeof model.pinned !== "boolean"
    ) {
      throw new ApiError(
        `The server returned invalid model state at index ${index}.`,
      );
    }
    // Optional display fields degrade to "unknown" (a null or malformed value
    // is dropped), so one odd field never reads as an unreachable engine.
    const optional: Record<string, number | null | undefined> = {};
    for (const field of [
      "size_gb",
      "idle_s",
      "keep_alive_s",
      "expires_in_s",
    ] as const) {
      const raw = model[field];
      if (raw === undefined) continue;
      optional[field] = isFiniteNumber(raw)
        ? raw
        : field === "size_gb"
          ? undefined
          : null;
    }
    const modelError = typeof model.error === "string" ? model.error : null;
    return {
      ...model,
      ...optional,
      error: modelError,
      id: requiredString(model.id, `models[${index}].id`),
      type: requiredString(model.type, `models[${index}].type`),
      loaded: model.loaded,
      loading: model.loading,
      pinned: model.pinned,
    } as EngineModelStatus;
  });
  const requests = value.requests;
  if (!Array.isArray(requests.items))
    throw new ApiError(failureMessage("field"));
  const throughput = value.throughput;
  const memory = value.memory;
  const last = value.last;
  const parsedLast =
    last === null
      ? null
      : isRecord(last)
        ? ({
            ...last,
            request_id: requiredString(last.request_id, "last.request_id"),
            prompt_tokens: requiredNumber(
              last.prompt_tokens,
              "last.prompt_tokens",
            ),
            completion_tokens: requiredNumber(
              last.completion_tokens,
              "last.completion_tokens",
            ),
            cached_tokens: requiredNumber(
              last.cached_tokens,
              "last.cached_tokens",
            ),
            prefill_tps: nullableNumber(last.prefill_tps, "last.prefill_tps"),
            decode_tps: nullableNumber(last.decode_tps, "last.decode_tps"),
            ttft_ms: nullableNumber(last.ttft_ms, "last.ttft_ms"),
            t: requiredNumber(last.t, "last.t"),
          } as EngineLastRequest)
        : (() => {
            throw new ApiError(failureMessage("field"));
          })();

  return {
    ...value,
    object: "yunshu.status",
    version: requiredString(value.version, "version"),
    state: requiredString(value.state, "state"),
    uptime_s: requiredNumber(value.uptime_s, "uptime_s"),
    load_error:
      value.load_error === null
        ? null
        : requiredString(value.load_error, "load_error"),
    models,
    memory: validateMemory(memory),
    requests: {
      active: requiredNumber(requests.active, "requests.active"),
      queued: requiredNumber(requests.queued, "requests.queued"),
      prefill: requiredNumber(requests.prefill, "requests.prefill"),
      decode: requiredNumber(requests.decode, "requests.decode"),
      items: requests.items.map(validateRequestRow),
    },
    last: parsedLast,
    throughput: {
      window_s: requiredNumber(throughput.window_s, "throughput.window_s"),
      requests: requiredNumber(throughput.requests, "throughput.requests"),
      prompt_tokens: requiredNumber(
        throughput.prompt_tokens,
        "throughput.prompt_tokens",
      ),
      completion_tokens: requiredNumber(
        throughput.completion_tokens,
        "throughput.completion_tokens",
      ),
      live_decode_tps: nullableNumber(
        throughput.live_decode_tps,
        "throughput.live_decode_tps",
      ),
      mean_prefill_tps: nullableNumber(
        throughput.mean_prefill_tps,
        "throughput.mean_prefill_tps",
      ),
      mean_decode_tps: nullableNumber(
        throughput.mean_decode_tps,
        "throughput.mean_decode_tps",
      ),
    },
    ...(value.gpu === undefined
      ? {}
      : {
          gpu: isRecord(value.gpu)
            ? value.gpu
            : (() => {
                throw new ApiError(failureMessage("field"));
              })(),
        }),
  } as EngineStatus;
}

export async function fetchStatus(
  connection: Connection,
  options: Pick<RequestOptions, "signal" | "timeoutMs"> = {},
): Promise<EngineStatus> {
  return parseEngineStatus(
    await requestJson<unknown>(connection, "/yunshu/status", options),
  );
}

export async function getModel(
  connection: Connection,
  modelId: string,
  options: Pick<RequestOptions, "signal" | "timeoutMs"> = {},
): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(
    connection,
    `/models/${encodeURIComponent(modelId)}`,
    options,
  );
}

export async function loadModel(
  connection: Connection,
  modelId: string,
  options: { pin?: boolean; signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<Record<string, unknown>> {
  return requestJson(connection, "/models/load", {
    method: "POST",
    body: {
      model: modelId,
      ...(options.pin === undefined ? {} : { pin: options.pin }),
    },
    signal: options.signal,
    timeoutMs: options.timeoutMs ?? 180_000,
  });
}

export async function unloadModel(
  connection: Connection,
  modelId: string,
  options: Pick<RequestOptions, "signal" | "timeoutMs"> = {},
): Promise<Record<string, unknown>> {
  return requestJson(
    connection,
    `/models/unload/${encodeURIComponent(modelId)}`,
    {
      method: "POST",
      signal: options.signal,
      timeoutMs: options.timeoutMs ?? 180_000,
    },
  );
}

export async function warmupModel(
  connection: Connection,
  input: WarmupRequest,
  options: Pick<RequestOptions, "signal" | "timeoutMs"> = {},
): Promise<Record<string, unknown>> {
  return requestJson(connection, "/yunshu/warmup", {
    method: "POST",
    body: input,
    signal: options.signal,
    timeoutMs: options.timeoutMs ?? 180_000,
  });
}

export async function cancelRequest(
  connection: Connection,
  requestId: string,
  options: Pick<RequestOptions, "signal" | "timeoutMs"> = {},
): Promise<Record<string, unknown>> {
  return requestJson(connection, `/requests/${encodeURIComponent(requestId)}`, {
    method: "DELETE",
    signal: options.signal,
    timeoutMs: options.timeoutMs ?? 180_000,
  });
}
