/**
 * Backend API contract types for the Yunshu webui.
 *
 * These mirror the shapes returned by the MLX inference engine (proxied via
 * next.config.js: /v1 = OpenAI-compatible, /api/v1 = admin, /health, /realtime).
 * The backend is fixed; keep these in sync with it.
 */

// ---- monitoring: GPU (nested under system.gpu) ---------------------------
// Apple-silicon UMA: there is one unified pool (`total_uma_bytes`); "free" is
// derived as total − active − cache. There is no separate `total_bytes` /
// `available_bytes` any more.
export interface GpuStats {
  active_bytes: number;
  peak_bytes: number;
  cache_bytes: number;
  total_uma_bytes: number;
  utilization_pct: number;
  mlx_version: string;
}

// ---- monitoring: system  (GET /api/v1/gw/monitoring/system) --------------
export interface SystemStats {
  cpu: { percent: number; physical_cores: number; logical_cores: number };
  memory: { total_bytes: number; used_bytes: number; available_bytes: number; percent: number };
  gpu: GpuStats;
  platform: string;
  python_version: string;
  pid: number;
  hostname: string;
  compute_utilization_pct?: number;
}

// ---- monitoring: engine  (GET /api/v1/gw/monitoring/engine) --------------
// NB: no uptime and no GPU here — uptime comes from /health, GPU from /system.
export interface EngineStats {
  loaded: boolean;
  model: string | null;
  engines: { model_id: string; stats?: Record<string, unknown> }[];
  active_requests: number;
  waiting_requests: number;
  step_counter: number;
  requests_processed: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
}

// ---- monitoring: models  (GET /api/v1/gw/monitoring/models) --------------
export interface MonitoringModel {
  model_id: string;
  loaded: boolean;
  pinned: boolean;
  size_bytes: number;
  stats?: Record<string, unknown>;
}

// ---- monitoring: requests  (GET /api/v1/gw/monitoring/requests) ----------
export interface RequestsStats {
  active: number;
  waiting: number;
  total_processed: number;
  last_minute: number;
  latency_percentiles?: Record<string, number>;
  token_percentiles?: Record<string, number>;
  endpoint_breakdown?: Record<string, unknown>;
  itl?: Record<string, unknown>;
}

// ---- monitoring: radix tree  (GET /api/v1/gw/monitoring/radix-tree) ------
export interface RadixTreeStats {
  models: { model_id: string; [k: string]: unknown }[];
}

// ---- health / version (no prefix) ----------------------------------------
/**
 * `/health` is the one endpoint that anything in front of the engine might
 * answer instead of the engine itself — a gateway, a load balancer, a stale
 * service on a mis-set `YUNSHU_BACKEND_URL`. Every field past `status` is
 * therefore optional, so TypeScript forces the guard at each use site.
 *
 * It was not, and `/settings` read `health.engine.loaded` behind a plain
 * `health ? …` null check. A backend answering `{"status":"ok"}` — a perfectly
 * ordinary health response — white-screened the entire page, on the one screen
 * a user opens to find out why the backend is unreachable.
 */
export interface HealthStatus {
  status: string;
  engine?: { loaded?: boolean };
  server_state?: string;
  uptime_seconds?: number;
  sleep?: unknown;
}

export interface VersionInfo {
  version: string;
  service: string;
  description: string;
}

// ---- models  (GET /v1/models) ---------------------------------------------
// `type` / `size_gb` / `loaded` / `stats` only appear WITH a valid token; an
// anonymous caller sees only `id` (+ the OpenAI envelope fields).
export interface Model {
  id: string;
  object?: string;
  created?: number;
  owned_by?: string;
  loaded?: boolean;
  size_gb?: number;
  type?: string;
  stats?: Record<string, unknown>;
}

// ---- chat / completions ---------------------------------------------------
export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export type MessageRole = "user" | "assistant" | "system" | "tool";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  imageUrl?: string;
  /** Attached images as data-URIs (vision / VLM messages). Sent as image_url parts. */
  images?: string[];
  reasoning?: string;
  thinking?: boolean;
  tokens?: number;
  latencyMs?: number;
  logprobs?: { tokens: string[]; token_logprobs: number[] };
  toolCalls?: ToolCall[];
  toolCallId?: string;
  streaming?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  model: string;
  createdAt: number;
  updatedAt: number;
}

export interface CompletionResult {
  id: string;
  choices: {
    text: string;
    index: number;
    finish_reason: string | null;
    logprobs: Record<string, unknown> | null;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// ---- tokenize -------------------------------------------------------------
export interface TokenizeResult {
  tokens: number[];
  model: string;
}
export interface DetokenizeResult {
  text: string;
  model: string;
}
export interface TokenCountResult {
  token_count: number;
  over_context_limit: boolean;
}

// ---- embeddings -----------------------------------------------------------
export interface EmbeddingResult {
  data: { index: number; embedding: number[] }[];
  usage: { total_tokens: number };
}

// ---- mcp ------------------------------------------------------------------
export interface MCPServer {
  name: string;
  status: string;
  tool_count?: number;
  url?: string;
}
export interface MCPTool {
  name: string;
  description?: string;
  server?: string;
  inputSchema?: Record<string, unknown>;
}

// ---- realtime -------------------------------------------------------------
export interface WsMessage {
  id: string;
  direction: "send" | "recv";
  type: string;
  data: unknown;
  timestamp: number;
}

// ---- batch ----------------------------------------------------------------
export interface BatchItem {
  id: string;
  input: string;
  output?: string;
  status: "pending" | "running" | "completed" | "error";
  error?: string;
  tokens?: number;
}

// ---- monitoring SLO -------------------------------------------------------
export interface SLOAlert {
  id: string;
  metric: string;
  threshold: number;
  current: number;
  unit: string;
  severity: "critical" | "warning" | "info";
  message: string;
}
export interface SLOConfig {
  ttft_p95_ms: number;
  tps_min: number;
  gpu_memory_max_pct: number;
  error_rate_max_pct: number;
}
