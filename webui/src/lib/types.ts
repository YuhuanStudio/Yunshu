/**
 * Backend API contract types for the Yunshu webui.
 *
 * These mirror the shapes returned by the MLX inference engine (proxied via
 * next.config.js: /v1 = OpenAI-compatible, /api/v1 = admin, /health, /realtime).
 * The backend is fixed; keep these in sync with it.
 */

// ---- GPU / memory ---------------------------------------------------------
export interface GpuMemory {
  total_bytes: number;
  active_bytes: number;
  peak_bytes: number;
  cache_bytes: number;
  available_bytes: number;
  utilization_pct: number;
}

// ---- monitoring: engine + system -----------------------------------------
export interface EngineStats {
  model: string | null;
  loaded: boolean;
  running: boolean;
  active_requests: number;
  waiting_requests: number;
  step_counter: number;
  requests_processed: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  uptime_seconds: number;
  gpu_memory: GpuMemory | null;
}

export interface SystemStats {
  cpu_percent: number;
  memory_total_bytes: number;
  memory_used_bytes: number;
  memory_available_bytes: number;
  gpu: GpuMemory;
  python_version: string;
  mlx_version: string;
}

export interface RadixTreeStats {
  enabled: boolean;
  match_hits: number;
  match_total: number;
  total_nodes: number;
  total_blocks: number;
  total_tokens: number;
  total_ref_count: number;
  leaf_count: number;
  max_depth: number;
  eviction_strategy: string;
  eviction_stats?: { total_freed_blocks?: number };
}

export interface HardwareProfile {
  chip_name: string;
  chip_generation: string;
  chip_tier: string;
  total_memory_gb: number;
  working_set_gb: number;
  gpu_cores: string;
  mlx_version: string;
  mlx_lm_version: string;
  adaptive_defaults?: Record<string, unknown>;
  error?: boolean;
}

// ---- models ---------------------------------------------------------------
export interface Model {
  id: string;
  loaded?: boolean;
  size_gb?: number;
  type?: string;
  stats?: Record<string, unknown>;
}

export interface DiscoveredModel {
  model_type: string;
  engine_type: string;
  estimated_size_gb: number;
}

export interface LoRAAdapter {
  adapter_id: string;
  loaded: boolean;
  merged: boolean;
  path?: string;
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
