// Shared fixture engine for the interactive-state sweep (idle engine, 128 GiB machine, all admin routes).
const GB = 2 ** 30;
const T = Math.floor(Date.now() / 1000);
export const T0 = 1_800_000_000;
export const statusBody = (o = {}) => ({
  object: "yunshu.status", version: "0.1.5", state: "running", uptime_s: 8100, load_error: null,
  models: [
    { id: "mlx-community/Qwen3.8-27B-oQ4e-mtp", type: "VLMEngine", loaded: true, loading: false, pinned: true, size_gb: 16.9, idle_s: 0 },
    { id: "mlx-community/Qwen3.5-9B-4bit", type: "LLM", loaded: false, loading: false, pinned: false, size_gb: 5.8 },
    { id: "org/Llama-3.2-3B", type: "LLM", loaded: false, loading: false, pinned: false, size_gb: 2.1 },
  ],
  memory: { active_gb: 24.9, cache_gb: 2.25, peak_gb: 34.2, total_gb: 137.438953472 },
  requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
  last: { request_id: "req_25ab1b3f793ab", prompt_tokens: 53, completion_tokens: 300, cached_tokens: 52, prefill_tps: 954.8, decode_tps: 35.1, ttft_ms: 344, t: T - 60 },
  throughput: { window_s: 60, requests: 1, prompt_tokens: 53, completion_tokens: 300, live_decode_tps: null, mean_prefill_tps: 900, mean_decode_tps: 35.1 },
  ...o,
});
const key = (id, name, over = {}) => ({ id, name, prefix: "ysk-" + id.slice(-4), created: 1790000000, enabled: true, scopes: ["infer"], expires: null, expired: false,
  quotas: { requests_per_day: 1000, tokens_per_day: null, max_concurrent: 2 }, last_used: T - 120, window: { requests: 640, tokens: 120000, inflight: 0 }, ...over });
const setting = (o) => ({ name: "X", value: 1, default: 1, type: "int", stability: "stable", applies: "live", source: "default", secret: false, minimum: null, description: "", ...o });
const CONFIG = [
  setting({ name: "YUNSHU_LOG_LEVEL", value: "INFO", default: "INFO", type: "enum", choices: ["DEBUG", "INFO", "WARNING"], description: "Log level." }),
  setting({ name: "YUNSHU_QUEUE_LIMIT", value: 64, default: 64, minimum: 0, description: "Requests in flight at once." }),
  setting({ name: "YUNSHU_PREFIX_MAX_ENTRIES", value: 64, default: 64, minimum: 1, applies: "reload", description: "Prefix cache entries." }),
  setting({ name: "YUNSHU_KEEP_ALIVE_TIMEOUT", value: 5, default: 5, applies: "restart", source: "env", description: "Idle HTTP seconds." }),
  setting({ name: "YUNSHU_AUTH_TOKEN", value: "***", default: null, type: "str", secret: true, description: "Static admin token." }),
  setting({ name: "YUNSHU_PREFILL_GDN", value: false, default: false, type: "bool", stability: "experimental", description: "Experimental switch." }),
];
const recent = () => Array.from({ length: 30 }, (_, i) => ({ request_id: "req_" + (100000 + i).toString(16) + "abcd", t: T - i * 40, path: "/v1/chat/completions", model: "Qwen3.8-27B-oQ4e-mtp", status: 200, finish_reason: i % 7 === 5 ? "length" : "stop", stream: true, t0_wall: T - i * 40 - 3,
  offsets_ms: { arrive: 0, admit: 30, first_token: 300 + i * 9, last_token: 3300, done: 3320 }, queue_wait_ms: 30, ttft_ms: 300 + i * 9, prompt_tokens: 4000 + i * 100, cached_tokens: i % 3 ? 3500 : 0, completion_tokens: 100, prefill_tps: 900, decode_tps: 35,
  cache: { tier: "ram" }, speculative: i % 4 === 0 ? null : { mode: "mtp", drafted: 120, accepted: 96 - i, rounds: 20, copy: { rounds: 3, tokens: 9 } }, cancelled: false }));
const job = (o) => ({ id: "j1", repo: "mlx-community/Qwen3-4B-4bit", revision: null, allow_patterns: null, state: "running", error: null, path: "~/.yunshu/models/x", bytes_total: 8.1 * GB, bytes_done: 3.2 * GB, files_total: 12, files_done: 3, active_files: ["model-00004-of-00005.safetensors"], rate_bps: 45 * 2 ** 20, eta_s: 108, created: 1, started: 1, finished: null, registered: false, already_present: false, ...o });
const logRec = (id, level, msg) => ({ id, t: T0 + id, level, logger: id % 2 ? "yunshu.engine" : "uvicorn.access", msg });
export function install(ctx, mode = {}) {
  // mode.status: function -> status body | {code}; mode.offline; mode.unauth
  const MSG = ["model loaded Qwen3.8-27B-oQ4e-mtp (16.9 GiB)", "prefix cache restored 8000 tokens from ram", "POST /v1/chat/completions 200", "speculative lane: MTP block 6, 5.1 tokens per round", "POST /v1/messages 200", "apc checkpoint saved (12000 tokens)", "GET /v1/yunshu/status 200", "POST /v1/responses 200"];
  const nowS = Math.floor(Date.now() / 1000);
  const logs = Array.from({ length: 90 }, (_, i) => ({ id: i + 1, t: nowS - (90 - i) * 7, level: i === 37 ? "WARNING" : "INFO", logger: i % 3 ? "uvicorn.access" : "yunshu.engine", msg: i === 37 ? "memory pressure normal, swap 0 B" : MSG[i % MSG.length] }));
  return ctx.route("**/v1/**", (route) => {
    const req = route.request(); const url = new URL(req.url()); const p = url.pathname; const m = req.method();
    const json = (b, code = 200) => route.fulfill({ status: code, contentType: "application/json", body: JSON.stringify(b) });
    if (p === "/v1/yunshu/status") { if (mode.offline) return json({ detail: "bad gateway" }, 502); if (mode.unauth) return json({ detail: "Unauthorized" }, 401); const s = mode.status?.(); if (s?.code) return json({ detail: "boom" }, s.code); return json(s ?? statusBody()); }
    if (p === "/v1/models") return json({ object: "list", data: statusBody().models.map((x) => ({ id: x.id })) });
    if (p === "/v1/yunshu/memory") return json({ object: "yunshu.memory", total_gb: 137.438953472, free_gb: 100, host: { pressure_level: "normal", swap_used_gb: 0, swap_total_gb: 0, wired_limit_gb: 100 }, mlx: { active_gb: 24.9, cache_gb: 2.25, peak_gb: 34.2, recommended_working_set_gb: 100 }, owners: [{ kind: "weights", id: "Qwen3.8-27B", bytes: 16.9e9, gb: 16.9, reclaimable: false, estimated: false, source: "p" }, { kind: "apc_ram", id: null, bytes: 2.4e9, gb: 2.4, reclaimable: true, estimated: false, source: "p" }], attribution_overshoot_gb: null, limits: { apc_max_gb: 8, apc_warm_max_gb: 4, guard_margin_pct: 10 } });
    if (p === "/v1/yunshu/host") return json({ object: "yunshu.host", telemetry: { state: "ok", sampled_at: Date.now() / 1000 - 1, interval_s: 1, watts: { gpu: 14.6, package: 20.4 }, gpu: { frequency_mhz: 1296, active_ratio: 0.78 }, temperature: { state: "ok", die_max_c: 61 }, reasons: {} }, thermal: { state: "normal" }, memory_pressure: { state: "normal" }, power: { state: "ac" } });
    if (p === "/v1/yunshu/requests/recent") return json({ object: "list", data: recent(), count: 30, capacity: 512 });
    if (p === "/v1/yunshu/keys") { if (m === "POST") return json({ ...key("key_new", "New key"), secret: "ysk-zz99-secret-value-0123456789" }, 201); return json({ object: "list", data: mode.noKeys ? [] : [key("key_aaa", "MacBook"), key("key_bbb", "CI", { enabled: false, scopes: ["infer", "admin"], expires: T + 86400 * 20, quotas: { requests_per_day: null, tokens_per_day: 500000, max_concurrent: null }, last_used: null, window: { requests: 0, tokens: 0, inflight: 0 } })] }); }
    if (p === "/v1/yunshu/usage") return json({ object: "list", data: Array.from({ length: 10 }, (_, i) => ({ key: "key_aaa", name: "MacBook", day: new Date(Date.now() - i * 864e5).toISOString().slice(0, 10), requests: 100 + i * 17, prompt_tokens: 40000 + i * 900, completion_tokens: 9000, cached_tokens: 0, errors: 0 })) });
    if (p === "/v1/yunshu/config") { if (m === "PATCH") { const b = JSON.parse(req.postData() || "{}"); return json({ dry_run: !!b.dry_run, results: Object.fromEntries(Object.keys(b.settings || {}).map((n) => [n, { status: n === "YUNSHU_KEEP_ALIVE_TIMEOUT" ? "overridden" : "applied", applies: "live", source: "file", reset: false }])), restart_required: false, reload_required: false, restart: null }); } return json({ object: "yunshu.config", settings: CONFIG, warnings: [], experimental_count: 1, experimental_max: 8 }); }
    if (p === "/v1/yunshu/service") return json({ label: "ai.yunshu.server", plist: "~/Library/LaunchAgents/ai.yunshu.server.plist", installed: true, loaded: true, pid: 4242, state: "running", last_exit_code: 0, log: "~/.yunshu/logs/service.log", under_launchd: true, version: "0.1.5", uptime_s: 93784, cli: { status: "yunshu service status", restart: "yunshu service restart", install: "yunshu service install" } });
    if (p === "/v1/yunshu/service/restart") return json({ restarting: true, active_requests: 1, drain_timeout_s: 30 }, 202);
    if (p === "/v1/yunshu/cors") return json({ origins: ["http://localhost:3000"], wildcard: false, credentials: true, source: "default", default: "http://localhost:3000", warnings: [] });
    if (p === "/v1/yunshu/downloads") return json({ downloads: [job({}), job({ id: "j2", repo: "org/finished", state: "done", files_done: 12, bytes_done: 2 * GB, bytes_total: 2 * GB, finished: 2, registered: true, rate_bps: null, eta_s: null })], active: 1, free_bytes: 212 * GB, models_dir: "~/.yunshu/models" });
    if (p === "/v1/yunshu/models/local") return json({ models: [{ id: "org/on-disk", path: "~/.yunshu/models/org/on-disk", source: "models_dir", size_bytes: 4.4 * GB, model_type: "qwen3", kind: "llm", architecture: "Qwen3", parameters: "8B", quantization: { bits: 4, group_size: 64 }, context_length: 32768, capabilities: ["tools", "reasoning"], complete: true, complete_reason: null, registered_as: null, loaded: false }, { id: "org/half", path: "~/.yunshu/models/org/half", source: "models_dir", size_bytes: 1.1 * GB, model_type: "qwen3", capabilities: [], complete: false, complete_reason: "missing model-00002-of-00002.safetensors", registered_as: null, loaded: false }], total_bytes: 30 * GB, models_dir: "/models", free_bytes: 212 * GB });
    if (p.startsWith("/v1/yunshu/models/") && p.endsWith("/fit")) return json({ model: "x", verdict: "tight", reason: "fits only after evicting org/other-idle", weights_bytes: 20 * GB, kv_reserve_bytes: 2 * GB, needed_bytes: 22 * GB, budget_bytes: 48 * GB, used_bytes: 36 * GB, free_bytes: 12 * GB, free_bytes_after_evict: 37 * GB, would_evict: ["org/other-idle"], loaded: false, basis: { estimated: true } });
    if (p === "/v1/yunshu/cache" || p === "/v1/yunshu/cache/tiers") return json({ enabled: true, caches: [{ model: "Qwen3.8-27B", tiers: [{ name: "ram", used_bytes: 2.4 * GB, cap_bytes: 8 * GB, entries: 14, hits: 80 }, { name: "warm", used_bytes: 0.6 * GB, cap_bytes: 4 * GB, entries: 5, hits: 6, mode: "int8" }, { name: "ssd", used_bytes: 12 * GB, cap_bytes: 64 * GB, entries: 31, hits: 9 }], lookups: { hit: 95, miss: 25, by_tier: { ram: 80, warm: 6, ssd: 9 } }, entries: Array.from({ length: 22 }, (_, i) => ({ key: (0xa1b2c3d4 + i * 4097).toString(16).padStart(8, "0"), tokens: 1000 + i * 700, bytes: (200 + i * 30) * 1e6, tier: i % 4 === 0 ? "warm" : "ram", lru_rank: i, hits: i % 5, last_hit_age_s: i % 3 === 0 ? null : i * 41 })), entries_truncated: false }] });
    if (p === "/v1/yunshu/cache/clear") return json({ cleared: [{ model: "Qwen3.8-27B", tier: "ram", entries: 14, freed_bytes: 2.4 * GB }], freed_bytes: 2.4 * GB });
    if (p === "/v1/yunshu/logs/stream") return; // held open like a live SSE stream
    if (p === "/v1/yunshu/logs") return json({ records: logs, next_id: 91, dropped: 0, capacity: 2000, server_time: nowS });
    if (p === "/v1/models/load") return json({ status: "loaded" });
    if (p === "/v1/models/unload") return json({ status: "unloaded" });
    if (p === "/v1/chat/completions") return route.fulfill({ status: 200, contentType: "text/event-stream", body: `data: ${JSON.stringify({ choices: [{ delta: { content: "你好，" }, logprobs: { content: [{ token: "你好", logprob: -0.11, top_logprobs: [{ token: "你好", logprob: -0.11 }, { token: "嗨", logprob: -2.4 }] }] } }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: { content: "世界" }, finish_reason: "stop", logprobs: { content: [{ token: "世界", logprob: -1.9, top_logprobs: [{ token: "世界", logprob: -1.9 }, { token: "朋友", logprob: -0.8 }] }] } }], usage: { prompt_tokens: 9, completion_tokens: 2 } })}\n\ndata: [DONE]\n\n` });
    return json({ detail: "Not Found" }, 404);
  });
}
