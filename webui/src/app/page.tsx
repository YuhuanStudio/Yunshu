"use client";

import { fmtBytes, guessModelType } from "@/lib/utils";
import { useEffect, useState, useRef, useCallback } from "react";
import {
  Activity,
  Zap,
  HardDrive,
  Clock,
  Brain,
  Eye,
  Volume2,
  Mic,
  ImageIcon,
  Loader2,
  Download,
  Upload,
  RefreshCw,
  TrendingUp,
} from "lucide-react";

interface EngineStats {
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
  gpu_memory: {
    total_bytes: number;
    active_bytes: number;
    peak_bytes: number;
    cache_bytes: number;
    available_bytes: number;
    utilization_pct: number;
  } | null;
}

interface SystemStats {
  cpu_percent: number;
  memory_total_bytes: number;
  memory_used_bytes: number;
  memory_available_bytes: number;
  gpu: {
    total_bytes: number;
    active_bytes: number;
    peak_bytes: number;
    cache_bytes: number;
    available_bytes: number;
    utilization_pct: number;
  };
  python_version: string;
  mlx_version: string;
}

interface ModelInfo {
  id: string;
  loaded?: boolean;
}

const typeConfig: Record<string, { icon: typeof Brain; color: string; bg: string }> = {
  LLM: { icon: Brain, color: "text-blue-400", bg: "bg-blue-500/15" },
  VLM: { icon: Eye, color: "text-purple-400", bg: "bg-purple-500/15" },
  TTS: { icon: Volume2, color: "text-amber-400", bg: "bg-amber-500/15" },
  ASR: { icon: Mic, color: "text-emerald-400", bg: "bg-emerald-500/15" },
  IMAGE_GEN: { icon: ImageIcon, color: "text-rose-400", bg: "bg-rose-500/15" },
};

const REQUEST_HISTORY_LEN = 30;

export default function DashboardPage() {
  const [engine, setEngine] = useState<EngineStats | null>(null);
  const [system, setSystem] = useState<SystemStats | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [requestHistory, setRequestHistory] = useState<number[]>([]);
  const [gpuHistory, setGpuHistory] = useState<number[]>([]);
  const [actionModelId, setActionModelId] = useState<string | null>(null);
  const [loadInput, setLoadInput] = useState("");
  const mounted = useRef(true);

  const fetchData = useCallback(async () => {
    try {
      const [engRes, modRes, sysRes] = await Promise.all([
        fetch("/api/v1/monitoring/engine"),
        fetch("/v1/models"),
        fetch("/api/v1/monitoring/system"),
      ]);
      if (engRes.ok && mounted.current) {
        const eng = await engRes.json();
        setEngine(eng);
        setRequestHistory((prev) => [...prev.slice(-(REQUEST_HISTORY_LEN - 1)), eng.active_requests ?? 0]);
      }
      if (modRes.ok && mounted.current) {
        const d = await modRes.json();
        setModels(d.data || []);
      }
      if (sysRes.ok && mounted.current) {
        const sys = await sysRes.json();
        setSystem(sys);
        setGpuHistory((prev) => [...prev.slice(-(REQUEST_HISTORY_LEN - 1)), sys.gpu?.utilization_pct ?? 0]);
      }
    } catch {
      // server unreachable
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    fetchData();
    const id = setInterval(fetchData, 5000);
    return () => {
      mounted.current = false;
      clearInterval(id);
    };
  }, [fetchData]);

  const handleLoad = async (overrideId?: string) => {
    const modelId = overrideId || loadInput;
    if (!modelId.trim()) return;
    setActionModelId(modelId);
    try {
      await fetch("/v1/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelId }),
      });
      setLoadInput("");
      await fetchData();
    } catch {
    } finally {
      setActionModelId(null);
    }
  };

  const handleUnload = async (id: string) => {
    setActionModelId(id);
    try {
      await fetch("/api/v1/admin/models/unload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: id }),
      });
      await fetchData();
    } catch {
    } finally {
      setActionModelId(null);
    }
  };

  if (loading) return <LoadingSkeleton />;

  const stats = [
    {
      title: "Requests",
      value: engine?.requests_processed ?? 0,
      format: "number" as const,
      subtitle: `${engine?.active_requests ?? 0} active`,
      icon: Activity,
      accent: "text-blue-400",
    },
    {
      title: "Tokens/s",
      value: engine
        ? engine.total_completion_tokens / Math.max(engine.uptime_seconds, 1)
        : 0,
      format: "decimal" as const,
      subtitle: `${fmtTok(engine?.total_prompt_tokens ?? 0)} prompt`,
      icon: Zap,
      accent: "text-amber-400",
    },
    {
      title: "GPU Memory",
      value: engine?.gpu_memory?.utilization_pct ?? 0,
      format: "percent" as const,
      subtitle: `${fmtBytes(engine?.gpu_memory?.active_bytes ?? 0)} / ${fmtBytes(engine?.gpu_memory?.total_bytes ?? 0)}`,
      icon: HardDrive,
      accent: "text-emerald-400",
    },
    {
      title: "Uptime",
      value: engine?.uptime_seconds ?? 0,
      format: "duration" as const,
      subtitle: `${engine?.step_counter ?? 0} steps`,
      icon: Clock,
      accent: "text-purple-400",
    },
  ];

  return (
    <div className="p-6 space-y-6 page-enter">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Dashboard</h2>
          <p className="text-sm text-[var(--color-text-secondary)] mt-0.5">
            {engine?.model ?? "No model loaded"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => { setLoading(true); fetchData(); }}
            className="p-2 rounded-lg text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
            <span
              className={`w-2 h-2 rounded-full ${
                engine?.running ? "bg-[var(--color-success)]" : "bg-[var(--color-danger)]"
              }`}
            />
            {engine?.running ? "Running" : "Stopped"}
          </div>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-4 gap-4">
        {stats.map((s) => (
          <div
            key={s.title}
            className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 hover:border-[var(--color-border)]/80 transition-colors"
          >
            <div className="flex items-center gap-2 mb-2">
              <s.icon className={`w-4 h-4 ${s.accent}`} />
              <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wide">
                {s.title}
              </span>
            </div>
            <div className="text-2xl font-bold tabular-nums">{formatValue(s.value, s.format)}</div>
            {s.subtitle && (
              <div className="text-xs text-[var(--color-text-secondary)] mt-1">{s.subtitle}</div>
            )}
          </div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-2 gap-4">
        {/* Active Requests Chart */}
        <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-sm flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-blue-400" />
              Active Requests
            </h3>
            <span className="text-xs text-[var(--color-text-secondary)] tabular-nums">
              {engine?.active_requests ?? 0} now
            </span>
          </div>
          {requestHistory.length < 2 ? (
            <div className="h-28 flex items-center justify-center text-xs text-[var(--color-text-secondary)]">
              Collecting data...
            </div>
          ) : (
            <svg
              viewBox={`0 0 ${REQUEST_HISTORY_LEN} 40`}
              className="w-full h-28"
              preserveAspectRatio="none"
            >
              <defs>
                <linearGradient id="reqGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                </linearGradient>
              </defs>
              <polygon
                fill="url(#reqGrad)"
                points={
                  requestHistory
                    .map((v, i) => {
                      const x = (i / (REQUEST_HISTORY_LEN - 1)) * REQUEST_HISTORY_LEN;
                      const y = 40 - (v / Math.max(...requestHistory, 1)) * 36;
                      return `${x},${y}`;
                    })
                    .join(" ") +
                  ` ${REQUEST_HISTORY_LEN},40 0,40`
                }
              />
              <polyline
                className="sparkline-path"
                points={requestHistory
                  .map((v, i) => {
                    const x = (i / (REQUEST_HISTORY_LEN - 1)) * REQUEST_HISTORY_LEN;
                    const y = 40 - (v / Math.max(...requestHistory, 1)) * 36;
                    return `${x},${y}`;
                  })
                  .join(" ")}
              />
            </svg>
          )}
        </div>

        {/* GPU Utilization Chart */}
        <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-sm flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-emerald-400" />
              GPU Utilization
            </h3>
            <span className="text-xs text-[var(--color-text-secondary)] tabular-nums">
              {(engine?.gpu_memory?.utilization_pct ?? 0).toFixed(1)}%
            </span>
          </div>
          {gpuHistory.length < 2 ? (
            <div className="h-28 flex items-center justify-center text-xs text-[var(--color-text-secondary)]">
              Collecting data...
            </div>
          ) : (
            <svg
              viewBox={`0 0 ${REQUEST_HISTORY_LEN} 40`}
              className="w-full h-28"
              preserveAspectRatio="none"
            >
              <defs>
                <linearGradient id="gpuGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22c55e" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#22c55e" stopOpacity="0" />
                </linearGradient>
              </defs>
              <polygon
                fill="url(#gpuGrad)"
                points={
                  gpuHistory
                    .map((v, i) => {
                      const x = (i / (REQUEST_HISTORY_LEN - 1)) * REQUEST_HISTORY_LEN;
                      const y = 40 - (v / 100) * 36;
                      return `${x},${y}`;
                    })
                    .join(" ") +
                  ` ${REQUEST_HISTORY_LEN},40 0,40`
                }
              />
              <polyline
                fill="none"
                stroke="#22c55e"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={gpuHistory
                  .map((v, i) => {
                    const x = (i / (REQUEST_HISTORY_LEN - 1)) * REQUEST_HISTORY_LEN;
                    const y = 40 - (v / 100) * 36;
                    return `${x},${y}`;
                  })
                  .join(" ")}
              />
            </svg>
          )}
        </div>
      </div>

      {/* System Stats Bar */}
      {system && (
        <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
          <h3 className="font-semibold text-sm mb-3">System</h3>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 text-sm">
            <KV label="CPU" value={`${system.cpu_percent.toFixed(1)}%`} />
            <KV label="RAM" value={`${fmtBytes(system.memory_used_bytes)} / ${fmtBytes(system.memory_total_bytes)}`} />
            <KV label="GPU Active" value={fmtBytes(system.gpu.active_bytes)} />
            <KV label="GPU Available" value={fmtBytes(system.gpu.available_bytes)} />
            <KV label="MLX" value={system.mlx_version} />
          </div>
        </div>
      )}

      {/* Quick Actions + Model List */}
      <div className="grid grid-cols-3 gap-4">
        {/* Quick Actions */}
        <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
          <h3 className="font-semibold text-sm mb-3">Quick Actions</h3>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-[var(--color-text-secondary)] block mb-1">
                Load Model
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={loadInput}
                  onChange={(e) => setLoadInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleLoad()}
                  placeholder="model-id or path"
                  className="flex-1 bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-[var(--color-accent)] min-w-0"
                />
                <button
                  onClick={handleLoad}
                  disabled={!loadInput.trim() || actionModelId !== null}
                  className="flex items-center gap-1 bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] disabled:opacity-50 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap"
                >
                  {actionModelId && actionModelId === loadInput ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Download className="w-3 h-3" />
                  )}
                  Load
                </button>
              </div>
            </div>
            <div className="pt-2 border-t border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">
                {models.filter((m) => m.loaded).length} loaded,{" "}
                {models.filter((m) => !m.loaded).length} available
              </p>
            </div>
          </div>
        </div>

        {/* Model List */}
        <div className="col-span-2 bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)]">
          <div className="px-4 py-3 border-b border-[var(--color-border)] flex items-center justify-between">
            <h3 className="font-semibold text-sm">Models</h3>
            <span className="text-xs text-[var(--color-text-secondary)]">
              {models.length} discovered
            </span>
          </div>
          {models.length === 0 ? (
            <div className="p-6 text-center text-[var(--color-text-secondary)] text-sm">
              No models loaded. Use{" "}
              <code className="bg-[var(--color-bg-tertiary)] px-1.5 py-0.5 rounded text-xs">
                yunshu serve --model &lt;model&gt;
              </code>{" "}
              to start.
            </div>
          ) : (
            <div className="divide-y divide-[var(--color-border)] max-h-64 overflow-auto">
              {models.map((m) => {
                const type = guessModelType(m.id);
                const cfg = typeConfig[type] || typeConfig.LLM;
                const Icon = cfg.icon;
                const isLoading = actionModelId === m.id;
                return (
                  <div key={m.id} className="flex items-center gap-3 px-4 py-2.5">
                    <div className={`w-7 h-7 rounded-lg ${cfg.bg} flex items-center justify-center`}>
                      <Icon className={`w-3.5 h-3.5 ${cfg.color}`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-sm truncate">{m.id}</div>
                      <div className="text-xs text-[var(--color-text-secondary)]">{type}</div>
                    </div>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        m.loaded
                          ? "bg-[var(--color-success)]/15 text-[var(--color-success)]"
                          : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                      }`}
                    >
                      {m.loaded ? "Loaded" : "Available"}
                    </span>
                    <button
                      onClick={() => m.loaded ? handleUnload(m.id) : handleLoad(m.id)}
                      disabled={isLoading}
                      className={`p-1.5 rounded-lg transition-colors disabled:opacity-50 ${
                        m.loaded
                          ? "text-[var(--color-danger)] hover:bg-[var(--color-danger)]/10"
                          : "text-[var(--color-accent)] hover:bg-[var(--color-accent-muted)]"
                      }`}
                      title={m.loaded ? "Unload" : "Load"}
                    >
                      {isLoading ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : m.loaded ? (
                        <Upload className="w-3.5 h-3.5" />
                      ) : (
                        <Download className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Engine Details */}
      {engine && (
        <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
          <h3 className="font-semibold text-sm mb-3">Engine Details</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
            <KV label="Prompt Tokens" value={fmtTok(engine.total_prompt_tokens)} />
            <KV label="Completion Tokens" value={fmtTok(engine.total_completion_tokens)} />
            <KV label="Total Tokens" value={fmtTok(engine.total_prompt_tokens + engine.total_completion_tokens)} />
            <KV label="Steps" value={engine.step_counter.toLocaleString()} />
          </div>
        </div>
      )}
    </div>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-[var(--color-text-secondary)]">{label}</div>
      <div className="font-medium tabular-nums">{value}</div>
    </div>
  );
}

function formatValue(v: number, f: "number" | "decimal" | "percent" | "duration") {
  switch (f) {
    case "decimal": return v.toFixed(1);
    case "percent": return `${v.toFixed(1)}%`;
    case "duration": return fmtDur(v);
    default: return v.toLocaleString();
  }
}

function fmtTok(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toString();
}

function fmtDur(s: number): string {
  if (s < 60) return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function LoadingSkeleton() {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <div className="skeleton w-32 h-7" />
          <div className="skeleton w-48 h-4 mt-2" />
        </div>
        <div className="skeleton w-20 h-5" />
      </div>
      <div className="grid grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-3">
            <div className="skeleton w-16 h-3" />
            <div className="skeleton w-24 h-7" />
            <div className="skeleton w-20 h-3" />
          </div>
        ))}
      </div>
    </div>
  );
}
