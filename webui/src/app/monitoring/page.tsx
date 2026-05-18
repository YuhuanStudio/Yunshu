"use client";

import { fmtBytes } from "@/lib/utils";
import { useEffect, useState, useRef } from "react";
import {
  Cpu,
  HardDrive,
  Activity,
  RefreshCw,
  Gauge,
  MemoryStick,
  CircleDot,
  AlertTriangle,
  Zap,
  Thermometer,
  Network,
  Brain,
  Wrench,
  Layers,
  GitBranch,
  ArrowRightLeft,
  BarChart3,
  Eye,
  Radio,
  Shield,
} from "lucide-react";

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

interface SLOAlert {
  id: string;
  metric: string;
  threshold: number;
  current: number;
  unit: string;
  severity: "critical" | "warning" | "info";
  message: string;
}

interface SLOConfig {
  ttft_p95_ms: number;
  tps_min: number;
  gpu_memory_max_pct: number;
  error_rate_max_pct: number;
}

const MAX_HISTORY = 60;

export default function MonitoringPage() {
  const [system, setSystem] = useState<SystemStats | null>(null);
  const [engine, setEngine] = useState<Record<string, unknown> | null>(null);
  const [specDecode, setSpecDecode] = useState<Record<string, unknown> | null>(null);
  const [kvCache, setKvCache] = useState<Record<string, unknown> | null>(null);
  const [requests, setRequests] = useState<Record<string, unknown> | null>(null);
  const [memoryGuard, setMemoryGuard] = useState<Record<string, unknown> | null>(null);
  const [ssdCache, setSsdCache] = useState<Record<string, unknown> | null>(null);
  const [prefillProgress, setPrefillProgress] = useState<Record<string, unknown> | null>(null);
  const [radixTree, setRadixTree] = useState<Record<string, unknown> | null>(null);
  const [hwProfile, setHwProfile] = useState<Record<string, unknown> | null>(null);
  const [meshStatus, setMeshStatus] = useState<Record<string, unknown> | null>(null);
  const [engineTuning, setEngineTuning] = useState<Record<string, unknown> | null>(null);
  const [modelStats, setModelStats] = useState<Record<string, unknown> | null>(null);
  const [dataParallel, setDataParallel] = useState<Record<string, unknown> | null>(null);
  const [perModel, setPerModel] = useState<Record<string, unknown> | null>(null);
  const [thinkingSegments, setThinkingSegments] = useState<Record<string, unknown> | null>(null);
  const [metalKernels, setMetalKernels] = useState<Record<string, unknown> | null>(null);
  const [aneEmbeddings, setAneEmbeddings] = useState<Record<string, unknown> | null>(null);
  const [externalPrefill, setExternalPrefill] = useState<Record<string, unknown> | null>(null);
  const [healthDashboard, setHealthDashboard] = useState<Record<string, unknown> | null>(null);
  const [reasoningTokens, setReasoningTokens] = useState<Record<string, unknown> | null>(null);
  const [responseCache, setResponseCache] = useState<Record<string, unknown> | null>(null);
  const [inflightPrefix, setInflightPrefix] = useState<Record<string, unknown> | null>(null);
  const [requestCoalescer, setRequestCoalescer] = useState<Record<string, unknown> | null>(null);
  const [tokenScheduler, setTokenScheduler] = useState<Record<string, unknown> | null>(null);
  const [kvMigration, setKvMigration] = useState<Record<string, unknown> | null>(null);
  const [attentionEviction, setAttentionEviction] = useState<Record<string, unknown> | null>(null);
  const [batchSize, setBatchSize] = useState<Record<string, unknown> | null>(null);
  const [autoTuner, setAutoTuner] = useState<Record<string, unknown> | null>(null);
  const [memoryPressure, setMemoryPressure] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [gpuHistory, setGpuHistory] = useState<number[]>([]);
  const [sloAlerts, setSloAlerts] = useState<SLOAlert[]>([]);
  const mounted = useRef(true);

  const sloConfig: SLOConfig = {
    ttft_p95_ms: 1200,
    tps_min: 30,
    gpu_memory_max_pct: 90,
    error_rate_max_pct: 5,
  };

  useEffect(() => {
    mounted.current = true;
    const fetchData = async () => {
      try {
        // Use aggregated endpoint to reduce 31 fetches to 4
        const [sysRes, engRes, gwSysRes, allRes, radixRes, hwRes, meshRes, modelsRes] = await Promise.all([
          fetch("/api/v1/monitoring/system").catch(() => null),
          fetch("/api/v1/monitoring/engine").catch(() => null),
          fetch("/api/v1/gw/monitoring/system").catch(() => null),
          fetch("/api/v1/gw/monitoring/all").catch(() => null),
          fetch("/api/v1/admin/radix-tree").catch(() => null),
          fetch("/api/v1/admin/hardware-profile").catch(() => null),
          fetch("/api/v1/mesh/status").catch(() => null),
          fetch("/v1/models").catch(() => null),
        ]);
        let sysData: SystemStats | null = null;
        let engData: Record<string, unknown> | null = null;
        // Try control-plane system endpoint first (returns SystemStats shape)
        if (sysRes && sysRes.ok) {
          sysData = await sysRes.json();
        }
        // Fallback to gateway system endpoint (different response shape — normalize)
        if (!sysData && gwSysRes && gwSysRes.ok) {
          const gw = await gwSysRes.json();
          sysData = {
            cpu_percent: gw?.cpu?.percent ?? 0,
            memory_total_bytes: gw?.memory?.total_bytes ?? 0,
            memory_used_bytes: gw?.memory?.used_bytes ?? 0,
            memory_available_bytes: gw?.memory?.available_bytes ?? 0,
            gpu: {
              total_bytes: gw?.gpu?.total_uma_bytes ?? gw?.gpu?.total_bytes ?? 0,
              active_bytes: gw?.gpu?.active_bytes ?? 0,
              peak_bytes: gw?.gpu?.peak_bytes ?? 0,
              cache_bytes: gw?.gpu?.cache_bytes ?? 0,
              available_bytes: gw?.gpu?.available_bytes ?? 0,
              utilization_pct: gw?.gpu?.utilization_pct ?? 0,
            },
            python_version: gw?.python_version ?? "",
            mlx_version: gw?.gpu?.mlx_version ?? "",
          };
        }
        if (sysData && mounted.current) {
          setSystem(sysData);
          setGpuHistory((prev) => [...prev.slice(-(MAX_HISTORY - 1)), sysData!.gpu?.utilization_pct ?? 0]);
        }
        if (engRes && engRes.ok) {
          engData = await engRes.json();
          if (mounted.current) setEngine(engData);
        }
        // Unpack aggregated monitoring data
        if (allRes && allRes.ok) {
          const all = await allRes.json();
          if (mounted.current) {
            if (all.spec_decode) setSpecDecode(all.spec_decode);
            if (all.kv_cache) setKvCache(all.kv_cache);
            if (all.requests) setRequests(all.requests);
            if (all.memory_guard) setMemoryGuard(all.memory_guard);
            if (all.ssd_cache) setSsdCache(all.ssd_cache);
            if (all.prefill_progress) setPrefillProgress(all.prefill_progress);
            if (all.data_parallel) setDataParallel(all.data_parallel);
            if (all.per_model) setPerModel(all.per_model);
            if (all.thinking_segments) setThinkingSegments(all.thinking_segments);
            if (all.metal_kernels) setMetalKernels(all.metal_kernels);
            if (all.ane_embeddings) setAneEmbeddings(all.ane_embeddings);
            if (all.external_prefill) setExternalPrefill(all.external_prefill);
            if (all.health_dashboard) setHealthDashboard(all.health_dashboard);
            if (all.reasoning_tokens) setReasoningTokens(all.reasoning_tokens);
            if (all.response_cache) setResponseCache(all.response_cache);
            if (all.inflight_prefix_sharing) setInflightPrefix(all.inflight_prefix_sharing);
            if (all.request_coalescer) setRequestCoalescer(all.request_coalescer);
            if (all.token_scheduler) setTokenScheduler(all.token_scheduler);
            if (all.kv_migration) setKvMigration(all.kv_migration);
            if (all.attention_eviction) setAttentionEviction(all.attention_eviction);
            if (all.batch_size) setBatchSize(all.batch_size);
            if (all.auto_tuner) setAutoTuner(all.auto_tuner);
            if (all.memory_pressure) setMemoryPressure(all.memory_pressure);
          }
        }
        if (radixRes && radixRes.ok) {
          const radixData = await radixRes.json();
          if (mounted.current) setRadixTree(radixData);
        }
        if (hwRes && hwRes.ok) {
          const hwData = await hwRes.json();
          if (mounted.current) setHwProfile(hwData);
        }
        if (meshRes && meshRes.ok) {
          const meshData = await meshRes.json();
          if (mounted.current) setMeshStatus(meshData);
        }
        if (modelsRes && modelsRes.ok) {
          const modelsData = await modelsRes.json();
          if (mounted.current) setModelStats(modelsData);
        }
        if (mounted.current) {
          setLastUpdate(new Date());
          evaluateSLOs(sysData, engData);
        }
      } catch {
      } finally {
        if (mounted.current) setLoading(false);
      }
    };
    fetchData();
    const id = setInterval(fetchData, 5000);
    return () => {
      mounted.current = false;
      clearInterval(id);
    };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--color-text-secondary)]">
        Loading monitoring data...
      </div>
    );
  }

  function evaluateSLOs(sys: SystemStats | null, eng: Record<string, unknown> | null) {
    const alerts: SLOAlert[] = [];
    if (sys?.gpu) {
      const gpuPct = sys.gpu.total_bytes > 0 ? (sys.gpu.active_bytes / sys.gpu.total_bytes) * 100 : 0;
      if (gpuPct > sloConfig.gpu_memory_max_pct) {
        alerts.push({
          id: "gpu-mem",
          metric: "GPU Memory",
          threshold: sloConfig.gpu_memory_max_pct,
          current: gpuPct,
          unit: "%",
          severity: gpuPct > 95 ? "critical" : "warning",
          message: `GPU memory at ${gpuPct.toFixed(1)}% (threshold: ${sloConfig.gpu_memory_max_pct}%)`,
        });
      }
    }
    if (eng) {
      const ttft = typeof eng.ttft_p95_ms === "number" ? eng.ttft_p95_ms : null;
      if (ttft !== null && ttft > sloConfig.ttft_p95_ms) {
        alerts.push({
          id: "ttft",
          metric: "TTFT P95",
          threshold: sloConfig.ttft_p95_ms,
          current: ttft,
          unit: "ms",
          severity: ttft > sloConfig.ttft_p95_ms * 2 ? "critical" : "warning",
          message: `P95 TTFT is ${ttft.toFixed(0)}ms (SLO: <${sloConfig.ttft_p95_ms}ms)`,
        });
      }
      const tps = typeof eng.tok_per_sec === "number" ? eng.tok_per_sec : null;
      if (tps !== null && tps < sloConfig.tps_min) {
        alerts.push({
          id: "tps",
          metric: "Throughput",
          threshold: sloConfig.tps_min,
          current: tps,
          unit: "tok/s",
          severity: tps < sloConfig.tps_min / 2 ? "critical" : "warning",
          message: `Throughput is ${tps.toFixed(1)} tok/s (SLO: ≥${sloConfig.tps_min})`,
        });
      }
      const errRate = typeof eng.error_rate_pct === "number" ? eng.error_rate_pct : null;
      if (errRate !== null && errRate > sloConfig.error_rate_max_pct) {
        alerts.push({
          id: "errors",
          metric: "Error Rate",
          threshold: sloConfig.error_rate_max_pct,
          current: errRate,
          unit: "%",
          severity: errRate > 20 ? "critical" : "warning",
          message: `Error rate is ${errRate.toFixed(1)}% (SLO: <${sloConfig.error_rate_max_pct}%)`,
        });
      }
    }
    setSloAlerts(alerts);
  }

  return (
    <div className="p-6 space-y-6 page-enter">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Monitoring</h2>
        <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
          <RefreshCw className="w-3 h-3" />
          {lastUpdate ? lastUpdate.toLocaleTimeString() : "—"}
        </div>
      </div>

      {!system && !engine ? (
        <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-8 text-center text-[var(--color-text-secondary)] text-sm">
          Monitoring data not available. Make sure the server is running with the control plane enabled.
        </div>
      ) : (
        <>
          {/* SLO Alerts */}
          {sloAlerts.length > 0 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-3">
              <h3 className="font-semibold text-sm flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-500" />
                SLO Alerts ({sloAlerts.length})
              </h3>
              {sloAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`flex items-start gap-3 p-3 rounded-lg border ${
                    alert.severity === "critical"
                      ? "bg-red-500/10 border-red-500/30"
                      : alert.severity === "warning"
                      ? "bg-amber-500/10 border-amber-500/30"
                      : "bg-blue-500/10 border-blue-500/30"
                  }`}
                >
                  <AlertTriangle
                    className={`w-4 h-4 shrink-0 mt-0.5 ${
                      alert.severity === "critical" ? "text-red-500" : "text-amber-500"
                    }`}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{alert.metric}</span>
                      <span
                        className={`text-xs px-1.5 py-0.5 rounded-full font-medium ${
                          alert.severity === "critical"
                            ? "bg-red-500/20 text-red-400"
                            : "bg-amber-500/20 text-amber-400"
                        }`}
                      >
                        {alert.severity.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-xs text-[var(--color-text-secondary)] mt-1">{alert.message}</div>
                  </div>
                  <div className="text-sm font-bold tabular-nums shrink-0">
                    {alert.current.toFixed(alert.unit === "%" ? 1 : 0)} {alert.unit}
                  </div>
                </div>
              ))}
            </div>
          )}
          {/* GPU Memory */}
          {system?.gpu && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <HardDrive className="w-4 h-4 text-[var(--color-accent)]" />
                  GPU Memory
                </h3>
                <span className="text-xs text-[var(--color-text-secondary)]">
                  {fmtBytes(system.gpu.active_bytes)} / {fmtBytes(system.gpu.total_bytes)}
                </span>
              </div>
              <div className="space-y-3">
                <MemBar label="Active" value={system.gpu.active_bytes} max={system.gpu.total_bytes} color="bg-blue-500" />
                <MemBar label="Peak" value={system.gpu.peak_bytes} max={system.gpu.total_bytes} color="bg-amber-500" />
                <MemBar label="Cache" value={system.gpu.cache_bytes} max={system.gpu.total_bytes} color="bg-emerald-500" />
              </div>

              {/* GPU Gauge */}
              <div className="flex items-center gap-6 pt-2">
                <GaugeCircle pct={system.gpu.utilization_pct} />
                <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm flex-1">
                  <KV icon={HardDrive} label="Total" value={fmtBytes(system.gpu.total_bytes)} />
                  <KV icon={Activity} label="Active" value={fmtBytes(system.gpu.active_bytes)} />
                  <KV icon={CircleDot} label="Available" value={fmtBytes(system.gpu.available_bytes)} />
                  <KV icon={Gauge} label="Utilization" value={`${system.gpu.utilization_pct.toFixed(1)}%`} />
                </div>
              </div>

              {/* Sparkline */}
              {gpuHistory.length > 1 && (
                <div className="pt-2">
                  <div className="text-xs text-[var(--color-text-secondary)] mb-1">GPU Utilization History</div>
                  <svg viewBox={`0 0 ${MAX_HISTORY} 30`} className="w-full h-16" preserveAspectRatio="none">
                    <polyline
                      className="sparkline-path"
                      points={gpuHistory
                        .map((v, i) => {
                          const x = (i / (MAX_HISTORY - 1)) * MAX_HISTORY;
                          const y = 30 - (v / 100) * 28;
                          return `${x},${y}`;
                        })
                        .join(" ")}
                    />
                    {/* Fill area */}
                    <polygon
                      fill="url(#sparkGrad)"
                      points={
                        gpuHistory
                          .map((v, i) => {
                            const x = (i / (MAX_HISTORY - 1)) * MAX_HISTORY;
                            const y = 30 - (v / 100) * 28;
                            return `${x},${y}`;
                          })
                          .join(" ") +
                        ` ${MAX_HISTORY},30 0,30`
                      }
                    />
                    <defs>
                      <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--color-accent)" stopOpacity="0.3" />
                        <stop offset="100%" stopColor="var(--color-accent)" stopOpacity="0" />
                      </linearGradient>
                    </defs>
                  </svg>
                </div>
              )}
            </div>
          )}

          {/* System */}
          {system && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Cpu className="w-4 h-4 text-[var(--color-accent)]" />
                System
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                <KV icon={Cpu} label="CPU" value={`${system.cpu_percent.toFixed(1)}%`} />
                <KV icon={MemoryStick} label="RAM Total" value={fmtBytes(system.memory_total_bytes)} />
                <KV icon={MemoryStick} label="RAM Used" value={fmtBytes(system.memory_used_bytes)} />
                <KV icon={MemoryStick} label="RAM Available" value={fmtBytes(system.memory_available_bytes)} />
                <KV icon={Activity} label="Python" value={system.python_version} />
                <KV icon={Activity} label="MLX" value={system.mlx_version} />
              </div>
            </div>
          )}

          {/* Engine */}
          {engine && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Activity className="w-4 h-4 text-[var(--color-accent)]" />
                Engine
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {Object.entries(engine).map(([key, value]) => (
                  <div key={key}>
                    <div className="text-xs text-[var(--color-text-secondary)]">
                      {key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </div>
                    <div className="font-medium tabular-nums">
                      {typeof value === "number"
                        ? value > 100000
                          ? fmtBytes(value)
                          : value.toLocaleString()
                        : String(value)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Speculative Decoding */}
          {specDecode && specDecode.models && (specDecode.models as Record<string, unknown>[]).length > 0 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Gauge className="w-4 h-4 text-[var(--color-accent)]" />
                Speculative Decoding
              </h3>
              {(specDecode.models as Record<string, unknown>[]).map((model: Record<string, unknown>, idx: number) => (
                <div key={idx} className="mb-3 last:mb-0">
                  <div className="text-xs text-[var(--color-text-secondary)] mb-1">
                    {String(model.model_id)} {model.ngram_enabled ? "(N-gram)" : ""} {model.spec_enabled ? "(Model-based)" : ""}
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                    {(model.ngram_stats as Record<string, number>) && Object.entries(model.ngram_stats as Record<string, number>).map(([k, v]) => (
                      <div key={k}>
                        <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                        <div className="font-medium tabular-nums">{v.toLocaleString()}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* KV Cache */}
          {kvCache && kvCache.caches && (kvCache.caches as Record<string, unknown>[]).length > 0 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <HardDrive className="w-4 h-4 text-[var(--color-accent)]" />
                KV Prefix Cache
              </h3>
              {(kvCache.caches as Record<string, unknown>[]).map((cache: Record<string, unknown>, idx: number) => (
                <div key={idx} className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                  {Object.entries(cache).map(([key, value]) => (
                    <div key={key}>
                      <div className="text-xs text-[var(--color-text-secondary)]">
                        {key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                      </div>
                      <div className="font-medium tabular-nums">
                        {typeof value === "number" ? value.toLocaleString() : String(value)}
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}

          {/* Latency Percentiles */}
          {requests && requests.latency_percentiles && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Activity className="w-4 h-4 text-[var(--color-accent)]" />
                Latency Percentiles (Last 60s)
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                {Object.entries(requests.latency_percentiles as Record<string, number>).map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">{k}</div>
                    <div className="font-medium tabular-nums">{typeof v === "number" ? `${v.toFixed(1)}ms` : String(v)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ITL (Inter-Token Latency) */}
          {requests && (requests as Record<string, unknown>).itl && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <CircleDot className="w-4 h-4 text-[var(--color-accent)]" />
                Inter-Token Latency
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {Object.entries((requests as Record<string, unknown>).itl as Record<string, unknown>).map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">
                      {k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </div>
                    <div className="font-medium tabular-nums">
                      {typeof v === "number" ? v.toFixed(2) : String(v)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Memory Guard */}
          {memoryGuard && memoryGuard.active && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <AlertTriangle className="w-4 h-4 text-[var(--color-accent)]" />
                Memory Guard
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Pressure Level</div>
                  <div className="font-medium">{String(memoryGuard.pressure_level || "normal")}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Evictions</div>
                  <div className="font-medium tabular-nums">{Number(memoryGuard.eviction_count || 0).toLocaleString()}</div>
                </div>
              </div>
            </div>
          )}

          {/* SSD KV Cache */}
          {ssdCache && ssdCache.active && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <HardDrive className="w-4 h-4 text-[var(--color-accent)]" />
                SSD KV Cache
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {Object.entries(ssdCache).filter(([k]) => k !== "active").map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">
                      {k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </div>
                    <div className="font-medium tabular-nums">
                      {typeof v === "number" ? v.toLocaleString() : String(v)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Prefill Progress */}
          {prefillProgress && prefillProgress.active && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Activity className="w-4 h-4 text-[var(--color-accent)]" />
                Active Prefills
              </h3>
              <div className="space-y-2">
                {Object.entries((prefillProgress.requests || {}) as Record<string, Record<string, unknown>[]>).map(([model, reqs]) => (
                  <div key={model}>
                    <div className="text-xs text-[var(--color-text-secondary)] mb-1">{model}</div>
                    {reqs.map((req, idx) => (
                      <div key={idx} className="flex items-center gap-3 text-sm py-1">
                        <div className="flex-1">
                          <div className="flex justify-between text-xs mb-0.5">
                            <span>{req.request_id}</span>
                            <span>{String(req.progress_pct || 0)}%</span>
                          </div>
                          <div className="w-full bg-[var(--color-bg-tertiary)] rounded-full h-1.5">
                            <div
                              className="bg-[var(--color-accent)] h-1.5 rounded-full transition-all duration-300"
                              style={{ width: `${Number(req.progress_pct || 0)}%` }}
                            />
                          </div>
                        </div>
                        <span className="text-xs text-[var(--color-text-secondary)] tabular-nums shrink-0">
                          {String(req.speed_tok_s || 0)} tok/s
                        </span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* RadixTree Stats */}
          {radixTree && radixTree.enabled && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <HardDrive className="w-4 h-4 text-[var(--color-accent)]" />
                RadixTree Prefix Cache
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Total Nodes</div>
                  <div className="font-medium tabular-nums">{Number(radixTree.total_nodes || 0).toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Total Blocks</div>
                  <div className="font-medium tabular-nums">{Number(radixTree.total_blocks || 0).toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Total Tokens</div>
                  <div className="font-medium tabular-nums">{Number(radixTree.total_tokens || 0).toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Active Refs</div>
                  <div className="font-medium tabular-nums">{Number(radixTree.total_ref_count || 0).toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Leaf Count</div>
                  <div className="font-medium tabular-nums">{Number(radixTree.leaf_count || 0).toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Max Depth</div>
                  <div className="font-medium tabular-nums">{Number(radixTree.max_depth || 0)}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Eviction Strategy</div>
                  <div className="font-medium">{String(radixTree.eviction_strategy || "lru").toUpperCase()}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Evicted (Total)</div>
                  <div className="font-medium tabular-nums">
                    {Number((radixTree.eviction_stats as Record<string, number>)?.total_freed_blocks || 0).toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Hardware Profile */}
          {hwProfile && !hwProfile.error && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Cpu className="w-4 h-4 text-[var(--color-accent)]" />
                Hardware Profile
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Chip</div>
                  <div className="font-medium text-xs">{String(hwProfile.chip_name || "—")}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Generation</div>
                  <div className="font-medium">{String(hwProfile.chip_generation || "—")} {String(hwProfile.chip_tier || "")}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Total Memory</div>
                  <div className="font-medium tabular-nums">{Number(hwProfile.total_memory_gb || 0).toFixed(1)} GB</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Working Set</div>
                  <div className="font-medium tabular-nums">{Number(hwProfile.working_set_gb || 0).toFixed(1)} GB</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">GPU Cores</div>
                  <div className="font-medium tabular-nums">{String(hwProfile.gpu_cores || "—")}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">MLX / MLX-LM</div>
                  <div className="font-medium text-xs">{String(hwProfile.mlx_version || "—")} / {String(hwProfile.mlx_lm_version || "—")}</div>
                </div>
              </div>
              {(hwProfile.adaptive_defaults as Record<string, unknown>) && (
                <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
                  <div className="text-xs text-[var(--color-text-secondary)] mb-2">Adaptive Defaults</div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    {Object.entries(hwProfile.adaptive_defaults as Record<string, unknown>).map(([k, v]) => (
                      <div key={k}>
                        <div className="text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                        <div className="font-medium tabular-nums">
                          {typeof v === "number" ? (v > 1024 * 1024 ? fmtBytes(v) : v.toLocaleString()) : String(v)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Mesh Topology */}
          {meshStatus && meshStatus.topology_type && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Cpu className="w-4 h-4 text-[var(--color-accent)]" />
                Mesh Topology
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Topology</div>
                  <div className="font-medium">{String(meshStatus.topology_type || "—").toUpperCase()}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Node Count</div>
                  <div className="font-medium tabular-nums">{Number(meshStatus.node_count || 0)}</div>
                </div>
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Backend</div>
                  <div className="font-medium">{String(meshStatus.backend || "—")}</div>
                </div>
              </div>
            </div>
          )}

          {/* Engine Tuning — Auto-Tuner + SLO + Profiler */}
          {engineTuning && (engineTuning.engines as Record<string, unknown>[])?.length > 0 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-4">
              <h3 className="font-semibold text-sm flex items-center gap-2">
                <Zap className="w-4 h-4 text-[var(--color-accent)]" />
                Auto-Tuner & Profiling
              </h3>
              {(engineTuning.engines as Record<string, unknown>[]).map((eng: Record<string, unknown>, idx: number) => {
                const tuner = eng.auto_tuner as Record<string, unknown> | undefined;
                const slo = eng.slo as Record<string, unknown> | undefined;
                const profiler = eng.profiler as Record<string, unknown> | undefined;
                const schedulerProfiling = eng.scheduler_profiling as Record<string, unknown> | undefined;
                return (
                  <div key={idx} className="space-y-3">
                    <div className="text-xs text-[var(--color-text-secondary)]">{String(eng.model_id || "default")}</div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                      {tuner && Object.entries(tuner).map(([k, v]) => (
                        <div key={k}>
                          <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}</div>
                          <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                        </div>
                      ))}
                      {slo && Object.entries(slo).map(([k, v]) => (
                        <div key={k}>
                          <div className="text-xs text-[var(--color-text-secondary)]">SLO: {k.replace(/_/g, " ")}</div>
                          <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                        </div>
                      ))}
                      {schedulerProfiling && Object.entries(schedulerProfiling).slice(0, 6).map(([k, v]) => (
                        <div key={k}>
                          <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                          <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Engine Stats — TurboQuant / SpecPrefill / Checkpoint / Warmup */}
          {modelStats && (modelStats.data as Record<string, unknown>[])?.length > 0 && (() => {
            const models = (modelStats.data as Record<string, unknown>[]).filter(
              (m) => m.stats && typeof m.stats === "object"
            );
            if (models.length === 0) return null;
            return (
              <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-4">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <Thermometer className="w-4 h-4 text-[var(--color-accent)]" />
                  Engine Optimizations
                </h3>
                {models.map((model: Record<string, unknown>, idx: number) => {
                  const stats = model.stats as Record<string, unknown>;
                  const turboQuant = stats.turbo_quant as Record<string, unknown> | undefined;
                  const specPrefill = stats.spec_prefill_engine as Record<string, unknown> | undefined;
                  const checkpoint = stats.checkpoint as Record<string, unknown> | undefined;
                  const warmup = (stats.model_optimizations as Record<string, unknown>)?.warmup as Record<string, unknown> | undefined;
                  const kvQuant = stats.kv_prefix_compression as Record<string, unknown> | undefined;
                  const hybridKV = stats.hybrid_kv as Record<string, unknown> | undefined;
                  const hasAny = turboQuant || specPrefill || checkpoint || warmup || kvQuant || hybridKV;
                  if (!hasAny) return null;
                  return (
                    <div key={idx} className="space-y-3">
                      <div className="text-xs text-[var(--color-text-secondary)] font-medium">{String(model.id || "default")}</div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                        {turboQuant && Object.entries(turboQuant).slice(0, 6).map(([k, v]) => (
                          <div key={`tq-${k}`}>
                            <div className="text-xs text-[var(--color-text-secondary)]">TurboQuant: {k.replace(/_/g, " ")}</div>
                            <div className="font-medium tabular-nums">{typeof v === "number" ? (v > 1024 ? fmtBytes(v) : v.toLocaleString()) : String(v)}</div>
                          </div>
                        ))}
                        {specPrefill && Object.entries(specPrefill).slice(0, 6).map(([k, v]) => (
                          <div key={`sp-${k}`}>
                            <div className="text-xs text-[var(--color-text-secondary)]">SpecPrefill: {k.replace(/_/g, " ")}</div>
                            <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                          </div>
                        ))}
                        {checkpoint && Object.entries(checkpoint).slice(0, 6).map(([k, v]) => (
                          <div key={`cp-${k}`}>
                            <div className="text-xs text-[var(--color-text-secondary)]">Checkpoint: {k.replace(/_/g, " ")}</div>
                            <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                          </div>
                        ))}
                        {warmup && Object.entries(warmup).slice(0, 6).map(([k, v]) => (
                          <div key={`wu-${k}`}>
                            <div className="text-xs text-[var(--color-text-secondary)]">Warmup: {k.replace(/_/g, " ")}</div>
                            <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                          </div>
                        ))}
                        {kvQuant && Object.entries(kvQuant).slice(0, 6).map(([k, v]) => (
                          <div key={`kvq-${k}`}>
                            <div className="text-xs text-[var(--color-text-secondary)]">KV Compress: {k.replace(/_/g, " ")}</div>
                            <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                          </div>
                        ))}
                        {hybridKV && Object.entries(hybridKV).slice(0, 4).map(([k, v]) => (
                          <div key={`hkv-${k}`}>
                            <div className="text-xs text-[var(--color-text-secondary)]">HybridKV: {k.replace(/_/g, " ")}</div>
                            <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })()}

          {/* Health Dashboard */}
          {healthDashboard && healthDashboard.enabled !== false && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-3">
              <h3 className="font-semibold text-sm flex items-center gap-2">
                <Shield className="w-4 h-4 text-[var(--color-accent)]" />
                Health Dashboard
              </h3>
              {healthDashboard.score !== undefined && (
                <div className="flex items-center gap-4">
                  <div className={`text-3xl font-bold ${
                    Number(healthDashboard.score) >= 80 ? "text-[var(--color-success)]" :
                    Number(healthDashboard.score) >= 50 ? "text-amber-500" : "text-[var(--color-danger)]"
                  }`}>
                    {Number(healthDashboard.score).toFixed(0)}
                  </div>
                  <div className="text-sm text-[var(--color-text-secondary)]">/ 100 health score</div>
                </div>
              )}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {Object.entries(healthDashboard).filter(([k]) => !["enabled", "error", "reason", "score"].includes(k)).map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">
                      {k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </div>
                    <div className="font-medium tabular-nums">
                      {typeof v === "number" ? v.toLocaleString() : typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Data Parallel */}
          {dataParallel && dataParallel.active && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Network className="w-4 h-4 text-[var(--color-accent)]" />
                Data Parallel
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {Object.entries(dataParallel).filter(([k]) => k !== "active").map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">
                      {k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </div>
                    <div className="font-medium tabular-nums">
                      {typeof v === "number" ? v.toLocaleString() : typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Per-Model Metrics */}
          {perModel && !perModel.error && perModel._summary && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-4">
              <h3 className="font-semibold text-sm flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-[var(--color-accent)]" />
                Per-Model Metrics
              </h3>
              {Object.entries(perModel).filter(([k]) => k !== "_summary" && k !== "error").map(([modelId, data]) => (
                <div key={modelId}>
                  <div className="text-xs text-[var(--color-text-secondary)] font-medium mb-2">{modelId}</div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                    {Object.entries(data as Record<string, unknown>).slice(0, 8).map(([k, v]) => (
                      <div key={k}>
                        <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                        <div className="font-medium tabular-nums">
                          {typeof v === "number" ? (v > 100000 ? fmtBytes(v) : v.toLocaleString()) : String(v)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Auto-Tuner State */}
          {autoTuner && autoTuner.models && (autoTuner.models as Record<string, unknown>[]).length > 0 && (() => {
            const models = (autoTuner.models as Record<string, unknown>[]).filter((m) => m.auto_tuner || m.profiler || m.slo);
            if (models.length === 0) return null;
            return (
              <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-4">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <Wrench className="w-4 h-4 text-[var(--color-accent)]" />
                  Auto-Tuner State
                </h3>
                {models.map((model: Record<string, unknown>, idx: number) => {
                  const tuner = model.auto_tuner as Record<string, unknown> | undefined;
                  const profiler = model.profiler as Record<string, unknown> | undefined;
                  const slo = model.slo as Record<string, unknown> | undefined;
                  return (
                    <div key={idx} className="space-y-2">
                      <div className="text-xs text-[var(--color-text-secondary)] font-medium">{String(model.model_id || "default")}</div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                        {tuner && Object.entries(tuner).slice(0, 6).map(([k, v]) => (
                          <div key={`at-${k}`}>
                            <div className="text-xs text-[var(--color-text-secondary)]">Tuner: {k.replace(/_/g, " ")}</div>
                            <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                          </div>
                        ))}
                        {profiler && Object.entries(profiler).slice(0, 4).map(([k, v]) => (
                          <div key={`pf-${k}`}>
                            <div className="text-xs text-[var(--color-text-secondary)]">Profiler: {k.replace(/_/g, " ")}</div>
                            <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                          </div>
                        ))}
                        {slo && Object.entries(slo).slice(0, 4).map(([k, v]) => (
                          <div key={`slo-${k}`}>
                            <div className="text-xs text-[var(--color-text-secondary)]">SLO: {k.replace(/_/g, " ")}</div>
                            <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })()}

          {/* Memory Pressure */}
          {memoryPressure && memoryPressure.active && (memoryPressure.models as Record<string, unknown>[])?.length > 0 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-3">
              <h3 className="font-semibold text-sm flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-[var(--color-accent)]" />
                Memory Pressure
              </h3>
              {(memoryPressure.models as Record<string, unknown>[]).map((model: Record<string, unknown>, idx: number) => (
                <div key={idx}>
                  <div className="text-xs text-[var(--color-text-secondary)] font-medium mb-2">{String(model.model_id)}</div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                    {Object.entries(model).filter(([k]) => k !== "model_id").map(([sectionKey, sectionVal]) => {
                      if (typeof sectionVal === "object" && sectionVal !== null) {
                        return Object.entries(sectionVal as Record<string, unknown>).map(([k, v]) => (
                          <div key={`${sectionKey}-${k}`}>
                            <div className="text-xs text-[var(--color-text-secondary)]">
                              {sectionKey.replace(/_/g, " ")}: {k.replace(/_/g, " ")}
                            </div>
                            <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                          </div>
                        ));
                      }
                      return (
                        <div key={sectionKey}>
                          <div className="text-xs text-[var(--color-text-secondary)]">{sectionKey.replace(/_/g, " ")}</div>
                          <div className="font-medium tabular-nums">{typeof sectionVal === "number" ? sectionVal.toLocaleString() : String(sectionVal)}</div>
                        </div>
                      );
                    }).flat()}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Thinking Segments */}
          {thinkingSegments && thinkingSegments.active && (thinkingSegments.models as Record<string, unknown>[])?.length > 0 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Brain className="w-4 h-4 text-[var(--color-accent)]" />
                Thinking Segments
              </h3>
              {(thinkingSegments.models as Record<string, unknown>[]).map((model: Record<string, unknown>, idx: number) => (
                <div key={idx} className="mb-2 last:mb-0">
                  <div className="text-xs text-[var(--color-text-secondary)] mb-1">{String(model.model_id)}</div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                    {Object.entries(model).filter(([k]) => k !== "model_id").map(([k, v]) => (
                      <div key={k}>
                        <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                        <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Reasoning Tokens */}
          {reasoningTokens && reasoningTokens.total_reasoning_tokens !== undefined && Number(reasoningTokens.total_reasoning_tokens) > 0 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Brain className="w-4 h-4 text-[var(--color-accent)]" />
                Reasoning Tokens
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-xs text-[var(--color-text-secondary)]">Total Reasoning Tokens</div>
                  <div className="font-medium tabular-nums">{Number(reasoningTokens.total_reasoning_tokens).toLocaleString()}</div>
                </div>
                {(reasoningTokens.engines as Record<string, unknown>[])?.map((eng: Record<string, unknown>, idx: number) => (
                  <div key={idx}>
                    <div className="text-xs text-[var(--color-text-secondary)]">{String(eng.model_id)}</div>
                    <div className="font-medium tabular-nums">{Number(eng.reasoning_tokens).toLocaleString()}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Metal Kernels */}
          {metalKernels && metalKernels.active && (metalKernels.models as Record<string, unknown>[])?.length > 0 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Cpu className="w-4 h-4 text-[var(--color-accent)]" />
                Metal Kernels
              </h3>
              {(metalKernels.models as Record<string, unknown>[]).map((model: Record<string, unknown>, idx: number) => (
                <div key={idx} className="mb-2 last:mb-0">
                  <div className="text-xs text-[var(--color-text-secondary)] mb-1">{String(model.model_id)}</div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                    <div>
                      <div className="text-xs text-[var(--color-text-secondary)]">Enabled</div>
                      <div className="font-medium">{model.enabled ? "Yes" : "No"}</div>
                    </div>
                    {Object.entries(model).filter(([k]) => !["model_id", "enabled"].includes(k)).map(([k, v]) => (
                      <div key={k}>
                        <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                        <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* ANE Embeddings */}
          {aneEmbeddings && aneEmbeddings.enabled && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Cpu className="w-4 h-4 text-[var(--color-accent)]" />
                ANE Embeddings
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {Object.entries(aneEmbeddings).filter(([k]) => !["enabled", "error"].includes(k)).map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                    <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* External Prefill */}
          {externalPrefill && externalPrefill.enabled && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <ArrowRightLeft className="w-4 h-4 text-[var(--color-accent)]" />
                External Prefill
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {Object.entries(externalPrefill).filter(([k]) => !["enabled", "error"].includes(k)).map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                    <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Response Cache */}
          {responseCache && responseCache.cache_module && (responseCache.cache_module as Record<string, unknown>).enabled !== false && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Layers className="w-4 h-4 text-[var(--color-accent)]" />
                Response Cache
              </h3>
              {((responseCache.engines as Record<string, unknown>[]) || []).length > 0 && (
                <div className="mb-3">
                  {(responseCache.engines as Record<string, unknown>[]).map((eng: Record<string, unknown>, idx: number) => {
                    const rc = eng.response_cache as Record<string, unknown>;
                    return (
                      <div key={idx} className="mb-2 last:mb-0">
                        <div className="text-xs text-[var(--color-text-secondary)] mb-1">{String(eng.model_id)}</div>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                          {rc && Object.entries(rc).map(([k, v]) => (
                            <div key={k}>
                              <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                              <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              {responseCache.cache_module && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                  {Object.entries(responseCache.cache_module as Record<string, unknown>).map(([k, v]) => (
                    <div key={k}>
                      <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                      <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Inflight Prefix Sharing */}
          {inflightPrefix && inflightPrefix.enabled && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <GitBranch className="w-4 h-4 text-[var(--color-accent)]" />
                Inflight Prefix Sharing
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {Object.entries(inflightPrefix).filter(([k]) => k !== "enabled").map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                    <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Request Coalescer */}
          {requestCoalescer && requestCoalescer.enabled && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Radio className="w-4 h-4 text-[var(--color-accent)]" />
                Request Coalescer
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {Object.entries(requestCoalescer).filter(([k]) => k !== "enabled").map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                    <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Token Scheduler */}
          {tokenScheduler && tokenScheduler.enabled !== false && (tokenScheduler.token_scheduler || tokenScheduler.priority_inversion || tokenScheduler.fairness) && (() => {
            const ts = tokenScheduler.token_scheduler as Record<string, unknown> | undefined;
            const pi = tokenScheduler.priority_inversion as Record<string, unknown> | undefined;
            const fair = tokenScheduler.fairness as Record<string, unknown> | undefined;
            if (!ts && !pi && !fair) return null;
            return (
              <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-3">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-[var(--color-accent)]" />
                  Token Scheduler &amp; Fairness
                </h3>
                {ts && (
                  <div>
                    <div className="text-xs text-[var(--color-text-secondary)] font-medium mb-1">Token Scheduler</div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                      {Object.entries(ts).slice(0, 8).map(([k, v]) => (
                        <div key={k}>
                          <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                          <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {pi && (
                  <div>
                    <div className="text-xs text-[var(--color-text-secondary)] font-medium mb-1">Priority Inversion</div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                      {Object.entries(pi).slice(0, 6).map(([k, v]) => (
                        <div key={k}>
                          <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                          <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {fair && (
                  <div>
                    <div className="text-xs text-[var(--color-text-secondary)] font-medium mb-1">Fairness</div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                      {Object.entries(fair).slice(0, 6).map(([k, v]) => (
                        <div key={k}>
                          <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                          <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })()}

          {/* KV Migration */}
          {kvMigration && kvMigration.enabled !== false && Object.keys(kvMigration).length > 1 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <ArrowRightLeft className="w-4 h-4 text-[var(--color-accent)]" />
                KV Migration (Multi-Tier)
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                {Object.entries(kvMigration).filter(([k]) => !["enabled", "reason"].includes(k)).map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                    <div className="font-medium tabular-nums">
                      {typeof v === "number" ? (v > 1024 * 1024 ? fmtBytes(v) : v.toLocaleString()) : typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Attention Eviction (H2O) */}
          {attentionEviction && attentionEviction.enabled !== false && (attentionEviction.models as Record<string, unknown>[])?.length > 0 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Eye className="w-4 h-4 text-[var(--color-accent)]" />
                Attention Eviction (H2O)
              </h3>
              {(attentionEviction.models as Record<string, unknown>[]).map((model: Record<string, unknown>, idx: number) => (
                <div key={idx} className="mb-2 last:mb-0">
                  <div className="text-xs text-[var(--color-text-secondary)] mb-1">{String(model.model_id)}</div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                    {Object.entries(model).filter(([k]) => k !== "model_id").map(([k, v]) => (
                      <div key={k}>
                        <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                        <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Batch Size Distribution */}
          {batchSize && batchSize.enabled !== false && Object.keys(batchSize).length > 1 && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <h3 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <BarChart3 className="w-4 h-4 text-[var(--color-accent)]" />
                Batch Size Distribution
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                {Object.entries(batchSize).filter(([k]) => !["enabled", "reason"].includes(k)).map(([k, v]) => (
                  <div key={k}>
                    <div className="text-xs text-[var(--color-text-secondary)]">{k.replace(/_/g, " ")}</div>
                    <div className="font-medium tabular-nums">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function MemBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-[var(--color-text-secondary)]">{label}</span>
        <span className="tabular-nums">{fmtBytes(value)} ({pct.toFixed(1)}%)</span>
      </div>
      <div className="w-full bg-[var(--color-bg-tertiary)] rounded-full h-1.5">
        <div
          className={`${color} h-1.5 rounded-full transition-all duration-700`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  );
}

function GaugeCircle({ pct }: { pct: number }) {
  const r = 36;
  const c = 2 * Math.PI * r;
  const offset = c - (pct / 100) * c;
  return (
    <div className="relative w-24 h-24 shrink-0">
      <svg viewBox="0 0 80 80" className="w-full h-full -rotate-90">
        <circle cx="40" cy="40" r={r} fill="none" stroke="var(--color-bg-tertiary)" strokeWidth="6" />
        <circle
          cx="40" cy="40" r={r}
          fill="none"
          stroke="var(--color-accent)"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          className="transition-all duration-700"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-lg font-bold tabular-nums">{pct.toFixed(0)}%</span>
      </div>
    </div>
  );
}

function KV({ icon: Icon, label, value }: { icon: typeof Cpu; label: string; value: string }) {
  return (
    <div className="flex items-start gap-2">
      <Icon className="w-3.5 h-3.5 text-[var(--color-text-secondary)] mt-0.5 shrink-0" />
      <div>
        <div className="text-xs text-[var(--color-text-secondary)]">{label}</div>
        <div className="font-medium tabular-nums">{value}</div>
      </div>
    </div>
  );
}
