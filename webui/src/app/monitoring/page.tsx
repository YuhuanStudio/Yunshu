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
        const [sysRes, engRes] = await Promise.all([
          fetch("/api/v1/monitoring/system"),
          fetch("/api/v1/monitoring/engine"),
        ]);
        let sysData: SystemStats | null = null;
        let engData: Record<string, unknown> | null = null;
        if (sysRes.ok) {
          sysData = await sysRes.json();
          if (mounted.current) {
            setSystem(sysData);
            setGpuHistory((prev) => [...prev.slice(-(MAX_HISTORY - 1)), sysData!.gpu?.utilization_pct ?? 0]);
          }
        }
        if (engRes.ok) {
          engData = await engRes.json();
          if (mounted.current) setEngine(engData);
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
    const id = setInterval(fetchData, 3000);
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
