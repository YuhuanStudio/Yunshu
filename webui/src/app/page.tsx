"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Button,
  Card,
  Input,
  AreaChart,
  SegmentedBar,
  Spinner,
  Badge,
  StatusIndicator,
} from "yunui";
import { StatCard } from "yunui/patterns";
import { Activity, Cpu, Clock, Download, RefreshCw, Zap } from "lucide-react";
import { api, usePolling } from "@/lib/api";
import { fmtBytes, fmtNumber, fmtDuration, fmtPct } from "@/lib/format";
import { ModelTypeChip, guessModelType } from "@/lib/model-type";
import type { EngineStats, SystemStats, Model } from "@/lib/types";

const HISTORY_LEN = 30;

export default function DashboardPage() {
  const engine = usePolling<EngineStats>((s) => api.get("/api/v1/monitoring/engine", s), 5000);
  const system = usePolling<SystemStats>((s) => api.get("/api/v1/monitoring/system", s), 5000);
  const models = usePolling<{ data: Model[] }>((s) => api.get("/v1/models", s), 5000);

  const [reqHistory, setReqHistory] = useState<number[]>([]);
  const [gpuHistory, setGpuHistory] = useState<number[]>([]);
  const [loadInput, setLoadInput] = useState("");
  const [busy, setBusy] = useState<string | null>(null);

  const eng = engine.data;
  const sys = system.data;

  useEffect(() => {
    if (eng) setReqHistory((h) => [...h.slice(-(HISTORY_LEN - 1)), eng.active_requests ?? 0]);
  }, [eng]);
  useEffect(() => {
    if (sys) setGpuHistory((h) => [...h.slice(-(HISTORY_LEN - 1)), sys.gpu?.utilization_pct ?? 0]);
  }, [sys]);

  const refreshAll = useCallback(() => {
    engine.refresh();
    system.refresh();
    models.refresh();
  }, [engine, system, models]);

  const loadModel = async (id: string) => {
    if (!id) return;
    setBusy(id);
    try {
      await api.post("/v1/models/load", { model: id });
      setLoadInput("");
      refreshAll();
    } catch {
      /* surfaced via connection status */
    } finally {
      setBusy(null);
    }
  };

  const unloadModel = async (id: string) => {
    setBusy(id);
    try {
      await api.post(`/v1/models/unload/${encodeURIComponent(id)}`);
      refreshAll();
    } finally {
      setBusy(null);
    }
  };

  const gpu = sys?.gpu ?? eng?.gpu_memory ?? null;
  const modelList = models.data?.data ?? [];

  return (
    <div className="page-enter px-6 py-6 sm:px-8">
      <div className="mx-auto max-w-7xl">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h1 className="heading-xl">Dashboard</h1>
            <p className="text-body mt-1">Live engine, GPU and model status.</p>
          </div>
          <Button variant="secondary" size="sm" onClick={refreshAll}>
            <RefreshCw className="h-4 w-4" /> Refresh
          </Button>
        </div>

        {/* Stat grid */}
        <div className="mt-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatCard
            icon={Activity}
            label="Active requests"
            value={eng ? fmtNumber(eng.active_requests) : "—"}
            subtext={eng ? `${fmtNumber(eng.waiting_requests)} waiting` : undefined}
          />
          <StatCard
            icon={Zap}
            label="Processed"
            value={eng ? fmtNumber(eng.requests_processed) : "—"}
            subtext={eng ? `${fmtNumber(eng.total_completion_tokens)} out tokens` : undefined}
            tone="emerald"
          />
          <StatCard
            icon={Cpu}
            label="GPU utilization"
            value={gpu ? fmtPct(gpu.utilization_pct) : "—"}
            subtext={gpu ? `${fmtBytes(gpu.active_bytes)} active` : undefined}
            tone="blue"
          />
          <StatCard
            icon={Clock}
            label="Uptime"
            value={eng ? fmtDuration(eng.uptime_seconds) : "—"}
            subtext={sys ? `MLX ${sys.mlx_version}` : undefined}
            tone="purple"
          />
        </div>

        {/* Charts row */}
        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card className="p-5">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium">Request throughput</span>
              <Badge variant="info">{eng ? fmtNumber(eng.active_requests) : 0} now</Badge>
            </div>
            <AreaChart data={reqHistory} height={72} showGrid={false} formatValue={fmtNumber} ariaLabel="Request throughput" />
          </Card>
          <Card className="p-5">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium">GPU utilization</span>
              <Badge variant="info">{gpu ? fmtPct(gpu.utilization_pct) : "0%"}</Badge>
            </div>
            <AreaChart data={gpuHistory} tone="success" height={72} showGrid={false} formatValue={fmtPct} ariaLabel="GPU utilization" />
          </Card>
        </div>

        {/* GPU memory */}
        {gpu && (
          <Card className="mt-4 p-5">
            <div className="mb-3 text-sm font-medium">GPU memory</div>
            <SegmentedBar
              total={gpu.total_bytes}
              legend
              formatValue={(v) => fmtBytes(v)}
              segments={[
                { value: gpu.active_bytes, tone: "accent", label: "Active" },
                { value: gpu.cache_bytes, tone: "info", label: "Cache" },
                { value: Math.max(0, gpu.available_bytes), tone: "neutral", label: "Free" },
              ]}
            />
          </Card>
        )}

        {/* Quick load + models */}
        <Card className="mt-4 p-5">
          <div className="mb-4 flex items-center justify-between gap-3">
            <span className="text-sm font-medium">Models</span>
            <div className="flex w-full max-w-md items-center gap-2">
              <Input
                placeholder="Load a model by id…"
                value={loadInput}
                onChange={(e) => setLoadInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && loadModel(loadInput.trim())}
              />
              <Button
                size="sm"
                onClick={() => loadModel(loadInput.trim())}
                disabled={!loadInput.trim() || busy === loadInput.trim()}
              >
                <Download className="h-4 w-4" /> Load
              </Button>
            </div>
          </div>

          {models.loading && modelList.length === 0 ? (
            <div className="flex justify-center py-8">
              <Spinner />
            </div>
          ) : (
            <div className="divide-y divide-border">
              {modelList.map((m) => {
                const type = m.type ?? guessModelType(m.id);
                return (
                  <div key={m.id} className="flex items-center gap-3 py-2.5">
                    <ModelTypeChip type={type} />
                    <span className="min-w-0 flex-1 truncate text-sm">{m.id}</span>
                    {m.loaded ? (
                      <>
                        <StatusIndicator status="online">
                          <span className="text-success">Loaded</span>
                        </StatusIndicator>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => unloadModel(m.id)}
                          disabled={busy === m.id}
                        >
                          Unload
                        </Button>
                      </>
                    ) : (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => loadModel(m.id)}
                        disabled={busy === m.id}
                      >
                        Load
                      </Button>
                    )}
                  </div>
                );
              })}
              {modelList.length === 0 && (
                <p className="py-6 text-center text-sm text-muted-foreground">No models found.</p>
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
