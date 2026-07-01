"use client";

import { useEffect, useState, type ReactNode } from "react";
import {
  Card,
  Badge,
  Gauge,
  AreaChart,
  SegmentedBar,
  Spinner,
  Alert,
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
  cn,
} from "yunui";
import { StatCard } from "yunui/patterns";
import { Cpu, MemoryStick, Activity, Clock, ChevronDown, Layers, Server } from "lucide-react";
import { api, usePolling } from "@/lib/api";
import { fmtBytes, fmtNumber, fmtPct, fmtDuration } from "@/lib/format";
import type { EngineStats, SystemStats, RadixTreeStats, HardwareProfile } from "@/lib/types";

const HISTORY_LEN = 40;

/** Tone for a utilization-style percentage. */
function pctTone(pct: number): "success" | "warning" | "error" {
  if (pct >= 90) return "error";
  if (pct >= 70) return "warning";
  return "success";
}

function renderValue(v: unknown): ReactNode {
  if (v == null) return "—";
  if (typeof v === "boolean") return <Badge variant={v ? "success" : "default"}>{v ? "on" : "off"}</Badge>;
  if (typeof v === "number") return Number.isInteger(v) ? fmtNumber(v) : v.toFixed(2);
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return `[${v.length}]`;
  return <span className="text-muted-foreground">{"{…}"}</span>;
}

/** Render a flat object as a label/value grid. */
function MetricGroup({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data ?? {});
  if (entries.length === 0) return <p className="text-sm text-muted-foreground">No data.</p>;
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
      {entries.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <div className="truncate text-xs uppercase tracking-wide text-muted-foreground">{k.replace(/_/g, " ")}</div>
          <div className="truncate text-sm font-medium tabular-nums">{renderValue(v)}</div>
        </div>
      ))}
    </div>
  );
}

function Section({ title, icon, children, defaultOpen = false }: { title: string; icon?: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card className="p-0">
        <CollapsibleTrigger className="flex w-full items-center gap-2 px-5 py-3.5 text-left">
          {icon}
          <span className="text-sm font-medium capitalize">{title}</span>
          <ChevronDown className={cn("ml-auto h-4 w-4 text-muted-foreground transition-transform", open && "rotate-180")} />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="border-t border-border px-5 py-4">{children}</div>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}

function Info({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="truncate text-sm font-medium">{value ?? "—"}</div>
    </div>
  );
}

export default function MonitoringPage() {
  const engine = usePolling<EngineStats>((s) => api.get("/api/v1/monitoring/engine", s), 5000);
  const system = usePolling<SystemStats>((s) => api.get("/api/v1/monitoring/system", s), 5000);
  const radix = usePolling<RadixTreeStats>((s) => api.get("/api/v1/admin/radix-tree", s), 5000);
  const hw = usePolling<HardwareProfile>((s) => api.get("/api/v1/admin/hardware-profile", s), 5000);
  const all = usePolling<Record<string, unknown>>((s) => api.get("/api/v1/gw/monitoring/all", s), 5000);

  const [gpuHist, setGpuHist] = useState<number[]>([]);
  const sys = system.data;
  const eng = engine.data;
  const gpu = sys?.gpu ?? eng?.gpu_memory ?? null;

  useEffect(() => {
    if (gpu) setGpuHist((h) => [...h.slice(-(HISTORY_LEN - 1)), gpu.utilization_pct ?? 0]);
  }, [gpu]);

  if (system.loading && !sys) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }

  const memUsedPct = sys ? (sys.memory_used_bytes / sys.memory_total_bytes) * 100 : 0;
  const hitRate = radix.data && radix.data.match_total > 0 ? (radix.data.match_hits / radix.data.match_total) * 100 : 0;

  return (
    <div className="page-enter px-6 py-6 sm:px-8">
      <div className="mx-auto max-w-7xl space-y-4">
        <div>
          <h1 className="heading-xl">Monitoring</h1>
          <p className="text-body mt-1">Engine, GPU, cache and hardware telemetry — refreshed every 5s.</p>
        </div>

        {/* Top gauges */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <Card className="flex items-center gap-5 p-5">
            <Gauge value={gpu?.utilization_pct ?? 0} tone={pctTone(gpu?.utilization_pct ?? 0)} size={96} thickness={8} />
            <div className="min-w-0">
              <div className="text-sm font-medium">GPU utilization</div>
              <div className="mt-1 text-xs text-muted-foreground">{gpu ? `${fmtBytes(gpu.active_bytes)} active` : "—"}</div>
              <div className="mt-3 w-40">
                <AreaChart data={gpuHist} tone="success" height={44} showGrid={false} formatValue={fmtPct} ariaLabel="GPU utilization history" />
              </div>
            </div>
          </Card>

          <Card className="flex items-center gap-5 p-5">
            <Gauge value={memUsedPct} tone={pctTone(memUsedPct)} size={96} thickness={8} />
            <div className="min-w-0">
              <div className="text-sm font-medium">System memory</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {sys ? `${fmtBytes(sys.memory_used_bytes)} / ${fmtBytes(sys.memory_total_bytes)}` : "—"}
              </div>
              <div className="mt-1 text-xs text-muted-foreground">CPU {sys ? fmtPct(sys.cpu_percent) : "—"}</div>
            </div>
          </Card>

          <Card className="flex items-center gap-5 p-5">
            <Gauge value={hitRate} tone={hitRate >= 50 ? "success" : "warning"} size={96} thickness={8} />
            <div className="min-w-0">
              <div className="text-sm font-medium">RadixTree hit rate</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {radix.data ? `${fmtNumber(radix.data.match_hits)} / ${fmtNumber(radix.data.match_total)}` : "—"}
              </div>
              {radix.data && <div className="mt-1 text-xs text-muted-foreground">{fmtNumber(radix.data.total_nodes)} nodes</div>}
            </div>
          </Card>
        </div>

        {/* Engine stat grid */}
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatCard icon={Activity} label="Active" value={eng ? fmtNumber(eng.active_requests) : "—"} subtext={eng ? `${fmtNumber(eng.waiting_requests)} waiting` : undefined} />
          <StatCard icon={Server} label="Processed" value={eng ? fmtNumber(eng.requests_processed) : "—"} tone="emerald" />
          <StatCard icon={Cpu} label="Out tokens" value={eng ? fmtNumber(eng.total_completion_tokens) : "—"} tone="blue" />
          <StatCard icon={Clock} label="Uptime" value={eng ? fmtDuration(eng.uptime_seconds) : "—"} tone="purple" />
        </div>

        {/* GPU memory breakdown */}
        {gpu && (
          <Card className="p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium">
              <MemoryStick className="h-4 w-4 text-muted-foreground" /> GPU memory
            </div>
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
            <div className="mt-3 text-xs text-muted-foreground">Peak {fmtBytes(gpu.peak_bytes)}</div>
          </Card>
        )}

        {/* Hardware profile */}
        {hw.data && !hw.data.error && (
          <Section title="Hardware profile" icon={<Cpu className="h-4 w-4 text-muted-foreground" />} defaultOpen>
            <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
              <Info label="Chip" value={hw.data.chip_name} />
              <Info label="Generation" value={hw.data.chip_generation} />
              <Info label="Tier" value={hw.data.chip_tier} />
              <Info label="GPU cores" value={hw.data.gpu_cores} />
              <Info label="Total memory" value={`${hw.data.total_memory_gb} GB`} />
              <Info label="Working set" value={`${hw.data.working_set_gb} GB`} />
              <Info label="MLX" value={hw.data.mlx_version} />
              <Info label="mlx-lm" value={hw.data.mlx_lm_version} />
            </div>
          </Section>
        )}

        {/* RadixTree detail */}
        {radix.data && (
          <Section title="RadixTree cache" icon={<Layers className="h-4 w-4 text-muted-foreground" />}>
            <MetricGroup data={radix.data as unknown as Record<string, unknown>} />
          </Section>
        )}

        {/* Aggregated engine subsystems — generic render so nothing is dropped */}
        {all.error && <Alert variant="warning">Aggregated metrics unavailable.</Alert>}
        {all.data &&
          Object.entries(all.data)
            .filter(([, v]) => v && typeof v === "object" && !Array.isArray(v))
            .map(([key, v]) => (
              <Section key={key} title={key.replace(/_/g, " ")} icon={<Activity className="h-4 w-4 text-muted-foreground" />}>
                <MetricGroup data={v as Record<string, unknown>} />
              </Section>
            ))}
      </div>
    </div>
  );
}
