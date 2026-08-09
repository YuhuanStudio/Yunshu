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
import { Cpu, MemoryStick, Activity, Clock, ChevronDown, Server } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { api, usePolling } from "@/lib/api";
import { fmtBytes, fmtNumber, fmtPct, fmtDuration } from "@/lib/format";
import type { EngineStats, SystemStats, RadixTreeStats, HealthStatus } from "@/lib/types";

const HISTORY_LEN = 40;

/** Tone for a utilization-style percentage. */
function pctTone(pct: number): "success" | "warning" | "error" {
  if (pct >= 90) return "error";
  if (pct >= 70) return "warning";
  return "success";
}

/**
 * Best-effort RadixTree hit-rate from `/radix-tree` (shape `{models:[...]}`).
 * Each model row carries opaque stats; we look for a numeric `hit_rate` or a
 * `match_hits` / `match_total` pair and aggregate. Returns null when nothing
 * derivable is present (so the caller can drop the gauge).
 */
function deriveRadixHitRate(radix: RadixTreeStats | null): number | null {
  if (!radix?.models?.length) return null;
  let hits = 0;
  let total = 0;
  let sawRate = false;
  let rateSum = 0;
  let rateCount = 0;
  for (const m of radix.models) {
    const rec = m as Record<string, unknown>;
    if (typeof rec.hit_rate === "number") {
      sawRate = true;
      rateSum += rec.hit_rate <= 1 ? rec.hit_rate * 100 : rec.hit_rate;
      rateCount += 1;
    }
    if (typeof rec.match_hits === "number" && typeof rec.match_total === "number") {
      hits += rec.match_hits;
      total += rec.match_total;
    }
  }
  if (total > 0) return (hits / total) * 100;
  if (sawRate && rateCount > 0) return rateSum / rateCount;
  return null;
}

/** True for a plain (non-array) object we can recurse into. */
function isRecord(v: unknown): v is Record<string, unknown> {
  return v != null && typeof v === "object" && !Array.isArray(v);
}

function renderScalar(v: unknown): ReactNode {
  if (v == null) return "—";
  if (typeof v === "boolean") return <Badge variant={v ? "success" : "default"}>{v ? "on" : "off"}</Badge>;
  if (typeof v === "number") return Number.isInteger(v) ? fmtNumber(v) : v.toFixed(2);
  if (typeof v === "string") return v || "—";
  // Array of scalars → inline; array of objects → count.
  if (Array.isArray(v)) {
    if (v.length === 0) return <span className="text-muted-foreground">[]</span>;
    if (v.every((x) => x == null || typeof x !== "object")) return v.map((x) => String(x)).join(", ");
    return <span className="text-muted-foreground">{`[${v.length}]`}</span>;
  }
  return <span className="text-muted-foreground">—</span>;
}

/**
 * Render an object as a label/value grid; nested objects recurse into an
 * indented sub-group (up to `depth` levels) so no subsystem field is dropped.
 */
function MetricGroup({ data, depth = 0 }: { data: Record<string, unknown>; depth?: number }) {
  const entries = Object.entries(data ?? {});
  if (entries.length === 0) return <p className="text-sm text-muted-foreground">No data.</p>;
  const scalars = entries.filter(([, v]) => !isRecord(v));
  const nested = entries.filter(([, v]) => isRecord(v)) as [string, Record<string, unknown>][];
  return (
    <div className="space-y-3">
      {scalars.length > 0 && (
        <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
          {scalars.map(([k, v]) => (
            <div key={k} className="min-w-0">
              <div className="truncate text-xs uppercase tracking-wide text-muted-foreground">{k.replace(/_/g, " ")}</div>
              <div className="truncate text-sm font-medium tabular-nums">{renderScalar(v)}</div>
            </div>
          ))}
        </div>
      )}
      {depth < 3 &&
        nested.map(([k, v]) => (
          <div key={k} className="rounded-lg border border-border/60 bg-muted/20 px-3 py-2">
            <div className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">{k.replace(/_/g, " ")}</div>
            <MetricGroup data={v} depth={depth + 1} />
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
  const engine = usePolling<EngineStats>((s) => api.get("/api/v1/gw/monitoring/engine", s), 5000);
  const system = usePolling<SystemStats>((s) => api.get("/api/v1/gw/monitoring/system", s), 5000);
  const radix = usePolling<RadixTreeStats>((s) => api.get("/api/v1/gw/monitoring/radix-tree", s), 5000);
  const health = usePolling<HealthStatus>((s) => api.get("/health", s), 5000);
  const all = usePolling<Record<string, unknown>>((s) => api.get("/api/v1/gw/monitoring/all", s), 5000);

  const [gpuHist, setGpuHist] = useState<number[]>([]);
  const sys = system.data;
  const eng = engine.data;
  const gpu = sys?.gpu ?? null;

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

  const memPct = sys?.memory.percent ?? 0;
  const hitRate = deriveRadixHitRate(radix.data);
  const gpuFree = gpu ? Math.max(0, gpu.total_uma_bytes - gpu.active_bytes - gpu.cache_bytes) : 0;

  return (
    <PageShell
      title="Monitoring"
      description="Engine, GPU, cache and system telemetry — refreshed every 5s."
      className="space-y-4"
    >
        {/* Top gauges */}
        <div className={cn("grid grid-cols-1 gap-4", hitRate != null ? "lg:grid-cols-3" : "lg:grid-cols-2")}>
          <Card className="flex items-center gap-5 p-5">
            <Gauge ariaLabel="GPU utilization" value={gpu?.utilization_pct ?? 0} tone={pctTone(gpu?.utilization_pct ?? 0)} size={96} thickness={8} />
            <div className="min-w-0">
              <div className="text-sm font-medium">GPU utilization</div>
              <div className="mt-1 text-xs text-muted-foreground">{gpu ? `${fmtBytes(gpu.active_bytes)} active` : "—"}</div>
              <div className="mt-3 w-40">
                <AreaChart data={gpuHist} tone="success" height={44} showGrid={false} formatValue={fmtPct} ariaLabel="GPU utilization history" />
              </div>
            </div>
          </Card>

          <Card className="flex items-center gap-5 p-5">
            <Gauge ariaLabel="System memory" value={memPct} tone={pctTone(memPct)} size={96} thickness={8} />
            <div className="min-w-0">
              <div className="text-sm font-medium">System memory</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {sys ? `${fmtBytes(sys.memory.used_bytes)} / ${fmtBytes(sys.memory.total_bytes)}` : "—"}
              </div>
              <div className="mt-1 text-xs text-muted-foreground">CPU {sys ? fmtPct(sys.cpu.percent) : "—"}</div>
            </div>
          </Card>

          {hitRate != null && (
            <Card className="flex items-center gap-5 p-5">
              <Gauge ariaLabel="RadixTree hit rate" value={hitRate} tone={hitRate >= 50 ? "success" : "warning"} size={96} thickness={8} />
              <div className="min-w-0">
                <div className="text-sm font-medium">RadixTree hit rate</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {radix.data ? `${fmtNumber(radix.data.models.length)} model(s)` : "—"}
                </div>
              </div>
            </Card>
          )}
        </div>

        {/* Engine stat grid */}
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatCard icon={Activity} label="Active" value={eng ? fmtNumber(eng.active_requests) : "—"} subtext={eng ? `${fmtNumber(eng.waiting_requests)} waiting` : undefined} />
          <StatCard icon={Server} label="Processed" value={eng ? fmtNumber(eng.requests_processed) : "—"} tone="emerald" />
          <StatCard icon={Cpu} label="Out tokens" value={eng ? fmtNumber(eng.total_completion_tokens) : "—"} tone="blue" />
          <StatCard icon={Clock} label="Uptime" value={health.data?.uptime_seconds === undefined ? "—" : fmtDuration(health.data.uptime_seconds)} tone="purple" />
        </div>

        {/* GPU memory breakdown */}
        {gpu && (
          <Card className="p-5">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium">
              <MemoryStick className="h-4 w-4 text-muted-foreground" /> GPU memory
            </div>
            <SegmentedBar
              total={gpu.total_uma_bytes}
              legend
              formatValue={(v) => fmtBytes(v)}
              segments={[
                { value: gpu.active_bytes, tone: "accent", label: "Active" },
                { value: gpu.cache_bytes, tone: "info", label: "Cache" },
                { value: gpuFree, tone: "neutral", label: "Free" },
              ]}
            />
            <div className="mt-3 text-xs text-muted-foreground">Peak {fmtBytes(gpu.peak_bytes)}</div>
          </Card>
        )}

        {/* System / hardware info (from /system) */}
        {sys && (
          <Section title="System" icon={<Cpu className="h-4 w-4 text-muted-foreground" />} defaultOpen>
            <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
              <Info label="Platform" value={sys.platform} />
              <Info label="Hostname" value={sys.hostname} />
              <Info label="CPU cores" value={`${sys.cpu.physical_cores}P / ${sys.cpu.logical_cores}L`} />
              <Info label="MLX" value={sys.gpu.mlx_version} />
              <Info label="Python" value={sys.python_version} />
              <Info label="PID" value={sys.pid} />
              {sys.compute_utilization_pct != null && (
                <Info label="Compute util" value={fmtPct(sys.compute_utilization_pct)} />
              )}
            </div>
          </Section>
        )}

        {/* Aggregated engine subsystems (from /all) — recursive generic render
            so every subsystem field is surfaced, nothing collapsed to "{…}". */}
        {all.error && <Alert variant="warning">Aggregated metrics unavailable.</Alert>}
        {all.data &&
          (() => {
            const topScalars = Object.fromEntries(Object.entries(all.data).filter(([, v]) => !isRecord(v)));
            const subsystems = Object.entries(all.data).filter(([, v]) => isRecord(v)) as [
              string,
              Record<string, unknown>,
            ][];
            return (
              <>
                {Object.keys(topScalars).length > 0 && (
                  <Card className="p-5">
                    <MetricGroup data={topScalars} />
                  </Card>
                )}
                {subsystems.map(([key, v]) => (
                  <Section key={key} title={key.replace(/_/g, " ")} icon={<Activity className="h-4 w-4 text-muted-foreground" />}>
                    <MetricGroup data={v} />
                  </Section>
                ))}
              </>
            );
          })()}
    </PageShell>
  );
}
