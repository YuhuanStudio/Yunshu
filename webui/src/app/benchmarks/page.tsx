"use client";

import { useState, type ReactNode } from "react";
import {
  Button,
  Card,
  Badge,
  Alert,
  Spinner,
  Sparkline,
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
  cn,
  toast,
} from "yunui";
import { StatCard } from "yunui/patterns";
import { Gauge, Timer, Zap, type LucideIcon } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { api, ApiError } from "@/lib/api";
import { fmtNumber } from "@/lib/format";

type BenchType = "roofline" | "latency" | "throughput";

interface RooflineResult {
  sizes: number[];
  gflops: number[];
  bandwidth_gb_s?: number[];
}

interface TableResult {
  results: Array<Record<string, unknown>>;
}

type BenchResult = RooflineResult | TableResult;

interface BenchMeta {
  type: BenchType;
  title: string;
  description: string;
  icon: LucideIcon;
}

const BENCHMARKS: BenchMeta[] = [
  {
    type: "roofline",
    title: "Roofline",
    description: "Sweep problem sizes to chart compute throughput against memory bandwidth.",
    icon: Gauge,
  },
  {
    type: "latency",
    title: "Latency",
    description: "Measure per-request response times across the serving path.",
    icon: Timer,
  },
  {
    type: "throughput",
    title: "Throughput",
    description: "Push concurrent load and record sustained tokens per second.",
    icon: Zap,
  },
];

function isRoofline(r: BenchResult): r is RooflineResult {
  return Array.isArray((r as RooflineResult).gflops);
}

function fmtCell(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "number") return fmtNumber(value);
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

export default function BenchmarksPage() {
  const [results, setResults] = useState<Partial<Record<BenchType, BenchResult>>>({});
  const [loading, setLoading] = useState<Partial<Record<BenchType, boolean>>>({});
  const [errors, setErrors] = useState<Partial<Record<BenchType, string>>>({});

  const run = async (type: BenchType) => {
    setLoading((s) => ({ ...s, [type]: true }));
    setErrors((s) => ({ ...s, [type]: undefined }));
    try {
      const data = await api.post<BenchResult>(`/api/v1/bench/${type}`);
      setResults((s) => ({ ...s, [type]: data }));
      toast.success(`${type} benchmark complete`);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Benchmark failed";
      setErrors((s) => ({ ...s, [type]: msg }));
      toast.error(`${type} benchmark failed`, msg);
    } finally {
      setLoading((s) => ({ ...s, [type]: false }));
    }
  };

  return (
    <PageShell
      title="Benchmarks"
      description="Run performance benchmarks and inspect the results."
      width="wide"
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {BENCHMARKS.map(({ type, title, description, icon: Icon }) => {
          const busy = loading[type] ?? false;
          const result = results[type];
          const error = errors[type];
          const hasRun = result !== undefined || error !== undefined;

          return (
            <Card key={type} className="flex flex-col gap-4 p-5">
              <div className="flex items-start gap-3">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent">
                  <Icon className="h-5 w-5" />
                </span>
                <div className="min-w-0">
                  <h2 className="text-sm font-semibold">{title}</h2>
                  <p className="mt-0.5 text-sm text-muted-foreground">{description}</p>
                </div>
              </div>

              <Button
                variant={hasRun ? "secondary" : "primary"}
                size="sm"
                onClick={() => run(type)}
                disabled={busy}
                className="w-full"
              >
                {busy ? (
                  <>
                    <Spinner className="h-4 w-4" /> Running…
                  </>
                ) : hasRun ? (
                  "Run again"
                ) : (
                  "Run"
                )}
              </Button>

              {error && (
                <Alert variant="error" title="Benchmark failed">
                  {error}
                </Alert>
              )}

              {result && !error && <BenchOutput type={type} result={result} />}
            </Card>
          );
        })}
      </div>
    </PageShell>
  );
}

function BenchOutput({ type, result }: { type: BenchType; result: BenchResult }): ReactNode {
  if (type === "roofline" && isRoofline(result)) {
    const { sizes, gflops, bandwidth_gb_s } = result;
    const peakGflops = gflops.length ? Math.max(...gflops) : 0;
    const peakBw = bandwidth_gb_s?.length ? Math.max(...bandwidth_gb_s) : undefined;
    return (
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground">GFLOP/s</span>
          <Badge variant="info">{sizes.length} sizes</Badge>
        </div>
        <Sparkline area data={gflops} className="h-16 w-full" />
        <div className="grid grid-cols-2 gap-3">
          <StatCard label="Peak GFLOP/s" value={fmtNumber(peakGflops)} tone="blue" compact />
          <StatCard
            label="Peak GB/s"
            value={peakBw !== undefined ? fmtNumber(peakBw) : "—"}
            tone="emerald"
            compact
          />
        </div>
      </div>
    );
  }

  const rows = (result as TableResult).results ?? [];
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">No results returned.</p>;
  }
  const columns = Object.keys(rows[0]);

  return (
    <Table containerClassName="rounded-md border border-border">
      <TableHeader>
        <TableRow>
          {columns.map((col) => (
            <TableHead key={col}>{col}</TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row, i) => (
          <TableRow key={i}>
            {columns.map((col) => (
              <TableCell
                key={col}
                label={col}
                className={cn(typeof row[col] === "number" && "tabular-nums")}
              >
                {fmtCell(row[col])}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
