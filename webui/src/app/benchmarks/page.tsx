"use client";

import { useState } from "react";
import {
  Gauge,
  Timer,
  BarChart3,
  Play,
  Loader2,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

const benchmarks = [
  {
    id: "roofline",
    title: "Roofline",
    description: "GEMM throughput vs matrix size on Apple GPU",
    icon: Gauge,
    color: "text-blue-400",
    bg: "bg-blue-500/15",
  },
  {
    id: "latency",
    title: "Latency",
    description: "End-to-end request latency P50/P95/P99",
    icon: Timer,
    color: "text-amber-400",
    bg: "bg-amber-500/15",
  },
  {
    id: "throughput",
    title: "Throughput",
    description: "Concurrent request tokens/s at varying load",
    icon: BarChart3,
    color: "text-emerald-400",
    bg: "bg-emerald-500/15",
  },
];

interface BenchResult {
  [key: string]: string | number | boolean | null;
}

export default function BenchmarksPage() {
  const [running, setRunning] = useState<string | null>(null);
  const [results, setResults] = useState<BenchResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [benchType, setBenchType] = useState<string | null>(null);

  const runBenchmark = async (type: string) => {
    setRunning(type);
    setBenchType(type);
    setResults(null);
    setError(null);

    try {
      const res = await fetch(`/api/v1/bench/${type}`, { method: "POST" });
      if (!res.ok) {
        if (res.status === 404) {
          setError("Benchmark endpoint not available — start the server first.");
        } else {
          setError(`Benchmark failed: ${res.status} ${await res.text()}`);
        }
        return;
      }
      const data = await res.json();

      // Normalize results to array of flat key-value pairs
      if (Array.isArray(data.results)) {
        setResults(data.results);
      } else if (data.sizes && data.gflops) {
        // Roofline format
        setResults(
          data.sizes.map((s: number, i: number) => ({
            MatrixSize: `${s}x${s}`,
            GFLOPS: typeof data.gflops[i] === "number" ? data.gflops[i].toFixed(1) : data.gflops[i],
            Bandwidth_GB: data.bandwidth_gb_s?.[i] != null ? data.bandwidth_gb_s[i].toFixed(1) : "—",
          }))
        );
      } else if (data.results && Array.isArray(data.results)) {
        setResults(data.results);
      } else {
        // Flat object → single row
        setResults([data]);
      }
    } catch (err) {
      setError(`Connection error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRunning(null);
    }
  };

  return (
    <div className="p-6 space-y-6 page-enter">
      <h2 className="text-2xl font-bold">Benchmarks</h2>

      {/* Benchmark Cards */}
      <div className="grid grid-cols-3 gap-4">
        {benchmarks.map((b) => {
          const Icon = b.icon;
          const isRunning = running === b.id;
          const isDone = benchType === b.id && results && !running;

          return (
            <div
              key={b.id}
              className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4"
            >
              <div className="flex items-center gap-3 mb-2">
                <div className={`w-10 h-10 rounded-lg ${b.bg} flex items-center justify-center`}>
                  <Icon className={`w-5 h-5 ${b.color}`} />
                </div>
                <div>
                  <h3 className="font-semibold text-sm">{b.title}</h3>
                  <p className="text-xs text-[var(--color-text-secondary)]">{b.description}</p>
                </div>
              </div>
              <button
                onClick={() => runBenchmark(b.id)}
                disabled={running !== null}
                className={`mt-3 flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  isRunning
                    ? "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                    : "bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white disabled:opacity-50"
                }`}
              >
                {isRunning ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Running...</>
                ) : isDone ? (
                  <><CheckCircle2 className="w-4 h-4 text-[var(--color-success)]" /> Run Again</>
                ) : (
                  <><Play className="w-4 h-4" /> Run</>
                )}
              </button>
            </div>
          );
        })}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-[var(--color-danger)]/10 border border-[var(--color-danger)]/30 rounded-xl px-4 py-3 text-sm text-[var(--color-danger)] flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Results */}
      {results && results.length > 0 && (
        <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)]">
          <div className="px-4 py-3 border-b border-[var(--color-border)]">
            <h3 className="font-semibold text-sm">
              {benchmarks.find((b) => b.id === benchType)?.title ?? "Benchmark"} Results
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  {Object.keys(results[0]).map((key) => (
                    <th
                      key={key}
                      className="px-4 py-2 text-left text-xs text-[var(--color-text-secondary)] font-medium uppercase tracking-wide"
                    >
                      {key.replace(/_/g, " ")}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((row, i) => (
                  <tr key={i} className="border-b border-[var(--color-border)] last:border-0">
                    {Object.values(row).map((val, j) => (
                      <td key={j} className="px-4 py-2 tabular-nums">
                        {typeof val === "number" ? val.toLocaleString() : String(val ?? "—")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
