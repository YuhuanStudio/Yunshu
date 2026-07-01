"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Button,
  Card,
  Badge,
  Alert,
  Spinner,
  SegmentedSelect,
  Input,
  Textarea,
  NumberInput,
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
import {
  Gauge,
  Timer,
  Zap,
  Cpu,
  Layers,
  Grid3x3,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, ApiError, usePolling } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import type { Model } from "@/lib/types";

/* ------------------------------------------------------------------ *
 * Page-local types
 * ------------------------------------------------------------------ */

type BenchName =
  | "roofline"
  | "latency"
  | "throughput"
  | "model"
  | "batch"
  | "roofline-model"
  | "bfcl-eval";

interface BenchStatus {
  active: string | null;
  results_available: string[];
}

interface BenchMeta {
  name: BenchName;
  title: string;
  description: string;
  icon: LucideIcon;
}

const BENCHMARKS: BenchMeta[] = [
  {
    name: "roofline",
    title: "Roofline",
    description: "Sweep matrix sizes to chart raw compute (GFLOP/s) against memory bandwidth.",
    icon: Gauge,
  },
  {
    name: "latency",
    title: "Latency",
    description: "Measure per-request response times (p50/p95/p99) against the running server.",
    icon: Timer,
  },
  {
    name: "throughput",
    title: "Throughput",
    description: "Ramp concurrency and record sustained requests and tokens per second.",
    icon: Zap,
  },
  {
    name: "model",
    title: "Model (in-process)",
    description: "Latency sweep against the in-process engine — needs a model already loaded.",
    icon: Cpu,
  },
  {
    name: "batch",
    title: "Batch",
    description: "Single concurrency batch run; returns an ad-hoc metrics dictionary.",
    icon: Layers,
  },
  {
    name: "roofline-model",
    title: "GEMM Roofline",
    description: "Analytical roofline: per-GEMM arithmetic intensity and predicted throughput for a chip.",
    icon: Grid3x3,
  },
  {
    name: "bfcl-eval",
    title: "BFCL Eval",
    description: "Berkeley Function-Calling Leaderboard — tool-calling accuracy by category.",
    icon: Wrench,
  },
];

const BASE = "/api/v1/bench";

/* ------------------------------------------------------------------ *
 * Small input helpers
 * ------------------------------------------------------------------ */

function parseNums(raw: string): number[] {
  return raw
    .split(/[\s,]+/)
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

function parseStrs(raw: string): string[] {
  return raw
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** Parse newline-separated `M,N,K` rows into number[][], keeping only complete triples. */
function parseGemmSizes(raw: string): number[][] {
  return raw
    .split(/\n+/)
    .map((line) => parseNums(line))
    .filter((row) => row.length === 3);
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <span className="block text-sm font-medium">{label}</span>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Generic result renderers
 * ------------------------------------------------------------------ */

function fmtCell(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "number") return fmtNumber(value);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) return value.length ? value.map(fmtCell).join(", ") : "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function ResultTable({ rows }: { rows: Array<Record<string, unknown>> }) {
  if (!rows.length) {
    return <p className="text-sm text-muted-foreground">No rows returned.</p>;
  }
  const columns = Array.from(new Set(rows.flatMap((r) => Object.keys(r))));
  return (
    <Table containerClassName="rounded-md border border-border">
      <TableHeader>
        <TableRow>
          {columns.map((c) => (
            <TableHead key={c}>{c}</TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row, i) => (
          <TableRow key={i}>
            {columns.map((c) => (
              <TableCell
                key={c}
                label={c}
                className={cn(typeof row[c] === "number" && "tabular-nums")}
              >
                {fmtCell(row[c])}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

/** Render a flat dict (skipping array/object values) as a label/value grid. */
function KVGrid({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data).filter(
    ([, v]) => v == null || typeof v !== "object" || Array.isArray(v),
  );
  if (!entries.length) return <p className="text-sm text-muted-foreground">No fields returned.</p>;
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {entries.map(([k, v]) => (
        <StatCard key={k} label={k} value={fmtCell(v)} compact />
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

export default function BenchmarksPage() {
  const [active, setActive] = useState<BenchName>("roofline");

  // Per-bench results / errors persist across panel switches.
  const [results, setResults] = useState<Partial<Record<BenchName, unknown>>>({});
  const [errors, setErrors] = useState<Partial<Record<BenchName, string>>>({});
  const [runningLocal, setRunningLocal] = useState<BenchName | null>(null);

  // Models (for latency / throughput / model benches).
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .get<{ data: Model[] }>("/v1/models")
      .then((r) => {
        if (cancelled) return;
        setModels(r.data);
        setModel((m) => m || r.data[0]?.id || "");
      })
      .catch(() => {
        /* models optional; latency/throughput will surface their own errors */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Poll the shared bench status (only ONE bench may run at a time server-side).
  const { data: status, refresh } = usePolling<BenchStatus>(
    (signal) => api.get<BenchStatus>(`${BASE}/status`, signal),
    2000,
  );

  const serverBusy = status?.active != null;
  const anyBusy = serverBusy || runningLocal != null;

  const meta = useMemo(() => BENCHMARKS.find((b) => b.name === active)!, [active]);

  const run = async (name: BenchName, path: string, body?: unknown) => {
    setRunningLocal(name);
    setErrors((s) => ({ ...s, [name]: undefined }));
    try {
      const data = await api.post<unknown>(`${BASE}${path}`, body);
      setResults((s) => ({ ...s, [name]: data }));
      toast.success(`${name} benchmark complete`);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Benchmark failed";
      setErrors((s) => ({ ...s, [name]: msg }));
      toast.error(`${name} benchmark failed`, msg);
    } finally {
      setRunningLocal(null);
      refresh();
    }
  };

  const done = new Set(status?.results_available ?? []);

  return (
    <PageShell
      title="Benchmarks"
      description="Run performance benchmarks against the server and inspect the results. Only one benchmark can run at a time."
      width="wide"
    >
      {/* Shared status banner */}
      <Card className="mb-6 flex flex-wrap items-center gap-3 p-4">
        <span className="text-sm font-medium">Status</span>
        {serverBusy ? (
          <Badge variant="warning" className="gap-1.5">
            <Spinner className="h-3 w-3" /> running: {status?.active}
          </Badge>
        ) : (
          <Badge variant="success">idle</Badge>
        )}
        <div className="ml-auto flex flex-wrap items-center gap-1.5">
          {status?.results_available?.length ? (
            <>
              <span className="text-xs text-muted-foreground">results:</span>
              {status.results_available.map((r) => (
                <Badge key={r} variant="info">
                  {r}
                </Badge>
              ))}
            </>
          ) : (
            <span className="text-xs text-muted-foreground">no cached results</span>
          )}
        </div>
      </Card>

      {/* Bench selector */}
      <SegmentedSelect
        className="mb-6 flex-wrap"
        value={active}
        onChange={(v) => setActive(v as BenchName)}
        options={BENCHMARKS.map((b) => ({ value: b.name, label: b.title, icon: b.icon }))}
      />

      <Card className="space-y-5 p-6">
        <div className="flex items-start gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent">
            <meta.icon className="h-5 w-5" />
          </span>
          <div className="min-w-0">
            <h2 className="text-base font-semibold">{meta.title}</h2>
            <p className="mt-0.5 text-sm text-muted-foreground">{meta.description}</p>
          </div>
          {done.has(active) && (
            <Badge variant="success" className="ml-auto">
              cached
            </Badge>
          )}
        </div>

        {serverBusy && (
          <Alert variant="info" title="A benchmark is running">
            <span className="capitalize">{status?.active}</span> is in progress. Run buttons are
            disabled until it finishes.
          </Alert>
        )}

        <BenchPanel
          name={active}
          busy={anyBusy}
          running={runningLocal === active}
          hasRun={done.has(active) || results[active] !== undefined}
          models={models}
          model={model}
          setModel={setModel}
          run={run}
        />

        {errors[active] && (
          <Alert variant="error" title="Benchmark failed">
            {errors[active]}
          </Alert>
        )}

        {results[active] !== undefined && (
          <BenchResult name={active} result={results[active]} />
        )}
      </Card>
    </PageShell>
  );
}

/* ------------------------------------------------------------------ *
 * Per-bench input panels
 * ------------------------------------------------------------------ */

function RunButton({
  busy,
  running,
  onClick,
  hasRun,
}: {
  busy: boolean;
  running: boolean;
  onClick: () => void;
  hasRun: boolean;
}) {
  return (
    <Button variant={hasRun ? "secondary" : "primary"} onClick={onClick} disabled={busy}>
      {running ? (
        <>
          <Spinner className="h-4 w-4" /> Running…
        </>
      ) : hasRun ? (
        "Run again"
      ) : (
        "Run benchmark"
      )}
    </Button>
  );
}

function BenchPanel({
  name,
  busy,
  running,
  hasRun,
  models,
  model,
  setModel,
  run,
}: {
  name: BenchName;
  busy: boolean;
  running: boolean;
  hasRun: boolean;
  models: Model[];
  model: string;
  setModel: (id: string) => void;
  run: (name: BenchName, path: string, body?: unknown) => void;
}) {
  switch (name) {
    case "roofline":
      return <RooflinePanel busy={busy} running={running} hasRun={hasRun} run={run} />;
    case "latency":
      return (
        <LatencyPanel
          busy={busy}
          running={running}
          hasRun={hasRun}
          models={models}
          model={model}
          setModel={setModel}
          run={run}
        />
      );
    case "throughput":
      return (
        <ThroughputPanel
          busy={busy}
          running={running}
          hasRun={hasRun}
          models={models}
          model={model}
          setModel={setModel}
          run={run}
        />
      );
    case "model":
      return (
        <ModelBenchPanel
          busy={busy}
          running={running}
          hasRun={hasRun}
          models={models}
          model={model}
          setModel={setModel}
          run={run}
        />
      );
    case "batch":
      return <BatchPanel busy={busy} running={running} hasRun={hasRun} run={run} />;
    case "roofline-model":
      return <RooflineModelPanel busy={busy} running={running} hasRun={hasRun} run={run} />;
    case "bfcl-eval":
      return <BfclPanel busy={busy} running={running} hasRun={hasRun} run={run} />;
  }
}

type RunFn = (name: BenchName, path: string, body?: unknown) => void;
interface PanelBase {
  busy: boolean;
  running: boolean;
  hasRun: boolean;
  run: RunFn;
}
interface ModelPanelBase extends PanelBase {
  models: Model[];
  model: string;
  setModel: (id: string) => void;
}

const DTYPES = [
  { value: "float16", label: "float16" },
  { value: "bfloat16", label: "bfloat16" },
  { value: "float32", label: "float32" },
];

function RooflinePanel({ busy, running, hasRun, run }: PanelBase) {
  const [sizes, setSizes] = useState("512, 1024, 2048, 4096");
  const [numWarmup, setNumWarmup] = useState(5);
  const [numIters, setNumIters] = useState(20);
  const [dtype, setDtype] = useState("float16");
  return (
    <div className="space-y-4">
      <Field label="Matrix sizes" hint="Comma-separated square GEMM dimensions.">
        <Input value={sizes} onChange={(e) => setSizes(e.target.value)} className="font-mono" />
      </Field>
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Warmup iters">
          <NumberInput value={numWarmup} onChange={setNumWarmup} min={0} max={100} />
        </Field>
        <Field label="Measured iters">
          <NumberInput value={numIters} onChange={setNumIters} min={1} max={1000} />
        </Field>
      </div>
      <Field label="Dtype">
        <SegmentedSelect options={DTYPES} value={dtype} onChange={setDtype} />
      </Field>
      <RunButton
        busy={busy}
        running={running}
        hasRun={hasRun}
        onClick={() =>
          run("roofline", "/roofline", {
            sizes: parseNums(sizes),
            num_warmup: numWarmup,
            num_iters: numIters,
            dtype,
          })
        }
      />
    </div>
  );
}

function ModelRow({
  models,
  model,
  setModel,
}: {
  models: Model[];
  model: string;
  setModel: (id: string) => void;
}) {
  return (
    <Field label="Model">
      {models.length ? (
        <ModelPicker models={models} value={model} onChange={setModel} />
      ) : (
        <Input value={model} onChange={(e) => setModel(e.target.value)} placeholder="model id" />
      )}
    </Field>
  );
}

function LatencyPanel({ busy, running, hasRun, models, model, setModel, run }: ModelPanelBase) {
  const [promptLengths, setPromptLengths] = useState("128, 512");
  const [maxTokensList, setMaxTokensList] = useState("64, 256");
  const [numRequests, setNumRequests] = useState(20);
  return (
    <div className="space-y-4">
      <ModelRow models={models} model={model} setModel={setModel} />
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Prompt lengths" hint="Comma-separated token counts.">
          <Input
            value={promptLengths}
            onChange={(e) => setPromptLengths(e.target.value)}
            className="font-mono"
          />
        </Field>
        <Field label="Max tokens list" hint="Comma-separated output caps.">
          <Input
            value={maxTokensList}
            onChange={(e) => setMaxTokensList(e.target.value)}
            className="font-mono"
          />
        </Field>
      </div>
      <Field label="Requests per point">
        <NumberInput value={numRequests} onChange={setNumRequests} min={1} max={1000} />
      </Field>
      <p className="text-xs text-muted-foreground">Targets the local server at localhost:8000.</p>
      <RunButton
        busy={busy}
        running={running}
        hasRun={hasRun}
        onClick={() =>
          run("latency", "/latency", {
            base_url: "http://localhost:8000",
            model,
            prompt_lengths: parseNums(promptLengths),
            max_tokens_list: parseNums(maxTokensList),
            num_requests: numRequests,
          })
        }
      />
    </div>
  );
}

function ThroughputPanel({ busy, running, hasRun, models, model, setModel, run }: ModelPanelBase) {
  const [concurrency, setConcurrency] = useState("1, 4, 8, 16");
  const [promptTokens, setPromptTokens] = useState(256);
  const [maxTokens, setMaxTokens] = useState(128);
  const [numRequests, setNumRequests] = useState(50);
  return (
    <div className="space-y-4">
      <ModelRow models={models} model={model} setModel={setModel} />
      <Field label="Concurrency levels" hint="Comma-separated concurrent client counts.">
        <Input
          value={concurrency}
          onChange={(e) => setConcurrency(e.target.value)}
          className="font-mono"
        />
      </Field>
      <div className="grid gap-4 sm:grid-cols-3">
        <Field label="Prompt tokens">
          <NumberInput value={promptTokens} onChange={setPromptTokens} min={1} max={32768} />
        </Field>
        <Field label="Max tokens">
          <NumberInput value={maxTokens} onChange={setMaxTokens} min={1} max={8192} />
        </Field>
        <Field label="Requests / level">
          <NumberInput value={numRequests} onChange={setNumRequests} min={1} max={5000} />
        </Field>
      </div>
      <p className="text-xs text-muted-foreground">Targets the local server at localhost:8000.</p>
      <RunButton
        busy={busy}
        running={running}
        hasRun={hasRun}
        onClick={() =>
          run("throughput", "/throughput", {
            base_url: "http://localhost:8000",
            model,
            concurrency_levels: parseNums(concurrency),
            prompt_tokens: promptTokens,
            max_tokens: maxTokens,
            num_requests: numRequests,
          })
        }
      />
    </div>
  );
}

function ModelBenchPanel({ busy, running, hasRun, models, model, setModel, run }: ModelPanelBase) {
  const [promptLengths, setPromptLengths] = useState("128, 512");
  const [maxTokensList, setMaxTokensList] = useState("64, 256");
  const [numRequests, setNumRequests] = useState(20);
  const [stream, setStream] = useState("false");
  return (
    <div className="space-y-4">
      <Field label="Model" hint="Leave blank to use the currently loaded model.">
        {models.length ? (
          <ModelPicker models={models} value={model} onChange={setModel} />
        ) : (
          <Input value={model} onChange={(e) => setModel(e.target.value)} placeholder="model id" />
        )}
      </Field>
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Prompt lengths">
          <Input
            value={promptLengths}
            onChange={(e) => setPromptLengths(e.target.value)}
            className="font-mono"
          />
        </Field>
        <Field label="Max tokens list">
          <Input
            value={maxTokensList}
            onChange={(e) => setMaxTokensList(e.target.value)}
            className="font-mono"
          />
        </Field>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Requests per point">
          <NumberInput value={numRequests} onChange={setNumRequests} min={1} max={1000} />
        </Field>
        <Field label="Stream">
          <SegmentedSelect
            options={[
              { value: "false", label: "off" },
              { value: "true", label: "on" },
            ]}
            value={stream}
            onChange={setStream}
          />
        </Field>
      </div>
      <p className="text-xs text-muted-foreground">
        Runs against the in-process engine — a model must be loaded (503 otherwise).
      </p>
      <RunButton
        busy={busy}
        running={running}
        hasRun={hasRun}
        onClick={() =>
          run("model", "/model", {
            model: model || undefined,
            prompt_lengths: parseNums(promptLengths),
            max_tokens_list: parseNums(maxTokensList),
            num_requests: numRequests,
            stream: stream === "true",
          })
        }
      />
    </div>
  );
}

function BatchPanel({ busy, running, hasRun, run }: PanelBase) {
  const [concurrency, setConcurrency] = useState(8);
  const [numRequests, setNumRequests] = useState(50);
  const [promptTokens, setPromptTokens] = useState(256);
  const [maxTokens, setMaxTokens] = useState(128);
  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Concurrency">
          <NumberInput value={concurrency} onChange={setConcurrency} min={1} max={1024} />
        </Field>
        <Field label="Num requests">
          <NumberInput value={numRequests} onChange={setNumRequests} min={1} max={10000} />
        </Field>
        <Field label="Prompt tokens">
          <NumberInput value={promptTokens} onChange={setPromptTokens} min={1} max={32768} />
        </Field>
        <Field label="Max tokens">
          <NumberInput value={maxTokens} onChange={setMaxTokens} min={1} max={8192} />
        </Field>
      </div>
      <RunButton
        busy={busy}
        running={running}
        hasRun={hasRun}
        onClick={() => {
          const qs = new URLSearchParams({
            concurrency: String(concurrency),
            num_requests: String(numRequests),
            prompt_tokens: String(promptTokens),
            max_tokens: String(maxTokens),
          });
          run("batch", `/batch?${qs.toString()}`);
        }}
      />
    </div>
  );
}

function RooflineModelPanel({ busy, running, hasRun, run }: PanelBase) {
  const [chip, setChip] = useState("H100");
  const [gemms, setGemms] = useState("4096, 4096, 4096\n8192, 8192, 8192\n2048, 8192, 2048");
  return (
    <div className="space-y-4">
      <Field label="Chip" hint="Named accelerator profile (e.g. H100, A100, MI300X).">
        <Input value={chip} onChange={(e) => setChip(e.target.value)} />
      </Field>
      <Field label="GEMM sizes" hint="One `M, N, K` triple per line.">
        <Textarea
          value={gemms}
          onChange={(e) => setGemms(e.target.value)}
          rows={4}
          className="font-mono text-xs"
        />
      </Field>
      <RunButton
        busy={busy}
        running={running}
        hasRun={hasRun}
        onClick={() =>
          run("roofline-model", "/roofline-model", {
            chip,
            gemm_sizes: parseGemmSizes(gemms),
          })
        }
      />
    </div>
  );
}

function BfclPanel({ busy, running, hasRun, run }: PanelBase) {
  const [categories, setCategories] = useState("simple, parallel, multiple");
  const [maxSamples, setMaxSamples] = useState(50);
  return (
    <div className="space-y-4">
      <Field label="Categories" hint="Comma-separated BFCL category names.">
        <Input
          value={categories}
          onChange={(e) => setCategories(e.target.value)}
          className="font-mono"
        />
      </Field>
      <Field label="Max samples / category">
        <NumberInput value={maxSamples} onChange={setMaxSamples} min={1} max={2000} />
      </Field>
      <RunButton
        busy={busy}
        running={running}
        hasRun={hasRun}
        onClick={() =>
          run("bfcl-eval", "/bfcl-eval", {
            categories: parseStrs(categories),
            max_samples: maxSamples,
          })
        }
      />
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Per-bench result renderers
 * ------------------------------------------------------------------ */

function BenchResult({ name, result }: { name: BenchName; result: unknown }): ReactNode {
  const r = result as Record<string, unknown>;

  if (name === "roofline") {
    const sizes = (r.sizes as number[]) ?? [];
    const gflops = (r.gflops as number[]) ?? [];
    const bw = (r.bandwidth_gb_s as number[]) ?? [];
    const rows = sizes.map((size, i) => ({
      size,
      gflops: gflops[i],
      "bandwidth_gb_s": bw[i],
    }));
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="Peak GFLOP/s" value={fmtCell(r.peak_gflops)} tone="blue" compact />
          <StatCard label="Peak GB/s" value={fmtCell(r.peak_bandwidth_gb_s)} tone="emerald" compact />
          <StatCard label="Device" value={fmtCell(r.device)} compact />
          <StatCard label="Dtype" value={fmtCell(r.dtype)} compact />
        </div>
        <ResultTable rows={rows} />
      </div>
    );
  }

  if (name === "roofline-model") {
    const analyses = (r.analyses as Array<Record<string, unknown>>) ?? [];
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <StatCard label="Chip" value={fmtCell(r.chip)} compact />
          <StatCard label="Bandwidth GB/s" value={fmtCell(r.bandwidth_gbps)} tone="emerald" compact />
          <StatCard label="Compute TFLOP/s (fp16)" value={fmtCell(r.compute_tflops_fp16)} tone="blue" compact />
        </div>
        <ResultTable rows={analyses} />
      </div>
    );
  }

  if (name === "bfcl-eval") {
    const categories = (r.categories as Array<Record<string, unknown>>) ?? [];
    return (
      <div className="space-y-4">
        {r.model != null && <Badge variant="info">model: {String(r.model)}</Badge>}
        <ResultTable rows={categories} />
      </div>
    );
  }

  if (name === "batch") {
    return <KVGrid data={r} />;
  }

  // latency / throughput / model → { results: [...] } (+ scalar extras)
  const rows = (r.results as Array<Record<string, unknown>>) ?? [];
  const peak = r.peak_throughput;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        {r.model != null && <Badge variant="info">model: {String(r.model)}</Badge>}
        {peak != null && <Badge variant="success">peak throughput: {fmtCell(peak)}</Badge>}
      </div>
      <ResultTable rows={rows} />
    </div>
  );
}
