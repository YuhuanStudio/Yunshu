"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Button,
  Textarea,
  Input,
  Card,
  Badge,
  Alert,
  EmptyState,
  Spinner,
  NumberInput,
  SegmentedSelect,
  cn,
  toast,
} from "yunui";
import { StatCard } from "yunui/patterns";
import { Copy, Hash, Sparkles, Boxes, Ruler } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, ApiError } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import type { Model } from "@/lib/types";

// ---- page-local API contract ---------------------------------------------
type PoolingType = "AUTO" | "MEAN" | "CLS" | "LAST";

interface EmbeddingsRequest {
  model: string;
  input: string[];
  encoding_format: "float";
  dimensions?: number;
  pooling_type?: "MEAN" | "CLS" | "LAST";
  instruction?: string;
}

interface EmbeddingRow {
  index: number;
  embedding: number[];
}

interface EmbeddingsResponse {
  data: EmbeddingRow[];
  model?: string;
  usage: { prompt_tokens?: number; total_tokens: number };
}

const MAX_INPUTS = 2048;
const MAX_CHARS = 8192;
const MAX_DIMS = 8192;
const PREVIEW_CHIPS = 24;
const STRIP_DIMS = 48;

const POOLING_OPTIONS = [
  { value: "AUTO" as const, label: "Auto" },
  { value: "MEAN" as const, label: "Mean" },
  { value: "CLS" as const, label: "CLS" },
  { value: "LAST" as const, label: "Last" },
];

/** Split a textarea into one non-empty text per line. */
function splitInputs(raw: string): string[] {
  return raw
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
}

function l2norm(v: number[]): number {
  let sum = 0;
  for (const x of v) sum += x * x;
  return Math.sqrt(sum);
}

/** Cosine similarity of two vectors (robust even if not unit-normalized). */
function cosine(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

export default function EmbeddingsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [text, setText] = useState("");
  const [dimensions, setDimensions] = useState<number | undefined>(undefined);
  const [pooling, setPooling] = useState<PoolingType>("AUTO");
  const [instruction, setInstruction] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EmbeddingsResponse | null>(null);
  const [sentInputs, setSentInputs] = useState<string[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ data: Model[] }>("/v1/models", controller.signal)
      .then((res) => {
        const list = res.data ?? [];
        setModels(list);
        setModel((cur) => cur || list[0]?.id || "");
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError((e as Error).message);
      });
    return () => controller.abort();
  }, []);

  const inputs = useMemo(() => splitInputs(text), [text]);
  const tooManyInputs = inputs.length > MAX_INPUTS;
  const tooLong = inputs.find((l) => l.length > MAX_CHARS);
  const canRun = !!model && inputs.length > 0 && !tooManyInputs && !tooLong && !loading;

  const generate = async () => {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const body: EmbeddingsRequest = {
      model,
      input: inputs,
      encoding_format: "float",
    };
    if (dimensions && dimensions > 0) body.dimensions = dimensions;
    if (pooling !== "AUTO") body.pooling_type = pooling;
    if (instruction.trim()) body.instruction = instruction.trim();
    try {
      const res = await api.post<EmbeddingsResponse>("/v1/embeddings", body);
      setResult(res);
      setSentInputs(inputs);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const rows = result?.data ?? [];

  return (
    <PageShell
      title="Embeddings"
      description="Turn text into vectors, preview each embedding, and compare them side by side."
      width="narrow"
    >
      <div className="space-y-6">
        <Card className="space-y-4 p-5">
          <div className="space-y-2">
            <label className="text-sm font-medium">Model</label>
            <ModelPicker models={models} value={model} onChange={setModel} />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <label className="text-sm font-medium">Input text</label>
              <span className="text-xs text-muted-foreground">
                one text per line · {fmtNumber(inputs.length)} / {fmtNumber(MAX_INPUTS)}
              </span>
            </div>
            <Textarea
              rows={6}
              placeholder={"The quick brown fox…\nA second line becomes a second vector…"}
              value={text}
              onChange={(e) => setText(e.target.value)}
              error={
                tooManyInputs
                  ? `Too many inputs (max ${fmtNumber(MAX_INPUTS)})`
                  : tooLong
                    ? `A line exceeds ${fmtNumber(MAX_CHARS)} characters`
                    : undefined
              }
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">Dimensions</label>
              <NumberInput aria-label="Dimensions"
                min={1}
                max={MAX_DIMS}
                step={64}
                value={dimensions ?? undefined}
                placeholder="model default (Matryoshka)"
                onChange={(v) => setDimensions(Number.isFinite(v) && v > 0 ? v : undefined)}
              />
              <p className="text-xs text-muted-foreground">Optional · truncate to ≤ {fmtNumber(MAX_DIMS)}</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Pooling</label>
              <SegmentedSelect<PoolingType>
                options={POOLING_OPTIONS}
                value={pooling}
                onChange={setPooling}
              />
              <p className="text-xs text-muted-foreground">How token states collapse into one vector</p>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Instruction</label>
            <Input aria-label="Instruction"
              placeholder="Optional task instruction (e.g. Represent this sentence for retrieval)"
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
            />
          </div>

          <div className="flex justify-end">
            <Button onClick={generate} disabled={!canRun}>
              {loading ? <Spinner size="sm" /> : <Sparkles className="h-4 w-4" />}
              Generate
            </Button>
          </div>
        </Card>

        {error && (
          <Alert variant="error" title="Request failed">
            {error}
          </Alert>
        )}

        {loading && !result && (
          <div className="flex justify-center py-12">
            <Spinner />
          </div>
        )}

        {!loading && !error && !result && (
          <EmptyState
            icon={<Boxes className="h-8 w-8" />}
            title="No embeddings yet"
            description="Pick a model, enter one or more lines of text, and generate vectors to preview and compare."
          />
        )}

        {result && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <StatCard
                icon={Hash}
                label="Total tokens"
                value={fmtNumber(result.usage.total_tokens)}
                subtext={
                  result.usage.prompt_tokens
                    ? `${fmtNumber(result.usage.prompt_tokens)} prompt`
                    : undefined
                }
                tone="blue"
              />
              <StatCard
                icon={Boxes}
                label="Vectors returned"
                value={fmtNumber(rows.length)}
                subtext={rows[0] ? `${fmtNumber(rows[0].embedding.length)} dims each` : undefined}
                tone="purple"
              />
            </div>

            {rows.map((row) => (
              <EmbeddingCard
                key={row.index}
                index={row.index}
                embedding={row.embedding}
                label={sentInputs[row.index]}
              />
            ))}

            {rows.length >= 2 && <SimilarityMatrix rows={rows} labels={sentInputs} />}
          </div>
        )}
      </div>
    </PageShell>
  );
}

function EmbeddingCard({
  index,
  embedding,
  label,
}: {
  index: number;
  embedding: number[];
  label?: string;
}) {
  const norm = l2norm(embedding);

  const copyVector = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(embedding));
      toast.success("Copied", `Vector #${index} copied to clipboard`);
    } catch {
      toast.error("Copy failed", "Could not access the clipboard");
    }
  };

  return (
    <Card className="space-y-4 p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <span className="text-sm font-medium">Vector #{index}</span>
          {label && (
            <p className="truncate text-xs text-muted-foreground" title={label}>
              {label}
            </p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Badge variant="info">{fmtNumber(embedding.length)} dims</Badge>
          <Badge variant="default">‖v‖ {norm.toFixed(4)}</Badge>
          <Button variant="ghost" size="sm" onClick={copyVector}>
            <Copy className="h-3.5 w-3.5" /> Copy
          </Button>
        </div>
      </div>

      <VectorStrip embedding={embedding} />

      <div className="flex flex-wrap gap-1.5">
        {embedding.slice(0, PREVIEW_CHIPS).map((v, i) => (
          <span
            key={i}
            className="inline-flex rounded bg-muted px-1.5 py-0.5 text-xs tabular-nums text-muted-foreground"
          >
            {v >= 0 ? "+" : ""}
            {v.toFixed(4)}
          </span>
        ))}
        {embedding.length > PREVIEW_CHIPS && (
          <span className="inline-flex items-center px-1.5 py-0.5 text-xs text-muted-foreground">
            +{fmtNumber(embedding.length - PREVIEW_CHIPS)} more
          </span>
        )}
      </div>
    </Card>
  );
}

/** A diverging bar strip of the first N dims: positive up, negative down. */
function VectorStrip({ embedding }: { embedding: number[] }) {
  const slice = embedding.slice(0, STRIP_DIMS);
  const max = Math.max(1e-6, ...slice.map((v) => Math.abs(v)));
  return (
    <div className="flex h-16 items-center gap-px rounded-md bg-muted/40 px-2">
      {slice.map((v, i) => {
        const h = (Math.abs(v) / max) * 48;
        const up = v >= 0;
        return (
          <div key={i} className="flex h-full flex-1 flex-col justify-center" title={v.toFixed(6)}>
            <div className="flex h-1/2 flex-col justify-end">
              {up && (
                <div
                  className="w-full rounded-sm"
                  style={{ height: `${h}%`, backgroundColor: "var(--color-primary)" }}
                />
              )}
            </div>
            <div className="flex h-1/2 flex-col justify-start">
              {!up && (
                <div
                  className="w-full rounded-sm"
                  style={{ height: `${h}%`, backgroundColor: "var(--color-muted-foreground)" }}
                />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Client-side pairwise cosine-similarity heatmap. */
function SimilarityMatrix({ rows, labels }: { rows: EmbeddingRow[]; labels: string[] }) {
  const n = rows.length;
  const matrix = useMemo(
    () => rows.map((a) => rows.map((b) => cosine(a.embedding, b.embedding))),
    [rows],
  );

  return (
    <Card className="space-y-3 p-5">
      <div className="flex items-center gap-2">
        <Ruler className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">Pairwise cosine similarity</span>
      </div>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-1 text-xs tabular-nums">
          <thead>
            <tr>
              <th className="p-1" />
              {rows.map((r) => (
                <th key={r.index} className="p-1 font-medium text-muted-foreground">
                  #{r.index}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((rowVals, i) => (
              <tr key={i}>
                <th
                  className="max-w-40 truncate p-1 text-right font-medium text-muted-foreground"
                  title={labels[rows[i].index]}
                >
                  #{rows[i].index}
                </th>
                {rowVals.map((s, j) => {
                  // map [-1,1] → [0,1] opacity intensity
                  const t = Math.max(0, Math.min(1, (s + 1) / 2));
                  const strong = t > 0.75;
                  return (
                    <td
                      key={j}
                      className={cn(
                        "rounded-md px-2 py-1 text-center",
                        i === j && "ring-1 ring-border",
                        strong ? "text-primary-foreground" : "text-foreground",
                      )}
                      style={{ backgroundColor: `color-mix(in oklab, var(--color-primary) ${Math.round(t * 100)}%, transparent)` }}
                      title={`cos(#${rows[i].index}, #${rows[j].index}) = ${s.toFixed(4)}`}
                    >
                      {s.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted-foreground">
        {fmtNumber(n)} vectors · 1.00 = identical direction, 0 = orthogonal, computed in-browser.
      </p>
    </Card>
  );
}
