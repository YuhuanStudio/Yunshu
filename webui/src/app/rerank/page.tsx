"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
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
  CustomSelect,
  cn,
} from "yunui";
import {
  Play,
  ListOrdered,
  Scale,
  Tags,
  Layers,
  ArrowUpDown,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, ApiError } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import type { Model } from "@/lib/types";

// ---- page-local API contracts --------------------------------------------
type Tool = "rerank" | "score" | "classify" | "pooling";

interface RerankResult {
  index: number;
  relevance_score: number;
  document?: string | { text?: string };
}
interface RerankResponse {
  id?: string;
  results: RerankResult[];
}

interface ScoreRow {
  index: number;
  score: number;
}
interface ScoreResponse {
  data: ScoreRow[];
}

interface ClassifyRow {
  label: string;
  score: number;
  index: number;
}
interface ClassifyResponse {
  results: ClassifyRow[];
}

interface PoolingRow {
  data: number[];
}
interface PoolingResponse {
  data: PoolingRow[];
}

const TABS: { value: Tool; label: string; icon: LucideIcon }[] = [
  { value: "rerank", label: "Rerank", icon: ListOrdered },
  { value: "score", label: "Score", icon: Scale },
  { value: "classify", label: "Classify", icon: Tags },
  { value: "pooling", label: "Pooling", icon: Layers },
];

/** Split a textarea into one non-empty item per line. */
function lines(raw: string): string[] {
  return raw
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
}

function errMsg(e: unknown): string {
  return e instanceof ApiError ? e.message : (e as Error).message;
}

// ==========================================================================
export default function RerankPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [tab, setTab] = useState<Tool>("rerank");
  const [modelsError, setModelsError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ data: Model[] }>("/v1/models", controller.signal)
      .then((res) => setModels(res.data ?? []))
      .catch((e) => {
        if (!controller.signal.aborted) setModelsError((e as Error).message);
      });
    return () => controller.abort();
  }, []);

  return (
    <PageShell
      title="Reranking toolkit"
      description="Rerank, score, classify, and pool — the retrieval primitives that sit on top of embeddings."
      width="narrow"
    >
      <div className="space-y-6">
        <SegmentedSelect<Tool> options={TABS} value={tab} onChange={setTab} />

        {modelsError && (
          <Alert variant="warning" title="Could not load models">
            {modelsError}
          </Alert>
        )}

        {tab === "rerank" && <RerankTab models={models} />}
        {tab === "score" && <ScoreTab models={models} />}
        {tab === "classify" && <ClassifyTab models={models} />}
        {tab === "pooling" && <PoolingTab models={models} />}
      </div>
    </PageShell>
  );
}

// ---- shared bits ----------------------------------------------------------
function ModelField({
  models,
  value,
  onChange,
}: {
  models: Model[];
  value: string;
  onChange: (id: string) => void;
}) {
  useEffect(() => {
    if (!value && models[0]) onChange(models[0].id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models]);
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">Model</label>
      <ModelPicker models={models} value={value} onChange={onChange} />
    </div>
  );
}

function RunButton({
  loading,
  disabled,
  onClick,
}: {
  loading: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <div className="flex justify-end">
      <Button onClick={onClick} disabled={disabled || loading}>
        {loading ? <Spinner size="sm" /> : <Play className="h-4 w-4" />}
        Run
      </Button>
    </div>
  );
}

/** A horizontal score bar (0–1) with a numeric label. */
function ScoreBar({ value, label }: { value: number; label?: string }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="space-y-1">
      {label && <span className="text-xs text-muted-foreground">{label}</span>}
      <div className="flex items-center gap-2">
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full"
            style={{ width: `${pct}%`, backgroundColor: "var(--color-primary)" }}
          />
        </div>
        <span className="w-14 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
          {value.toFixed(4)}
        </span>
      </div>
    </div>
  );
}

// ==========================================================================
function RerankTab({ models }: { models: Model[] }) {
  const [model, setModel] = useState("");
  const [query, setQuery] = useState("");
  const [docsText, setDocsText] = useState("");
  const [topN, setTopN] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RerankResponse | null>(null);

  const docs = useMemo(() => lines(docsText), [docsText]);
  const canRun = !!model && !!query.trim() && docs.length > 0;

  const run = async () => {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.post<RerankResponse>("/v1/rerank", {
        model,
        query: query.trim(),
        documents: docs,
        return_documents: true,
        ...(topN && topN > 0 ? { top_n: topN } : {}),
      });
      setResult(res);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setLoading(false);
    }
  };

  const ranked = result ? [...result.results].sort((a, b) => b.relevance_score - a.relevance_score) : [];

  return (
    <div className="space-y-6">
      <Card className="space-y-4 p-5">
        <ModelField models={models} value={model} onChange={setModel} />
        <div className="space-y-2">
          <label className="text-sm font-medium">Query</label>
          <Input
            placeholder="What are you searching for?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <label className="text-sm font-medium">Documents</label>
            <span className="text-xs text-muted-foreground">
              one per line · {fmtNumber(docs.length)}
            </span>
          </div>
          <Textarea
            rows={6}
            placeholder={"Candidate passage one…\nCandidate passage two…"}
            value={docsText}
            onChange={(e) => setDocsText(e.target.value)}
          />
        </div>
        <div className="space-y-2 sm:max-w-[12rem]">
          <label className="text-sm font-medium">Top N</label>
          <NumberInput
            min={1}
            max={docs.length || undefined}
            value={topN ?? undefined}
            placeholder="all"
            onChange={(v) => setTopN(Number.isFinite(v) && v > 0 ? v : undefined)}
          />
        </div>
        <RunButton loading={loading} disabled={!canRun} onClick={run} />
      </Card>

      <Results error={error} loading={loading} empty={!result} icon={ListOrdered} emptyTitle="No ranking yet" emptyDesc="Enter a query and candidate documents to rank them by relevance.">
        {result && (
          <Card className="space-y-3 p-5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Ranked results</span>
              <Badge variant="info">{fmtNumber(ranked.length)} shown</Badge>
            </div>
            <ol className="space-y-3">
              {ranked.map((r, rank) => {
                const docText =
                  typeof r.document === "string" ? r.document : r.document?.text;
                return (
                  <li key={r.index} className="rounded-lg border border-border p-3">
                    <div className="mb-2 flex items-center gap-2">
                      <Badge variant={rank === 0 ? "success" : "default"}>#{rank + 1}</Badge>
                      <span className="text-xs text-muted-foreground">source index {r.index}</span>
                    </div>
                    {docText && <p className="mb-2 text-sm">{docText}</p>}
                    <ScoreBar value={r.relevance_score} />
                  </li>
                );
              })}
            </ol>
          </Card>
        )}
      </Results>
    </div>
  );
}

// ==========================================================================
const SCORING_OPTIONS = [
  { value: "cosine", label: "Cosine" },
  { value: "dot", label: "Dot product" },
  { value: "euclidean", label: "Euclidean" },
];

function ScoreTab({ models }: { models: Model[] }) {
  const [model, setModel] = useState("");
  const [t1, setT1] = useState("");
  const [t2, setT2] = useState("");
  const [scoringType, setScoringType] = useState("cosine");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ScoreResponse | null>(null);

  const a = useMemo(() => lines(t1), [t1]);
  const b = useMemo(() => lines(t2), [t2]);
  const canRun = !!model && a.length > 0 && b.length > 0;

  /** Send a bare string when a single line, else the string[] (enables broadcast). */
  const pack = (arr: string[]): string | string[] => (arr.length === 1 ? arr[0] : arr);

  const run = async () => {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.post<ScoreResponse>("/v1/score", {
        model,
        text_1: pack(a),
        text_2: pack(b),
        scoring_type: scoringType,
      });
      setResult(res);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setLoading(false);
    }
  };

  const rows = result?.data ?? [];
  const bounded = scoringType !== "euclidean"; // euclidean isn't a 0–1 bar

  return (
    <div className="space-y-6">
      <Card className="space-y-4 p-5">
        <ModelField models={models} value={model} onChange={setModel} />
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="text-sm font-medium">Text 1</label>
            <Textarea
              rows={4}
              placeholder="one text per line"
              value={t1}
              onChange={(e) => setT1(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Text 2</label>
            <Textarea
              rows={4}
              placeholder="one text per line"
              value={t2}
              onChange={(e) => setT2(e.target.value)}
            />
          </div>
        </div>
        <div className="space-y-2 sm:max-w-[16rem]">
          <label className="text-sm font-medium">Scoring type</label>
          <CustomSelect options={SCORING_OPTIONS} value={scoringType} onChange={setScoringType} />
        </div>
        <p className="text-xs text-muted-foreground">
          A single line on either side broadcasts against the other side&apos;s list.
        </p>
        <RunButton loading={loading} disabled={!canRun} onClick={run} />
      </Card>

      <Results error={error} loading={loading} empty={!result} icon={Scale} emptyTitle="No scores yet" emptyDesc="Enter two sets of texts and choose a scoring type.">
        {result && (
          <Card className="space-y-3 p-5">
            <div className="flex items-center gap-2">
              <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Scores</span>
              <Badge variant="info">{scoringType}</Badge>
            </div>
            <ul className="space-y-3">
              {rows.map((r) => (
                <li key={r.index}>
                  {bounded ? (
                    <ScoreBar value={r.score} label={`Pair #${r.index}`} />
                  ) : (
                    <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2">
                      <span className="text-sm">Pair #{r.index}</span>
                      <span className="text-sm tabular-nums">{r.score.toFixed(4)}</span>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </Card>
        )}
      </Results>
    </div>
  );
}

// ==========================================================================
function ClassifyTab({ models }: { models: Model[] }) {
  const [model, setModel] = useState("");
  const [inputText, setInputText] = useState("");
  const [labelsText, setLabelsText] = useState("");
  const [temperature, setTemperature] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [sentInputs, setSentInputs] = useState<string[]>([]);

  const inputs = useMemo(() => lines(inputText), [inputText]);
  const labels = useMemo(
    () =>
      labelsText
        .split(",")
        .map((l) => l.trim())
        .filter(Boolean),
    [labelsText],
  );
  const canRun = !!model && inputs.length > 0 && labels.length >= 2;

  const run = async () => {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.post<ClassifyResponse>("/v1/classify", {
        model,
        input: inputs.length === 1 ? inputs[0] : inputs,
        labels,
        ...(temperature != null ? { temperature } : {}),
      });
      setResult(res);
      setSentInputs(inputs);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setLoading(false);
    }
  };

  // Group flat results by input index, each already desc by score.
  const grouped = useMemo(() => {
    const map = new Map<number, ClassifyRow[]>();
    for (const r of result?.results ?? []) {
      const list = map.get(r.index) ?? [];
      list.push(r);
      map.set(r.index, list);
    }
    for (const list of map.values()) list.sort((x, y) => y.score - x.score);
    return [...map.entries()].sort((x, y) => x[0] - y[0]);
  }, [result]);

  return (
    <div className="space-y-6">
      <Card className="space-y-4 p-5">
        <ModelField models={models} value={model} onChange={setModel} />
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <label className="text-sm font-medium">Input</label>
            <span className="text-xs text-muted-foreground">
              one per line · {fmtNumber(inputs.length)}
            </span>
          </div>
          <Textarea
            rows={5}
            placeholder={"This movie was fantastic.\nWorst purchase I ever made."}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Labels</label>
          <Input
            placeholder="positive, negative, neutral"
            value={labelsText}
            onChange={(e) => setLabelsText(e.target.value)}
            error={labelsText.trim() && labels.length < 2 ? "Provide at least 2 comma-separated labels" : undefined}
          />
          {labels.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {labels.map((l) => (
                <Badge key={l} variant="default">
                  {l}
                </Badge>
              ))}
            </div>
          )}
        </div>
        <div className="space-y-2 sm:max-w-[12rem]">
          <label className="text-sm font-medium">Temperature</label>
          <NumberInput
            min={0.01}
            max={10}
            step={0.01}
            value={temperature ?? undefined}
            placeholder="0.07"
            onChange={(v) => setTemperature(Number.isFinite(v) && v > 0 ? v : undefined)}
          />
        </div>
        <RunButton loading={loading} disabled={!canRun} onClick={run} />
      </Card>

      <Results error={error} loading={loading} empty={!result} icon={Tags} emptyTitle="Nothing classified yet" emptyDesc="Enter one or more inputs and at least two labels to score them.">
        {result && (
          <div className="space-y-4">
            {grouped.map(([idx, rows]) => (
              <Card key={idx} className="space-y-3 p-5">
                <div className="min-w-0">
                  <span className="text-sm font-medium">Input #{idx}</span>
                  {sentInputs[idx] && (
                    <p className="truncate text-xs text-muted-foreground" title={sentInputs[idx]}>
                      {sentInputs[idx]}
                    </p>
                  )}
                </div>
                <div className="space-y-2">
                  {rows.map((r, i) => (
                    <div key={r.label} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className={cn("text-sm", i === 0 && "font-medium")}>{r.label}</span>
                      </div>
                      <ScoreBar value={r.score} />
                    </div>
                  ))}
                </div>
              </Card>
            ))}
          </div>
        )}
      </Results>
    </div>
  );
}

// ==========================================================================
const POOLING_TYPE_OPTIONS = [
  { value: "CLS", label: "CLS" },
  { value: "MEAN", label: "Mean" },
  { value: "LAST", label: "Last" },
];

function PoolingTab({ models }: { models: Model[] }) {
  const [model, setModel] = useState("");
  const [inputText, setInputText] = useState("");
  const [poolingType, setPoolingType] = useState("CLS");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PoolingResponse | null>(null);

  const inputs = useMemo(() => lines(inputText), [inputText]);
  const canRun = !!model && inputs.length > 0;

  const run = async () => {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.post<PoolingResponse>("/v1/pooling", {
        model,
        input: inputs.length === 1 ? inputs[0] : inputs,
        pooling_type: poolingType,
        encoding_format: "float",
      });
      setResult(res);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setLoading(false);
    }
  };

  const rows = result?.data ?? [];

  return (
    <div className="space-y-6">
      <Card className="space-y-4 p-5">
        <ModelField models={models} value={model} onChange={setModel} />
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <label className="text-sm font-medium">Input</label>
            <span className="text-xs text-muted-foreground">
              one per line · {fmtNumber(inputs.length)}
            </span>
          </div>
          <Textarea
            rows={5}
            placeholder="Raw text to pool into hidden states…"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
        </div>
        <div className="space-y-2 sm:max-w-[16rem]">
          <label className="text-sm font-medium">Pooling type</label>
          <CustomSelect options={POOLING_TYPE_OPTIONS} value={poolingType} onChange={setPoolingType} />
        </div>
        <p className="text-xs text-muted-foreground">
          Debug tool — returns raw, unnormalized pooled hidden states.
        </p>
        <RunButton loading={loading} disabled={!canRun} onClick={run} />
      </Card>

      <Results error={error} loading={loading} empty={!result} icon={Layers} emptyTitle="No pooled vectors yet" emptyDesc="Enter text to inspect the raw pooled hidden state.">
        {result && (
          <div className="space-y-4">
            {rows.map((r, i) => (
              <PoolingCard key={i} index={i} vector={r.data} />
            ))}
          </div>
        )}
      </Results>
    </div>
  );
}

function PoolingCard({ index, vector }: { index: number; vector: number[] }) {
  const slice = vector.slice(0, 48);
  const max = Math.max(1e-6, ...slice.map((v) => Math.abs(v)));
  return (
    <Card className="space-y-3 p-5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Vector #{index}</span>
        <Badge variant="info">{fmtNumber(vector.length)} dims</Badge>
      </div>
      <div className="flex h-16 items-end gap-px rounded-md bg-muted/40 px-2 py-1">
        {slice.map((v, i) => (
          <div
            key={i}
            className="flex-1 rounded-sm"
            style={{
              height: `${(Math.abs(v) / max) * 100}%`,
              backgroundColor: v >= 0 ? "var(--color-primary)" : "var(--color-muted-foreground)",
            }}
            title={v.toFixed(6)}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {slice.slice(0, 16).map((v, i) => (
          <span
            key={i}
            className="inline-flex rounded bg-muted px-1.5 py-0.5 text-xs tabular-nums text-muted-foreground"
          >
            {v >= 0 ? "+" : ""}
            {v.toFixed(4)}
          </span>
        ))}
      </div>
    </Card>
  );
}

// ---- shared results wrapper (error / loading / empty / body) --------------
function Results({
  error,
  loading,
  empty,
  icon,
  emptyTitle,
  emptyDesc,
  children,
}: {
  error: string | null;
  loading: boolean;
  empty: boolean;
  icon: LucideIcon;
  emptyTitle: string;
  emptyDesc: string;
  children?: ReactNode;
}) {
  const Icon = icon;
  if (error)
    return (
      <Alert variant="error" title="Request failed">
        {error}
      </Alert>
    );
  if (loading)
    return (
      <div className="flex justify-center py-12">
        <Spinner />
      </div>
    );
  if (empty)
    return <EmptyState icon={<Icon className="h-8 w-8" />} title={emptyTitle} description={emptyDesc} />;
  return <>{children}</>;
}
