"use client";

import { useEffect, useState } from "react";
import {
  Button,
  Textarea,
  Card,
  Badge,
  Alert,
  EmptyState,
  Spinner,
  Sparkline,
  cn,
  toast,
} from "yunui";
import { StatCard } from "yunui/patterns";
import { Copy, Hash, Sparkles, Boxes } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, ApiError } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import type { Model, EmbeddingResult } from "@/lib/types";

const PREVIEW_CHIPS = 24;
const SPARK_DIMS = 64;

export default function EmbeddingsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EmbeddingResult | null>(null);

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

  const generate = async () => {
    if (!model || !input.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.post<EmbeddingResult>("/v1/embeddings", {
        model,
        input,
      });
      setResult(res);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : (e as Error).message;
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const copyJson = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      toast.success("Copied", "Full embedding JSON copied to clipboard");
    } catch {
      toast.error("Copy failed", "Could not access the clipboard");
    }
  };

  const rows = result?.data ?? [];

  return (
    <PageShell
      title="Embeddings"
      description="Generate a vector embedding for input text and visualize it."
      width="narrow"
    >
      <div className="space-y-6">
        <Card className="space-y-4 p-5">
          <div className="space-y-2">
            <label className="text-sm font-medium">Model</label>
            <ModelPicker models={models} value={model} onChange={setModel} />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Input text</label>
            <Textarea
              rows={5}
              placeholder="Enter text to embed…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
          </div>

          <div className="flex justify-end">
            <Button onClick={generate} disabled={loading || !model || !input.trim()}>
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
            title="No embedding yet"
            description="Pick a model, enter some text, and generate a vector embedding to visualize it."
          />
        )}

        {result && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <StatCard
                icon={Hash}
                label="Total tokens"
                value={fmtNumber(result.usage.total_tokens)}
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
              <EmbeddingCard key={row.index} index={row.index} embedding={row.embedding} />
            ))}

            <div className="flex justify-end">
              <Button variant="secondary" size="sm" onClick={copyJson}>
                <Copy className="h-4 w-4" /> Copy JSON
              </Button>
            </div>
          </div>
        )}
      </div>
    </PageShell>
  );
}

function EmbeddingCard({ index, embedding }: { index: number; embedding: number[] }) {
  const chips = embedding.slice(0, PREVIEW_CHIPS);
  const spark = embedding.slice(0, SPARK_DIMS);

  return (
    <Card className="space-y-4 p-5">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium">Vector #{index}</span>
        <Badge variant="info">{fmtNumber(embedding.length)} dims</Badge>
      </div>

      <Sparkline area data={spark} className="h-16 w-full" />

      <div className="flex flex-wrap gap-1.5">
        {chips.map((v, i) => (
          <span
            key={i}
            className={cn(
              "inline-flex rounded bg-muted px-1.5 py-0.5 text-xs tabular-nums text-muted-foreground",
            )}
          >
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
