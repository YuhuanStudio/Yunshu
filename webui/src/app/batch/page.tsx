"use client";

import { useEffect, useRef, useState } from "react";
import {
  Button,
  NumberInput,
  Textarea,
  Card,
  Badge,
  Alert,
  Progress,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
  InlineStatus,
  cn,
  toast,
} from "yunui";
import { StatCard } from "yunui/patterns";
import { Play, Square, Download, Trash2, ListChecks } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import type { Model, CompletionResult, BatchItem } from "@/lib/types";

type InlineKind = "pending" | "processing" | "completed" | "failed";

const STATUS_MAP: Record<BatchItem["status"], InlineKind> = {
  pending: "pending",
  running: "processing",
  completed: "completed",
  error: "failed",
};

const STATUS_LABEL: Record<InlineKind, string> = {
  pending: "Pending",
  processing: "Running",
  completed: "Done",
  failed: "Failed",
};

function csvEscape(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}

export default function BatchPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);
  const [prompts, setPrompts] = useState("");

  const [items, setItems] = useState<BatchItem[]>([]);
  const [running, setRunning] = useState(false);
  const [tab, setTab] = useState("submit");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .get<{ data: Model[] }>("/v1/models")
      .then((res) => {
        if (cancelled) return;
        const list = res.data ?? [];
        setModels(list);
        setModel((prev) => prev || list[0]?.id || "");
      })
      .catch(() => {
        /* surfaced via connection status */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const total = items.length;
  const completed = items.filter((i) => i.status === "completed" || i.status === "error").length;
  const errored = items.filter((i) => i.status === "error").length;
  const totalTokens = items.reduce((sum, i) => sum + (i.tokens ?? 0), 0);
  const progressPct = total === 0 ? 0 : Math.round((completed / total) * 100);
  const queuedCount = prompts.split("\n").filter((l) => l.trim()).length;

  const patchItem = (id: string, patch: Partial<BatchItem>) => {
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)));
  };

  const runBatch = async () => {
    const lines = prompts
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    if (!model) {
      toast.error("Select a model first");
      return;
    }
    if (lines.length === 0) {
      toast.error("Enter at least one prompt");
      return;
    }

    const batch: BatchItem[] = lines.map((input, i) => ({
      id: `${Date.now()}-${i}`,
      input,
      status: "pending",
    }));
    setItems(batch);
    setRunning(true);
    setTab("results");

    const controller = new AbortController();
    abortRef.current = controller;

    for (const item of batch) {
      if (controller.signal.aborted) break;
      patchItem(item.id, { status: "running" });
      try {
        const res = await api.post<CompletionResult>(
          "/v1/completions",
          { model, prompt: item.input, max_tokens: maxTokens, temperature },
          controller.signal,
        );
        patchItem(item.id, {
          status: "completed",
          output: res.choices[0]?.text ?? "",
          tokens: res.usage?.completion_tokens ?? 0,
        });
      } catch (e) {
        if (controller.signal.aborted) break;
        const msg =
          e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Request failed";
        patchItem(item.id, { status: "error", error: msg });
      }
    }

    if (controller.signal.aborted) {
      setItems((prev) =>
        prev.map((it) =>
          it.status === "pending" || it.status === "running"
            ? { ...it, status: "error", error: "Aborted" }
            : it,
        ),
      );
    }

    abortRef.current = null;
    setRunning(false);
  };

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
  };

  const clearAll = () => {
    if (running) stop();
    setItems([]);
  };

  const exportCsv = () => {
    if (items.length === 0) return;
    const header = ["input", "output", "tokens", "status"];
    const rows = items.map((it) =>
      [
        csvEscape(it.input),
        csvEscape(it.output ?? it.error ?? ""),
        String(it.tokens ?? ""),
        it.status,
      ].join(","),
    );
    const csv = [header.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `batch-${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <PageShell
      title="Batch"
      description="Run many prompts through a model and collect the completions."
      width="wide"
    >
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="submit">Submit</TabsTrigger>
          <TabsTrigger value="results">
            Results{total > 0 ? ` (${completed}/${total})` : ""}
          </TabsTrigger>
        </TabsList>

        {/* ---- Submit ---- */}
        <TabsContent value="submit" className="mt-4">
          <Card className="space-y-5 p-5">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <label className="space-y-1.5">
                <span className="text-sm font-medium">Model</span>
                <ModelPicker models={models} value={model} onChange={setModel} />
              </label>

              <label className="space-y-1.5">
                <span className="text-sm font-medium">Max tokens</span>
                <NumberInput value={maxTokens} onChange={setMaxTokens} min={1} max={8192} step={16} />
              </label>

              <label className="space-y-1.5">
                <span className="text-sm font-medium">Temperature</span>
                <NumberInput
                  value={temperature}
                  onChange={setTemperature}
                  min={0}
                  max={2}
                  step={0.1}
                />
              </label>
            </div>

            <label className="block space-y-1.5">
              <span className="text-sm font-medium">Prompts (one per line)</span>
              <Textarea
                rows={8}
                placeholder={
                  "Summarize the plot of Hamlet.\nExplain photosynthesis in one sentence.\n…"
                }
                value={prompts}
                onChange={(e) => setPrompts(e.target.value)}
              />
            </label>

            <div className="flex items-center justify-between gap-3">
              <span className="text-sm text-muted-foreground">{queuedCount} prompt(s) queued</span>
              {running ? (
                <Button variant="secondary" onClick={stop}>
                  <Square className="h-4 w-4" /> Stop
                </Button>
              ) : (
                <Button onClick={runBatch} disabled={!model || queuedCount === 0}>
                  <Play className="h-4 w-4" /> Run batch
                </Button>
              )}
            </div>
          </Card>
        </TabsContent>

        {/* ---- Results ---- */}
        <TabsContent value="results" className="mt-4 space-y-4">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard icon={ListChecks} label="Total" value={fmtNumber(total)} />
            <StatCard label="Completed" value={fmtNumber(completed)} tone="emerald" />
            <StatCard
              label="Errors"
              value={fmtNumber(errored)}
              tone={errored > 0 ? "red" : undefined}
            />
            <StatCard label="Output tokens" value={fmtNumber(totalTokens)} tone="blue" />
          </div>

          <Card className="space-y-4 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span className="font-medium">Progress</span>
                  <span className="text-muted-foreground">
                    {completed}/{total} · {progressPct}%
                  </span>
                </div>
                <Progress value={progressPct} />
              </div>
              <div className="flex items-center gap-2">
                {running && (
                  <Button variant="secondary" size="sm" onClick={stop}>
                    <Square className="h-4 w-4" /> Stop
                  </Button>
                )}
                <Button variant="secondary" size="sm" onClick={exportCsv} disabled={total === 0}>
                  <Download className="h-4 w-4" /> Export CSV
                </Button>
                <Button variant="ghost" size="sm" onClick={clearAll} disabled={total === 0}>
                  <Trash2 className="h-4 w-4" /> Clear
                </Button>
              </div>
            </div>

            {errored > 0 && (
              <Alert variant="warning">
                {fmtNumber(errored)} of {fmtNumber(total)} prompts failed.
              </Alert>
            )}

            {total === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No batch yet. Submit prompts to see results here.
              </p>
            ) : (
              <div className="divide-y divide-border">
                {items.map((it, idx) => {
                  const kind = STATUS_MAP[it.status];
                  return (
                    <div key={it.id} className="space-y-2 py-3">
                      <div className="flex items-center gap-3">
                        <span className="w-8 shrink-0 text-xs tabular-nums text-muted-foreground">
                          #{idx + 1}
                        </span>
                        <InlineStatus status={kind} label={STATUS_LABEL[kind]} />
                        <span className="min-w-0 flex-1 truncate text-sm" title={it.input}>
                          {it.input}
                        </span>
                        {typeof it.tokens === "number" && (
                          <Badge variant="info">{fmtNumber(it.tokens)} tok</Badge>
                        )}
                      </div>

                      {it.output != null && it.output !== "" && (
                        <pre
                          className={cn(
                            "ml-11 max-h-48 overflow-auto whitespace-pre-wrap rounded-md",
                            "bg-muted p-3 font-mono text-xs text-foreground",
                          )}
                        >
                          {it.output}
                        </pre>
                      )}

                      {it.status === "error" && it.error && (
                        <p className="ml-11 text-xs text-error">{it.error}</p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </TabsContent>
      </Tabs>
    </PageShell>
  );
}
