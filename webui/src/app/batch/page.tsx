"use client";

import { useMemo, useState } from "react";
import {
  Button,
  NumberInput,
  Textarea,
  Card,
  Badge,
  Alert,
  Spinner,
  FileDropzone,
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
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
import { Play, Download, ListChecks, Upload, FileText } from "lucide-react";
import { api, ApiError, usePolling } from "@/lib/api";
import { authHeaders } from "@/lib/auth";
import { fmtNumber, fmtDuration } from "@/lib/format";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import type { Model } from "@/lib/types";

/**
 * Page-local mirror of the backend Batch contract (OpenAI-batch shaped). Kept
 * here rather than in lib/types so this page owns its own shapes.
 *
 *   POST /v1/batch { requests:[{custom_id, url?, body}], max_concurrent, timeout, model? }
 *        → BatchObject  (BLOCKING — runs synchronously and returns the result)
 *   POST /v1/batch/upload/csv (multipart: file + model + max_tokens + max_concurrent)
 *        → BatchObject
 *   GET  /v1/batch/{id}/results.csv → CSV download
 */
interface BatchRequestResult {
  custom_id: string;
  status: "success" | "error";
  response?: unknown;
  error?: unknown;
}
interface BatchObject {
  id: string;
  object: "batch";
  status: string;
  request_counts: { total: number; completed: number; failed: number };
  results: BatchRequestResult[];
  total: number;
  succeeded: number;
  failed: number;
  elapsed_s: number;
}

interface BatchRequest {
  custom_id: string;
  url: "/v1/chat/completions" | "/v1/completions";
  body: Record<string, unknown>;
}

function errMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return "Request failed";
}

/** Pull the assistant text out of a chat/completions response envelope. */
function extractText(response: unknown): string {
  if (!response || typeof response !== "object") return "";
  const r = response as Record<string, unknown>;
  const body = (r.body && typeof r.body === "object" ? r.body : r) as Record<string, unknown>;
  const choices = body.choices as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(choices) && choices.length > 0) {
    const c = choices[0];
    const message = c.message as Record<string, unknown> | undefined;
    if (message && typeof message.content === "string") return message.content;
    if (typeof c.text === "string") return c.text;
  }
  return "";
}

/** Render an error field (string or `{ message }`-ish object) as text. */
function extractError(error: unknown): string {
  if (!error) return "";
  if (typeof error === "string") return error;
  if (typeof error === "object") {
    const e = error as Record<string, unknown>;
    if (typeof e.message === "string") return e.message;
    return JSON.stringify(error);
  }
  return String(error);
}

export default function BatchPage() {
  const models = usePolling<{ data: Model[] }>((s) => api.get("/v1/models", s), 15000);
  const modelList = useMemo(() => models.data?.data ?? [], [models.data]);

  const [model, setModel] = useState("");
  const activeModel = model || modelList.find((m) => m.loaded)?.id || modelList[0]?.id || "";

  const [maxTokens, setMaxTokens] = useState(128);
  const [maxConcurrent, setMaxConcurrent] = useState(4);
  const [timeout, setTimeoutS] = useState(300);
  const [prompts, setPrompts] = useState("");
  const [mode, setMode] = useState("builder");

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BatchObject | null>(null);

  const queuedCount = prompts.split("\n").filter((l) => l.trim()).length;

  const submit = async (fn: () => Promise<BatchObject>) => {
    setRunning(true);
    setError(null);
    try {
      const res = await fn();
      setResult(res);
    } catch (e) {
      setError(errMessage(e));
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  const runBuilder = () => {
    if (!activeModel) {
      toast.error("Select a model first");
      return;
    }
    const lines = prompts
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length === 0) {
      toast.error("Enter at least one prompt");
      return;
    }
    const requests: BatchRequest[] = lines.map((prompt, i) => ({
      custom_id: `req-${i + 1}`,
      url: "/v1/chat/completions",
      body: {
        model: activeModel,
        messages: [{ role: "user", content: prompt }],
        max_tokens: maxTokens,
      },
    }));
    void submit(() =>
      api.post<BatchObject>("/v1/batch", {
        requests,
        max_concurrent: maxConcurrent,
        timeout,
        model: activeModel,
      }),
    );
  };

  const runCsv = (files: File[]) => {
    const file = files[0];
    if (!file) return;
    if (!activeModel) {
      toast.error("Select a model first");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    form.append("model", activeModel);
    form.append("max_tokens", String(maxTokens));
    form.append("max_concurrent", String(maxConcurrent));
    void submit(() => api.postForm<BatchObject>("/v1/batch/upload/csv", form));
  };

  const downloadCsv = async () => {
    if (!result) return;
    try {
      const res = await fetch(`/v1/batch/${result.id}/results.csv`, { headers: authHeaders() });
      if (res.status === 409) {
        toast.error("Batch still in progress — results not ready yet");
        return;
      }
      if (!res.ok) {
        toast.error(`Download failed (${res.status})`);
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `batch-${result.id}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      toast.error(errMessage(e));
    }
  };

  return (
    <PageShell
      title="Batch"
      description="Run many requests through a model in one blocking call and collect the results."
      width="wide"
    >
      {/* ---- Config (shared) ---- */}
      <Card className="p-5">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <label className="space-y-1.5">
            <span className="text-sm font-medium">Model</span>
            <ModelPicker models={modelList} value={activeModel} onChange={setModel} />
          </label>
          <label className="space-y-1.5">
            <span className="text-sm font-medium">Max tokens</span>
            <NumberInput value={maxTokens} onChange={setMaxTokens} min={1} max={8192} step={16} />
          </label>
          <label className="space-y-1.5">
            <span className="text-sm font-medium">Max concurrent</span>
            <NumberInput value={maxConcurrent} onChange={setMaxConcurrent} min={1} max={64} step={1} />
          </label>
          <label className="space-y-1.5">
            <span className="text-sm font-medium">Timeout (s)</span>
            <NumberInput value={timeout} onChange={setTimeoutS} min={1} max={3600} step={30} />
          </label>
        </div>
        {models.error && (
          <p className="mt-2 text-xs text-error">
            Could not load models: {errMessage(models.error)}
          </p>
        )}
      </Card>

      {/* ---- Input modes ---- */}
      <Tabs value={mode} onValueChange={setMode} className="mt-4">
        <TabsList>
          <TabsTrigger value="builder">
            <ListChecks className="h-4 w-4" /> Prompt builder
          </TabsTrigger>
          <TabsTrigger value="csv">
            <Upload className="h-4 w-4" /> CSV upload
          </TabsTrigger>
        </TabsList>

        <TabsContent value="builder" className="mt-4">
          <Card className="space-y-4 p-5">
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
              <span className="text-sm text-muted-foreground">
                {fmtNumber(queuedCount)} request(s) → /v1/chat/completions
              </span>
              <Button onClick={runBuilder} disabled={running || !activeModel || queuedCount === 0}>
                {running ? <Spinner size="sm" /> : <Play className="h-4 w-4" />} Run batch
              </Button>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="csv" className="mt-4">
          <Card className="space-y-4 p-5">
            <FileDropzone
              accept=".csv,text/csv"
              disabled={running}
              onFiles={runCsv}
              icon={<FileText className="h-6 w-6" />}
              label="Drop a CSV or click to choose"
              hint="Columns: custom_id, prompt | messages_json, system_prompt, max_tokens, temperature"
            />
            <p className="text-xs text-muted-foreground">
              The selected model, max tokens and max concurrent above are applied to the upload.
            </p>
          </Card>
        </TabsContent>
      </Tabs>

      {/* ---- Running state ---- */}
      {running && (
        <Card className="mt-4 flex items-center justify-center gap-3 p-8">
          <Spinner />
          <span className="text-sm text-muted-foreground">
            Running batch synchronously — this blocks until every request finishes…
          </span>
        </Card>
      )}

      {error && !running && (
        <Alert variant="error" className="mt-4" title="Batch failed">
          {error}
        </Alert>
      )}

      {/* ---- Results ---- */}
      {result && !running && (
        <div className="mt-4 space-y-4">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard icon={ListChecks} label="Total" value={fmtNumber(result.request_counts.total)} />
            <StatCard
              label="Succeeded"
              value={fmtNumber(result.succeeded ?? result.request_counts.completed)}
              tone="emerald"
            />
            <StatCard
              label="Failed"
              value={fmtNumber(result.failed ?? result.request_counts.failed)}
              tone={(result.failed ?? result.request_counts.failed) > 0 ? "red" : undefined}
            />
            <StatCard label="Elapsed" value={fmtDuration(result.elapsed_s)} tone="blue" />
          </div>

          <Card className="space-y-4 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-sm">
                <span className="font-medium">Results</span>
                <Badge variant="default">{result.status}</Badge>
                <span className="text-xs text-muted-foreground">batch {result.id}</span>
              </div>
              <Button variant="secondary" size="sm" onClick={downloadCsv}>
                <Download className="h-4 w-4" /> Download results.csv
              </Button>
            </div>

            {(result.failed ?? result.request_counts.failed) > 0 && (
              <Alert variant="warning">
                {fmtNumber(result.failed ?? result.request_counts.failed)} of{" "}
                {fmtNumber(result.request_counts.total)} requests failed.
              </Alert>
            )}

            {result.results.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No results returned.
              </p>
            ) : (
              <Table responsive>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-40">Custom ID</TableHead>
                    <TableHead className="w-24">Status</TableHead>
                    <TableHead>Response</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.results.map((r) => {
                    const ok = r.status === "success";
                    const text = ok ? extractText(r.response) : extractError(r.error);
                    return (
                      <TableRow key={r.custom_id}>
                        <TableCell label="Custom ID" className="font-mono text-xs">
                          {r.custom_id}
                        </TableCell>
                        <TableCell label="Status">
                          <Badge variant={ok ? "success" : "error"}>{r.status}</Badge>
                        </TableCell>
                        <TableCell label="Response">
                          <pre
                            className={cn(
                              "max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-xs",
                              ok ? "text-foreground" : "text-error",
                            )}
                          >
                            {text || (ok ? "(empty)" : "Unknown error")}
                          </pre>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            )}
          </Card>
        </div>
      )}
    </PageShell>
  );
}
