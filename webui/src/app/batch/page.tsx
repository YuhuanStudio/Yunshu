"use client";

import { useState, useRef, useCallback } from "react";
import {
  Layers,
  Play,
  Download,
  Trash2,
  Upload,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
} from "lucide-react";

const API_BASE =
  typeof window !== "undefined"
    ? window.location.origin
    : "http://localhost:8000";

interface BatchItem {
  id: string;
  input: string;
  output?: string;
  status: "pending" | "running" | "completed" | "error";
  error?: string;
  tokens?: number;
}

export default function BatchPage() {
  const [tab, setTab] = useState<"submit" | "results">("submit");
  const [model, setModel] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);
  const [inputText, setInputText] = useState("");
  const [items, setItems] = useState<BatchItem[]>([]);
  const [processing, setProcessing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const loadModels = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/models`);
      const data = await res.json();
      const ids = (data.data || []).map((m: any) => m.id);
      setModels(ids);
      if (!model && ids.length > 0) setModel(ids[0]);
    } catch {}
  }, []);

  useState(() => { loadModels(); });

  const handleSubmit = async () => {
    const lines = inputText
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length === 0 || processing) return;

    const batchItems: BatchItem[] = lines.map((line, i) => ({
      id: `batch-${Date.now()}-${i}`,
      input: line,
      status: "pending" as const,
    }));
    setItems(batchItems);
    setProcessing(true);
    setTab("results");

    const controller = new AbortController();
    abortRef.current = controller;

    for (let i = 0; i < batchItems.length; i++) {
      if (controller.signal.aborted) break;

      setItems((prev) =>
        prev.map((item, idx) =>
          idx === i ? { ...item, status: "running" as const } : item
        )
      );

      try {
        const res = await fetch(`${API_BASE}/v1/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: model || "default",
            prompt: batchItems[i].input,
            max_tokens: maxTokens,
            temperature,
          }),
          signal: controller.signal,
        });

        if (!res.ok) {
          const err = await res.text();
          setItems((prev) =>
            prev.map((item, idx) =>
              idx === i
                ? { ...item, status: "error" as const, error: err }
                : item
            )
          );
          continue;
        }

        const data = await res.json();
        const output = data.choices?.[0]?.text || "";
        const tokens = data.usage?.completion_tokens || 0;
        setItems((prev) =>
          prev.map((item, idx) =>
            idx === i
              ? { ...item, status: "completed" as const, output, tokens }
              : item
          )
        );
      } catch (e: any) {
        if (e.name !== "AbortError") {
          setItems((prev) =>
            prev.map((item, idx) =>
              idx === i
                ? { ...item, status: "error" as const, error: e.message }
                : item
            )
          );
        }
      }
    }

    setProcessing(false);
    abortRef.current = null;
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setProcessing(false);
  };

  const completedCount = items.filter((i) => i.status === "completed").length;
  const errorCount = items.filter((i) => i.status === "error").length;
  const totalCount = items.length;
  const progress =
    totalCount > 0
      ? ((completedCount + errorCount) / totalCount) * 100
      : 0;

  const exportResults = () => {
    const csv = [
      "input,output,status,tokens",
      ...items.map((i) =>
        [
          JSON.stringify(i.input),
          JSON.stringify(i.output || ""),
          i.status,
          i.tokens || 0,
        ].join(",")
      ),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `batch-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[var(--color-border)]">
        <h1 className="text-xl font-semibold flex items-center gap-2">
          <Layers className="w-5 h-5 text-[var(--color-accent)]" />
          Batch Inference
        </h1>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Submit multiple prompts for batch processing
        </p>
      </div>

      <div className="flex-1 overflow-auto p-6 space-y-4">
        {/* Tab Bar */}
        <div className="flex border-b border-[var(--color-border)]">
          {[
            { id: "submit" as const, label: "Submit" },
            { id: "results" as const, label: "Results" },
          ].map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                tab === id
                  ? "border-[var(--color-accent)] text-[var(--color-accent)]"
                  : "border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Submit Tab */}
        {tab === "submit" && (
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-1">
                <label className="text-xs text-[var(--color-text-secondary)] block mb-1">
                  Model
                </label>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  onClick={loadModels}
                  className="w-full px-3 py-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm outline-none"
                >
                  {models.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
              <div className="w-32">
                <label className="text-xs text-[var(--color-text-secondary)] block mb-1">
                  Max Tokens
                </label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value) || 128)}
                  className="w-full px-3 py-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm outline-none"
                />
              </div>
              <div className="w-32">
                <label className="text-xs text-[var(--color-text-secondary)] block mb-1">
                  Temperature
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={temperature}
                  onChange={(e) =>
                    setTemperature(parseFloat(e.target.value) || 0.7)
                  }
                  className="w-full px-3 py-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm outline-none"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-[var(--color-text-secondary)]">
                  Prompts (one per line)
                </label>
                <span className="text-xs text-[var(--color-text-secondary)]">
                  {inputText.split("\n").filter((l) => l.trim()).length} prompts
                </span>
              </div>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder={"Enter prompt 1\nEnter prompt 2\nEnter prompt 3"}
                rows={10}
                className="w-full p-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm font-mono resize-none outline-none focus:border-[var(--color-accent)]"
              />
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleSubmit}
                disabled={
                  !inputText.trim() || processing
                }
                className="px-4 py-2 text-sm rounded-lg bg-[var(--color-accent)] text-white hover:opacity-90 disabled:opacity-40 flex items-center gap-1.5"
              >
                <Play className="w-3.5 h-3.5" />
                Run Batch
              </button>
            </div>
          </div>
        )}

        {/* Results Tab */}
        {tab === "results" && (
          <div className="space-y-4">
            {/* Progress */}
            {items.length > 0 && (
              <div className="flex items-center gap-4">
                <div className="flex-1 h-2 rounded-full bg-[var(--color-bg-tertiary)] overflow-hidden">
                  <div
                    className="h-full bg-[var(--color-accent)] transition-all duration-300 rounded-full"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <span className="text-xs text-[var(--color-text-secondary)] shrink-0">
                  {completedCount + errorCount}/{totalCount}
                </span>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2">
              {processing && (
                <button
                  onClick={handleStop}
                  className="px-3 py-1.5 text-sm rounded-lg bg-[var(--color-danger)] text-white hover:opacity-90"
                >
                  Stop
                </button>
              )}
              {completedCount > 0 && (
                <button
                  onClick={exportResults}
                  className="px-3 py-1.5 text-sm rounded-lg border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] flex items-center gap-1.5"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export CSV
                </button>
              )}
              {items.length > 0 && (
                <button
                  onClick={() => setItems([])}
                  className="px-3 py-1.5 text-sm rounded-lg border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] flex items-center gap-1.5"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear
                </button>
              )}
            </div>

            {/* Results Table */}
            {items.length === 0 ? (
              <div className="text-center py-12 text-[var(--color-text-secondary)]">
                <Layers className="w-12 h-12 mx-auto mb-3 opacity-40" />
                <p className="text-sm">No batch results yet</p>
              </div>
            ) : (
              <div className="space-y-2">
                {items.map((item, i) => (
                  <div
                    key={item.id}
                    className="p-3 border border-[var(--color-border)] rounded-lg bg-[var(--color-bg-primary)]"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-mono text-[var(--color-text-secondary)] w-6">
                        {i + 1}
                      </span>
                      {item.status === "pending" && (
                        <Clock className="w-3.5 h-3.5 text-[var(--color-text-secondary)]" />
                      )}
                      {item.status === "running" && (
                        <Loader2 className="w-3.5 h-3.5 text-[var(--color-accent)] animate-spin" />
                      )}
                      {item.status === "completed" && (
                        <CheckCircle className="w-3.5 h-3.5 text-[var(--color-success)]" />
                      )}
                      {item.status === "error" && (
                        <XCircle className="w-3.5 h-3.5 text-[var(--color-danger)]" />
                      )}
                      <span className="text-sm flex-1 truncate">
                        {item.input}
                      </span>
                      {item.tokens != null && (
                        <span className="text-xs text-[var(--color-text-secondary)]">
                          {item.tokens} tokens
                        </span>
                      )}
                    </div>
                    {item.output && (
                      <pre className="mt-2 ml-9 text-xs font-mono whitespace-pre-wrap break-words text-[var(--color-text-secondary)] max-h-32 overflow-auto bg-[var(--color-bg-tertiary)] p-2 rounded">
                        {item.output}
                      </pre>
                    )}
                    {item.error && (
                      <p className="mt-2 ml-9 text-xs text-[var(--color-danger)]">
                        {item.error}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
