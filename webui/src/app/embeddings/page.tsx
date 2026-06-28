"use client";

import { useEffect, useState } from "react";
import { VectorSquare, Send, Copy, Check, Loader2 } from "lucide-react";

export default function EmbeddingsPage() {
  const [input, setInput] = useState("");
  const [model, setModel] = useState("");
  const [models, setModels] = useState<{ id: string }[]>([]);
  const [embeddings, setEmbeddings] = useState<number[][]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [totalTokens, setTotalTokens] = useState(0);

  useEffect(() => {
    fetch("/v1/models")
      .then((r) => r.json())
      .then((d) => setModels(d.data || []))
      .catch(() => {});
  }, []);

  const handleSubmit = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError("");
    setEmbeddings([]);

    try {
      const res = await fetch("/v1/embeddings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model || undefined,
          input: input.trim(),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error?.message || `Error ${res.status}`);
        return;
      }
      if (data.data) {
        const embs = data.data
          .sort((a: { index: number }, b: { index: number }) => a.index - b.index)
          .map((d: { embedding: number[] }) => d.embedding);
        setEmbeddings(embs);
      }
      if (data.usage) {
        setTotalTokens(data.usage.total_tokens || 0);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const copyEmbeddings = () => {
    navigator.clipboard.writeText(JSON.stringify(embeddings, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="flex items-center gap-3">
        <VectorSquare className="w-6 h-6 text-[var(--color-accent)]" />
        <h1 className="text-xl font-bold">Embeddings</h1>
      </div>

      {/* Input */}
      <div className="space-y-3">
        <div className="flex gap-2">
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="px-3 py-2 text-sm bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-lg min-w-[200px]"
          >
            <option value="">Default Model</option>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id}
              </option>
            ))}
          </select>
        </div>

        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter text to embed (one per line for batch)..."
          className="w-full h-32 px-4 py-3 bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-lg resize-y text-sm"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit();
          }}
        />

        <button
          onClick={handleSubmit}
          disabled={loading || !input.trim()}
          className="flex items-center gap-2 px-4 py-2 bg-[var(--color-accent)] text-white rounded-lg text-sm font-medium hover:opacity-90 disabled:opacity-50"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          Generate Embeddings
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Results */}
      {embeddings.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">
              Results ({embeddings.length} embedding{embeddings.length > 1 ? "s" : ""}, {totalTokens} tokens)
            </h2>
            <button
              onClick={copyEmbeddings}
              className="flex items-center gap-1 px-3 py-1 text-xs bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded hover:bg-[var(--color-bg-secondary)]"
            >
              {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              {copied ? "Copied" : "Copy JSON"}
            </button>
          </div>

          {embeddings.map((emb, i) => (
            <div key={i} className="bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-[var(--color-text-secondary)]">Embedding {i}</span>
                <span className="text-xs text-[var(--color-text-secondary)]">
                  {emb.length} dimensions
                </span>
              </div>
              <div className="flex gap-1 flex-wrap">
                {emb.slice(0, 20).map((v, j) => (
                  <span
                    key={j}
                    className="px-1.5 py-0.5 text-xs font-mono bg-[var(--color-bg-tertiary)] rounded"
                  >
                    {v.toFixed(4)}
                  </span>
                ))}
                {emb.length > 20 && (
                  <span className="px-1.5 py-0.5 text-xs text-[var(--color-text-secondary)]">
                    ... +{emb.length - 20} more
                  </span>
                )}
              </div>
              {/* Mini bar chart of first 50 dimensions */}
              <div className="flex items-end gap-px mt-3 h-8">
                {emb.slice(0, 50).map((v, j) => {
                  const h = Math.min(Math.abs(v) * 100, 100);
                  return (
                    <div
                      key={j}
                      className="flex-1 bg-[var(--color-accent)] opacity-60 rounded-t"
                      style={{ height: `${Math.max(h, 4)}%` }}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
