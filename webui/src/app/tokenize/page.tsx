"use client";

import { useState, useCallback } from "react";
import { ArrowRight, ArrowLeft, Hash, Type, Copy, AlertCircle } from "lucide-react";

const API_BASE =
  typeof window !== "undefined"
    ? window.location.origin
    : "http://localhost:8000";

interface TokenizeResult {
  tokens: number[];
  model: string;
}

interface DetokenizeResult {
  text: string;
  model: string;
}

export default function TokenizePage() {
  const [tab, setTab] = useState<"tokenize" | "detokenize" | "count">("tokenize");
  const [model, setModel] = useState("");
  const [models, setModels] = useState<string[]>([]);

  // Tokenize state
  const [text, setText] = useState("");
  const [addSpecial, setAddSpecial] = useState(true);
  const [tokens, setTokens] = useState<number[]>([]);
  const [tokenizeError, setTokenizeError] = useState("");

  // Detokenize state
  const [tokenIds, setTokenIds] = useState("");
  const [decodedText, setDecodedText] = useState("");
  const [detokenizeError, setDetokenizeError] = useState("");

  // Count state
  const [countText, setCountText] = useState("");
  const [maxTokens, setMaxTokens] = useState(0);
  const [tokenCount, setTokenCount] = useState<number | null>(null);
  const [overLimit, setOverLimit] = useState(false);
  const [countError, setCountError] = useState("");

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

  const handleTokenize = async () => {
    setTokenizeError("");
    setTokens([]);
    try {
      const res = await fetch(`${API_BASE}/v1/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model || "default",
          text,
          add_special_tokens: addSpecial,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        setTokenizeError(err.detail || "Tokenize failed");
        return;
      }
      const data: TokenizeResult = await res.json();
      setTokens(data.tokens);
    } catch (e: any) {
      setTokenizeError(e.message);
    }
  };

  const handleDetokenize = async () => {
    setDetokenizeError("");
    setDecodedText("");
    try {
      const ids = tokenIds
        .split(/[,\s\n]+/)
        .map((s) => s.trim())
        .filter(Boolean)
        .map(Number);
      if (ids.some(isNaN)) {
        setDetokenizeError("Invalid token IDs — must be numbers");
        return;
      }
      const res = await fetch(`${API_BASE}/v1/detokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model || "default",
          tokens: ids,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        setDetokenizeError(err.detail || "Detokenize failed");
        return;
      }
      const data: DetokenizeResult = await res.json();
      setDecodedText(data.text);
    } catch (e: any) {
      setDetokenizeError(e.message);
    }
  };

  const handleCount = async () => {
    setCountError("");
    setTokenCount(null);
    setOverLimit(false);
    try {
      const res = await fetch(`${API_BASE}/v1/token_count`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model || "default",
          prompt: countText,
          max_tokens: maxTokens,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        setCountError(err.detail || "Count failed");
        return;
      }
      const data = await res.json();
      setTokenCount(data.token_count);
      setOverLimit(data.over_context_limit);
    } catch (e: any) {
      setCountError(e.message);
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[var(--color-border)]">
        <h1 className="text-xl font-semibold flex items-center gap-2">
          <Hash className="w-5 h-5 text-[var(--color-accent)]" />
          Tokenizer
        </h1>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Tokenize, detokenize, and count tokens
        </p>
      </div>

      <div className="flex-1 overflow-auto p-6 space-y-4">
        {/* Model Selection */}
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-[var(--color-text-secondary)] shrink-0">
            Model
          </label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            onClick={loadModels}
            className="px-3 py-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm text-[var(--color-text-primary)] outline-none"
          >
            {models.length === 0 && <option value="">Loading...</option>}
            {models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        {/* Tab Bar */}
        <div className="flex border-b border-[var(--color-border)]">
          {[
            { id: "tokenize" as const, label: "Tokenize", icon: ArrowRight },
            { id: "detokenize" as const, label: "Detokenize", icon: ArrowLeft },
            { id: "count" as const, label: "Count", icon: Hash },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
                tab === id
                  ? "border-[var(--color-accent)] text-[var(--color-accent)]"
                  : "border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </button>
          ))}
        </div>

        {/* Tokenize Tab */}
        {tab === "tokenize" && (
          <div className="space-y-4">
            <div>
              <label className="text-xs text-[var(--color-text-secondary)] mb-1 block">
                Text to tokenize
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter text to tokenize..."
                className="w-full h-32 p-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm font-mono text-[var(--color-text-primary)] resize-none outline-none"
              />
            </div>
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={addSpecial}
                  onChange={(e) => setAddSpecial(e.target.checked)}
                  className="accent-[var(--color-accent)]"
                />
                <span className="text-xs text-[var(--color-text-secondary)]">
                  Add special tokens
                </span>
              </label>
              <button
                onClick={handleTokenize}
                disabled={!text.trim()}
                className="px-4 py-2 text-sm rounded-lg bg-[var(--color-accent)] text-white hover:opacity-90 disabled:opacity-40 flex items-center gap-1.5"
              >
                <ArrowRight className="w-3.5 h-3.5" />
                Tokenize
              </button>
            </div>
            {tokenizeError && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-[var(--color-danger)] text-sm">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {tokenizeError}
              </div>
            )}
            {tokens.length > 0 && (
              <div className="border border-[var(--color-border)] rounded-xl p-4 bg-[var(--color-bg-primary)]">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium">
                    Tokens ({tokens.length})
                  </span>
                  <button
                    onClick={() =>
                      navigator.clipboard.writeText(
                        `[${tokens.join(", ")}]`
                      )
                    }
                    className="p-1.5 rounded hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                  >
                    <Copy className="w-3.5 h-3.5" />
                  </button>
                </div>
                <div className="flex flex-wrap gap-1.5 max-h-64 overflow-auto">
                  {tokens.map((t, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 text-xs font-mono rounded bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Detokenize Tab */}
        {tab === "detokenize" && (
          <div className="space-y-4">
            <div>
              <label className="text-xs text-[var(--color-text-secondary)] mb-1 block">
                Token IDs (comma or space separated)
              </label>
              <textarea
                value={tokenIds}
                onChange={(e) => setTokenIds(e.target.value)}
                placeholder="e.g. 1234, 5678, 9012"
                className="w-full h-32 p-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm font-mono text-[var(--color-text-primary)] resize-none outline-none"
              />
            </div>
            <button
              onClick={handleDetokenize}
              disabled={!tokenIds.trim()}
              className="px-4 py-2 text-sm rounded-lg bg-[var(--color-accent)] text-white hover:opacity-90 disabled:opacity-40 flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Detokenize
            </button>
            {detokenizeError && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-[var(--color-danger)] text-sm">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {detokenizeError}
              </div>
            )}
            {decodedText && (
              <div className="border border-[var(--color-border)] rounded-xl p-4 bg-[var(--color-bg-primary)]">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium flex items-center gap-1.5">
                    <Type className="w-3.5 h-3.5" />
                    Decoded Text
                  </span>
                  <button
                    onClick={() => navigator.clipboard.writeText(decodedText)}
                    className="p-1.5 rounded hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                  >
                    <Copy className="w-3.5 h-3.5" />
                  </button>
                </div>
                <pre className="text-sm font-mono whitespace-pre-wrap break-words text-[var(--color-text-primary)]">
                  {decodedText}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Count Tab */}
        {tab === "count" && (
          <div className="space-y-4">
            <div>
              <label className="text-xs text-[var(--color-text-secondary)] mb-1 block">
                Prompt to count
              </label>
              <textarea
                value={countText}
                onChange={(e) => setCountText(e.target.value)}
                placeholder="Enter prompt text..."
                className="w-full h-32 p-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm font-mono text-[var(--color-text-primary)] resize-none outline-none"
              />
            </div>
            <div className="flex items-center gap-3">
              <div>
                <label className="text-xs text-[var(--color-text-secondary)]">
                  Max tokens (0 = no limit)
                </label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value) || 0)}
                  className="ml-2 w-28 px-3 py-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm text-[var(--color-text-primary)] outline-none"
                />
              </div>
              <button
                onClick={handleCount}
                disabled={!countText.trim()}
                className="px-4 py-2 text-sm rounded-lg bg-[var(--color-accent)] text-white hover:opacity-90 disabled:opacity-40 flex items-center gap-1.5"
              >
                <Hash className="w-3.5 h-3.5" />
                Count
              </button>
            </div>
            {countError && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-[var(--color-danger)] text-sm">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {countError}
              </div>
            )}
            {tokenCount !== null && (
              <div
                className={`border rounded-xl p-4 ${
                  overLimit
                    ? "border-[var(--color-danger)] bg-red-500/5"
                    : "border-[var(--color-border)] bg-[var(--color-bg-primary)]"
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="text-3xl font-bold text-[var(--color-text-primary)]">
                    {tokenCount}
                  </span>
                  <span className="text-sm text-[var(--color-text-secondary)]">
                    tokens
                  </span>
                  {overLimit && (
                    <span className="text-xs px-2 py-0.5 rounded bg-[var(--color-danger)] text-white">
                      Over limit ({maxTokens})
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
