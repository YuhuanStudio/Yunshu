"use client";

import { useState, useRef, useCallback } from "react";
import {
  Send,
  Copy,
  Trash2,
  Settings,
  ChevronDown,
  ChevronUp,
  FileText,
} from "lucide-react";

const API_BASE =
  typeof window !== "undefined"
    ? window.location.origin
    : "http://localhost:8000";

interface CompletionResult {
  id: string;
  choices: {
    text: string;
    index: number;
    finish_reason: string | null;
    logprobs: any | null;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export default function CompletionsPage() {
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("");
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1.0);
  const [topK, setTopK] = useState(0);
  const [minP, setMinP] = useState(0.0);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.0);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0.0);
  const [presencePenalty, setPresencePenalty] = useState(0.0);
  const [seed, setSeed] = useState<string>("");
  const [stop, setStop] = useState<string>("");
  const [specDecode, setSpecDecode] = useState(false);
  const [enableThinking, setEnableThinking] = useState(false);
  const [thinkingBudget, setThinkingBudget] = useState<string>("");
  const [echo, setEcho] = useState(false);
  const [streaming, setStreaming] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [results, setResults] = useState<CompletionResult[]>([]);
  const [streamingText, setStreamingText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const [models, setModels] = useState<string[]>([]);

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

  const handleGenerate = async () => {
    if (!prompt.trim() || isGenerating) return;
    setIsGenerating(true);
    setStreamingText("");

    const stopArr = stop
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);

    const body: any = {
      model: model || "default",
      prompt,
      max_tokens: maxTokens,
      temperature,
      top_p: topP,
      top_k: topK,
      min_p: minP,
      repetition_penalty: repetitionPenalty,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
      stream: streaming,
      echo,
      spec_decode: specDecode,
      enable_thinking: enableThinking,
    };
    if (seed) body.seed = parseInt(seed);
    if (stopArr.length > 0) body.stop = stopArr;
    if (thinkingBudget) body.thinking_budget = parseInt(thinkingBudget);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_BASE}/v1/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        const err = await res.text();
        setStreamingText(`Error: ${err}`);
        setIsGenerating(false);
        return;
      }

      if (streaming && res.body) {
        let accumulated = "";
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let resultId = "";
        let usage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") continue;
            try {
              const parsed = JSON.parse(data);
              resultId = parsed.id || resultId;
              if (parsed.usage) usage = parsed.usage;
              const text = parsed.choices?.[0]?.text || "";
              accumulated += text;
              setStreamingText(accumulated);
            } catch {}
          }
        }

        setResults((prev) => [
          {
            id: resultId || `cmpl-${Date.now()}`,
            choices: [
              { text: accumulated, index: 0, finish_reason: "stop", logprobs: null },
            ],
            usage,
          },
          ...prev,
        ]);
        setStreamingText("");
      } else {
        const data: CompletionResult = await res.json();
        setResults((prev) => [data, ...prev]);
      }
    } catch (e: any) {
      if (e.name !== "AbortError") {
        setStreamingText(`Error: ${e.message}`);
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setIsGenerating(false);
  };

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[var(--color-border)]">
        <h1 className="text-xl font-semibold flex items-center gap-2">
          <FileText className="w-5 h-5 text-[var(--color-accent)]" />
          Completions
        </h1>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          OpenAI-compatible text completion endpoint
        </p>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: Prompt & Results */}
        <div className="flex-1 flex flex-col overflow-hidden p-4 gap-4">
          {/* Prompt input */}
          <div className="border border-[var(--color-border)] rounded-xl overflow-hidden">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt here..."
              className="w-full h-40 p-4 bg-[var(--color-bg-primary)] text-[var(--color-text-primary)] resize-none outline-none text-sm font-mono"
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  handleGenerate();
                }
              }}
            />
            <div className="flex items-center justify-between px-3 py-2 bg-[var(--color-bg-secondary)] border-t border-[var(--color-border)]">
              <span className="text-xs text-[var(--color-text-secondary)]">
                {prompt.length} chars
              </span>
              <div className="flex gap-2">
                {isGenerating ? (
                  <button
                    onClick={handleStop}
                    className="px-4 py-1.5 text-sm rounded-lg bg-[var(--color-danger)] text-white hover:opacity-90"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    onClick={handleGenerate}
                    disabled={!prompt.trim()}
                    className="px-4 py-1.5 text-sm rounded-lg bg-[var(--color-accent)] text-white hover:opacity-90 disabled:opacity-40 flex items-center gap-1.5"
                  >
                    <Send className="w-3.5 h-3.5" />
                    Generate
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Streaming output */}
          {streamingText && (
            <div className="border border-[var(--color-accent-muted)] rounded-xl p-4 bg-[var(--color-bg-primary)]">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full bg-[var(--color-accent)] animate-pulse" />
                <span className="text-xs text-[var(--color-accent)] font-medium">
                  Streaming...
                </span>
              </div>
              <pre className="text-sm font-mono whitespace-pre-wrap break-words text-[var(--color-text-primary)]">
                {streamingText}
              </pre>
            </div>
          )}

          {/* Results */}
          {results.length > 0 && (
            <div className="flex-1 overflow-auto space-y-3">
              {results.map((result, i) => (
                <div
                  key={result.id + i}
                  className="border border-[var(--color-border)] rounded-xl p-4 bg-[var(--color-bg-primary)]"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-mono text-[var(--color-text-secondary)]">
                        {result.id}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]">
                        {result.usage?.prompt_tokens || 0}p +{" "}
                        {result.usage?.completion_tokens || 0}c tokens
                      </span>
                    </div>
                    <button
                      onClick={() =>
                        navigator.clipboard.writeText(
                          result.choices[0]?.text || ""
                        )
                      }
                      className="p-1.5 rounded hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                    >
                      <Copy className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <pre className="text-sm font-mono whitespace-pre-wrap break-words text-[var(--color-text-primary)]">
                    {result.choices[0]?.text}
                  </pre>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Settings Panel */}
        <div className="w-72 border-l border-[var(--color-border)] bg-[var(--color-bg-secondary)] overflow-auto p-4 space-y-4">
          {/* Model Selection */}
          <div>
            <label className="text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">
              Model
            </label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              onClick={loadModels}
              className="mt-1 w-full px-3 py-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm text-[var(--color-text-primary)] outline-none"
            >
              {models.length === 0 && (
                <option value="">Loading...</option>
              )}
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* Core Parameters */}
          <div className="space-y-3">
            <ParamSlider
              label="Temperature"
              value={temperature}
              min={0}
              max={2}
              step={0.1}
              onChange={setTemperature}
            />
            <ParamSlider
              label="Max Tokens"
              value={maxTokens}
              min={1}
              max={131072}
              step={1}
              onChange={setMaxTokens}
            />
            <ParamSlider
              label="Top P"
              value={topP}
              min={0}
              max={1}
              step={0.05}
              onChange={setTopP}
            />
          </div>

          {/* Toggles */}
          <div className="space-y-2">
            <Toggle label="Streaming" value={streaming} onChange={setStreaming} />
            <Toggle label="Echo" value={echo} onChange={setEcho} />
            <Toggle label="Spec Decode" value={specDecode} onChange={setSpecDecode} />
            <Toggle label="Thinking" value={enableThinking} onChange={setEnableThinking} />
          </div>

          {/* Advanced */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1.5 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] w-full"
          >
            <Settings className="w-3.5 h-3.5" />
            Advanced
            {showAdvanced ? (
              <ChevronUp className="w-3.5 h-3.5 ml-auto" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5 ml-auto" />
            )}
          </button>

          {showAdvanced && (
            <div className="space-y-3">
              <ParamSlider
                label="Top K"
                value={topK}
                min={0}
                max={200}
                step={1}
                onChange={setTopK}
              />
              <ParamSlider
                label="Min P"
                value={minP}
                min={0}
                max={1}
                step={0.01}
                onChange={setMinP}
              />
              <ParamSlider
                label="Repetition Penalty"
                value={repetitionPenalty}
                min={1}
                max={2}
                step={0.05}
                onChange={setRepetitionPenalty}
              />
              <ParamSlider
                label="Frequency Penalty"
                value={frequencyPenalty}
                min={-2}
                max={2}
                step={0.1}
                onChange={setFrequencyPenalty}
              />
              <ParamSlider
                label="Presence Penalty"
                value={presencePenalty}
                min={-2}
                max={2}
                step={0.1}
                onChange={setPresencePenalty}
              />
              <div>
                <label className="text-xs text-[var(--color-text-secondary)]">
                  Seed
                </label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  placeholder="Random"
                  className="mt-1 w-full px-3 py-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm text-[var(--color-text-primary)] outline-none"
                />
              </div>
              <div>
                <label className="text-xs text-[var(--color-text-secondary)]">
                  Stop (comma-separated)
                </label>
                <input
                  type="text"
                  value={stop}
                  onChange={(e) => setStop(e.target.value)}
                  placeholder="e.g. \n, ###"
                  className="mt-1 w-full px-3 py-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm text-[var(--color-text-primary)] outline-none"
                />
              </div>
              {enableThinking && (
                <div>
                  <label className="text-xs text-[var(--color-text-secondary)]">
                    Thinking Budget
                  </label>
                  <input
                    type="number"
                    value={thinkingBudget}
                    onChange={(e) => setThinkingBudget(e.target.value)}
                    placeholder="Unlimited"
                    className="mt-1 w-full px-3 py-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] text-sm text-[var(--color-text-primary)] outline-none"
                  />
                </div>
              )}
            </div>
          )}

          {/* Clear results */}
          {results.length > 0 && (
            <button
              onClick={() => setResults([])}
              className="w-full px-3 py-2 text-sm rounded-lg border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] flex items-center justify-center gap-1.5"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear Results
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <label className="text-xs text-[var(--color-text-secondary)]">
          {label}
        </label>
        <span className="text-xs font-mono text-[var(--color-text-primary)]">
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 mt-1 rounded-full appearance-none bg-[var(--color-bg-tertiary)] accent-[var(--color-accent)]"
      />
    </div>
  );
}

function Toggle({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between cursor-pointer">
      <span className="text-xs text-[var(--color-text-secondary)]">
        {label}
      </span>
      <div
        onClick={() => onChange(!value)}
        className={`w-9 h-5 rounded-full relative transition-colors ${
          value
            ? "bg-[var(--color-accent)]"
            : "bg-[var(--color-bg-tertiary)]"
        }`}
      >
        <div
          className={`w-3.5 h-3.5 rounded-full bg-white absolute top-0.5 transition-transform ${
            value ? "translate-x-[18px]" : "translate-x-0.5"
          }`}
        />
      </div>
    </label>
  );
}
