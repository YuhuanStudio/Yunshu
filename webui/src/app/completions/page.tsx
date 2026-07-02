"use client";

import { useEffect, useRef, useState } from "react";
import {
  Button,
  NumberInput,
  Textarea,
  Slider,
  Switch,
  Card,
  Badge,
  Alert,
  Spinner,
  InlineStatus,
  Separator,
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
  cn,
  toast,
} from "yunui";
import { SettingRow } from "yunui/patterns";
import { Play, Square, Copy, RefreshCw } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, streamSSE, ApiError } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import type { Model, CompletionResult } from "@/lib/types";

/** Flat logprobs payload as returned by /v1/completions (OpenAI legacy shape). */
interface FlatLogprobs {
  tokens: string[];
  token_logprobs: number[];
  top_logprobs?: Record<string, number>[];
  text_offset?: number[];
}

/** One rendered completion choice (streaming produces exactly one). */
interface Choice {
  text: string;
  finishReason: string | null;
  logprobs: FlatLogprobs | null;
}

/** A labeled slider row with a live numeric readout, for the settings panel. */
function SliderRow({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono tabular-nums">{value}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={(v) => onChange(v[0] ?? value)} />
    </div>
  );
}

/** A labeled numeric field row for the settings panel. */
function NumberRow({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <div className="w-28 shrink-0">
        <NumberInput value={value} onChange={onChange} min={min} max={max} step={step} />
      </div>
    </div>
  );
}

/** Compact per-token logprob strip: token chips shaded by confidence, with a
 *  tooltip listing the top-k alternatives. */
function LogprobsView({ lp }: { lp: FlatLogprobs }) {
  return (
    <TooltipProvider delayDuration={100}>
      <div className="flex flex-wrap gap-0.5 rounded-lg bg-muted/40 p-3 font-mono text-xs leading-relaxed">
        {lp.tokens.map((tok, i) => {
        const logp = lp.token_logprobs[i];
        const prob = typeof logp === "number" ? Math.exp(logp) : 1;
        // green (confident) → amber (uncertain).
        const hue = Math.round(prob * 120);
        const top = lp.top_logprobs?.[i];
        return (
          <Tooltip key={i}>
            <TooltipTrigger asChild>
              <span
                className="cursor-help whitespace-pre rounded px-1 py-0.5"
                style={{ backgroundColor: `hsl(${hue} 70% 50% / 0.18)` }}
              >
                {tok.replace(/\n/g, "⏎")}
              </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <div className="space-y-1 font-mono text-xs">
                <div className="text-muted-foreground">
                  logprob {typeof logp === "number" ? logp.toFixed(3) : "—"} · p{" "}
                  {(prob * 100).toFixed(1)}%
                </div>
                {top &&
                  Object.entries(top)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([alt, l]) => (
                      <div key={alt} className="flex justify-between gap-4">
                        <span className="truncate">{JSON.stringify(alt)}</span>
                        <span className="text-muted-foreground">{l.toFixed(2)}</span>
                      </div>
                    ))}
              </div>
            </TooltipContent>
          </Tooltip>
        );
      })}
      </div>
    </TooltipProvider>
  );
}

export default function CompletionsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState<string>("");
  const [prompt, setPrompt] = useState("");

  // Sampling params.
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(128);
  const [topP, setTopP] = useState(1);
  const [topK, setTopK] = useState(0);
  const [stop, setStop] = useState("");
  const [n, setN] = useState(1);

  // Legacy toggles.
  const [stream, setStream] = useState(true);
  const [echo, setEcho] = useState(false);
  const [logprobs, setLogprobs] = useState(0);
  const [seed, setSeed] = useState<number | null>(null);

  // Run state.
  const [choices, setChoices] = useState<Choice[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usage, setUsage] = useState<CompletionResult["usage"] | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const loadModels = () =>
    api.get<{ data: Model[] }>("/v1/models").then((r) => {
      setModels(r.data);
      setModel((m) => m || r.data[0]?.id || "");
    });

  useEffect(() => {
    let cancelled = false;
    api
      .get<{ data: Model[] }>("/v1/models")
      .then((r) => {
        if (cancelled) return;
        setModels(r.data);
        setModel((m) => m || r.data[0]?.id || "");
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  const stopRun = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
  };

  const generate = async () => {
    if (!model || !prompt.trim() || running) return;

    setRunning(true);
    setError(null);
    setChoices([]);
    setUsage(null);

    const stopSeqs = stop
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);

    const body: Record<string, unknown> = {
      model,
      prompt,
      max_tokens: maxTokens,
      temperature,
      top_p: topP,
      top_k: topK,
      stream,
      echo,
      n,
    };
    if (stopSeqs.length) body.stop = stopSeqs;
    if (logprobs > 0) body.logprobs = logprobs;
    if (seed !== null) body.seed = seed;
    if (stream) body.stream_options = { include_usage: true };

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      if (stream) {
        // Streaming yields a single, incrementally-built choice.
        setChoices([{ text: "", finishReason: null, logprobs: null }]);
        await streamSSE(
          "/v1/completions",
          body,
          (data) => {
            const choice = (data.choices as { text?: string; finish_reason?: string | null }[] | undefined)?.[0];
            if (choice?.text) {
              setChoices((prev) => {
                const first = prev[0] ?? { text: "", finishReason: null, logprobs: null };
                return [{ ...first, text: first.text + choice.text }, ...prev.slice(1)];
              });
            }
            if (choice?.finish_reason) {
              setChoices((prev) => (prev[0] ? [{ ...prev[0], finishReason: choice.finish_reason! }, ...prev.slice(1)] : prev));
            }
            const u = data.usage as CompletionResult["usage"] | undefined;
            if (u && typeof u.completion_tokens === "number") setUsage(u);
          },
          controller.signal,
        );
      } else {
        const res = await api.post<CompletionResult>("/v1/completions", body, controller.signal);
        setChoices(
          (res.choices ?? []).map((c) => ({
            text: c.text ?? "",
            finishReason: c.finish_reason ?? null,
            logprobs: (c.logprobs as FlatLogprobs | null) ?? null,
          })),
        );
        setUsage(res.usage ?? null);
      }
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setRunning(false);
    }
  };

  const copyOutput = async (text: string) => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      toast.success("Copied to clipboard");
    } catch {
      toast.success("Copy failed");
    }
  };

  const hasOutput = choices.some((c) => c.text);

  return (
    <PageShell
      title="Completions"
      description="OpenAI-compatible legacy text completion — raw prompt in, tokens out."
      width="wide"
    >
      <div className="flex flex-col gap-4 lg:flex-row">
        {/* Main: prompt + output */}
        <div className="min-w-0 flex-1 space-y-4">
          {error && (
            <Alert variant="error" title="Request failed">
              {error}
            </Alert>
          )}

          <Card className="p-5">
            <div className="mb-3 flex items-center justify-between gap-3">
              <span className="text-sm font-medium">Prompt</span>
              <span className="text-xs text-muted-foreground tabular-nums">{fmtNumber(prompt.length)} chars</span>
            </div>
            <Textarea
              rows={8}
              placeholder="Once upon a time,"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="font-mono text-sm"
            />
            <div className="mt-3 flex items-center gap-3">
              {running ? (
                <Button variant="destructive" onClick={stopRun}>
                  <Square className="h-4 w-4" /> Stop
                </Button>
              ) : (
                <Button onClick={generate} disabled={!model || !prompt.trim()}>
                  <Play className="h-4 w-4" /> {stream ? "Stream" : "Generate"}
                </Button>
              )}
              {running && stream && <InlineStatus status="processing" label="Streaming…" />}
              {usage && (
                <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
                  <Badge variant="default">{fmtNumber(usage.prompt_tokens)} prompt</Badge>
                  <Badge variant="info">{fmtNumber(usage.completion_tokens)} completion</Badge>
                  <Badge variant="default">{fmtNumber(usage.total_tokens)} total</Badge>
                </div>
              )}
            </div>
          </Card>

          {/* Output — one card per choice. */}
          {choices.length === 0 && !running ? (
            <Card className="p-5">
              <p className="py-10 text-center text-sm text-muted-foreground">Generated text will appear here.</p>
            </Card>
          ) : choices.length === 0 && running ? (
            <Card className="p-5">
              <div className="flex justify-center py-10">
                <Spinner />
              </div>
            </Card>
          ) : (
            choices.map((c, i) => (
              <Card key={i} className="p-5">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <span className="text-sm font-medium">{choices.length > 1 ? `Choice ${i + 1}` : "Output"}</span>
                  <div className="flex items-center gap-2">
                    {c.finishReason && (
                      <Badge variant={c.finishReason === "stop" ? "success" : "warning"}>{c.finishReason}</Badge>
                    )}
                    <Button aria-label="Copy output" variant="ghost" size="icon" onClick={() => copyOutput(c.text)} disabled={!c.text}>
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                {c.text ? (
                  <pre className="whitespace-pre-wrap break-words rounded-lg bg-muted/40 p-4 font-mono text-sm">
                    {c.text}
                  </pre>
                ) : (
                  <div className="flex justify-center py-6">
                    <Spinner />
                  </div>
                )}
                {c.logprobs && c.logprobs.tokens?.length > 0 && (
                  <div className="mt-3">
                    <p className="mb-1.5 text-xs font-medium text-muted-foreground">
                      Token logprobs (hover for top {logprobs} alternatives)
                    </p>
                    <LogprobsView lp={c.logprobs} />
                  </div>
                )}
              </Card>
            ))
          )}
        </div>

        {/* Right: settings panel */}
        <aside className="w-full shrink-0 space-y-4 lg:w-72">
          <Card className="p-5">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium">Model</span>
              <Button aria-label="Refresh models" variant="ghost" size="icon" onClick={() => loadModels().catch(() => {})}>
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
            <ModelPicker models={models} value={model} onChange={setModel} />
          </Card>

          <Card className="space-y-4 p-5">
            <span className="text-sm font-medium">Sampling</span>
            <SliderRow label="Temperature" value={temperature} onChange={setTemperature} min={0} max={2} step={0.05} />
            <NumberRow label="Max tokens" value={maxTokens} onChange={setMaxTokens} min={1} max={8192} step={1} />
            <SliderRow label="Top P" value={topP} onChange={setTopP} min={0} max={1} step={0.01} />
            <NumberRow label="Top K" value={topK} onChange={setTopK} min={0} step={1} />
            <NumberRow label="Choices (n)" value={n} onChange={setN} min={1} max={8} step={1} />
            <div className="space-y-1.5">
              <span className="text-sm text-muted-foreground">Stop sequences</span>
              <Textarea
                rows={2}
                value={stop}
                onChange={(e) => setStop(e.target.value)}
                placeholder="One per line…"
                className="font-mono text-xs"
              />
            </div>
          </Card>

          <Card className="space-y-4 p-5">
            <span className="text-sm font-medium">Legacy options</span>
            <SettingRow
              title="Stream"
              description="Server-sent token deltas"
              control={<Switch checked={stream} onCheckedChange={setStream} />}
            />
            <SettingRow
              title="Echo"
              description="Include the prompt in output"
              control={<Switch checked={echo} onCheckedChange={setEcho} />}
            />

            <Separator />

            <div className="space-y-1.5">
              <SliderRow label="Logprobs (0 = off)" value={logprobs} onChange={(v) => setLogprobs(Math.round(v))} min={0} max={5} step={1} />
              {stream && logprobs > 0 && (
                <p className="text-xs text-muted-foreground">Logprobs are returned on non-streamed requests.</p>
              )}
            </div>

            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="text-muted-foreground">Seed</span>
              <div className="flex items-center gap-1.5">
                <div className="w-32">
                  <NumberInput value={seed ?? 0} onChange={(v) => setSeed(v)} step={1} />
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className={cn(seed === null && "opacity-40")}
                  onClick={() => setSeed(null)}
                  disabled={seed === null}
                >
                  Clear
                </Button>
              </div>
            </div>
            {seed === null && <p className="-mt-2 text-xs text-muted-foreground">Random seed each run.</p>}
          </Card>

          {!hasOutput && running && stream && (
            <p className="text-center text-xs text-muted-foreground">Receiving tokens…</p>
          )}
        </aside>
      </div>
    </PageShell>
  );
}
