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
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
  Separator,
  cn,
  toast,
} from "yunui";
import { SettingRow } from "yunui/patterns";
import { Play, Square, Copy, ChevronDown, RefreshCw } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, streamSSE, ApiError } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import type { Model, CompletionResult } from "@/lib/types";

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
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => onChange(v[0] ?? value)}
      />
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

export default function CompletionsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState<string>("");
  const [prompt, setPrompt] = useState("");

  // Generation params.
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(128);
  const [topP, setTopP] = useState(1);
  const [topK, setTopK] = useState(0);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0);
  const [presencePenalty, setPresencePenalty] = useState(0);
  const [seed, setSeed] = useState<number | null>(null);

  // Toggles.
  const [stream, setStream] = useState(true);
  const [echo, setEcho] = useState(false);
  const [specDecode, setSpecDecode] = useState(false);
  const [enableThinking, setEnableThinking] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Run state.
  const [output, setOutput] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [finishReason, setFinishReason] = useState<string | null>(null);
  const [completionTokens, setCompletionTokens] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const loadModels = () =>
    api
      .get<{ data: Model[] }>("/v1/models")
      .then((r) => {
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

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
  };

  const generate = async () => {
    if (!model || !prompt.trim() || running) return;

    setRunning(true);
    setError(null);
    setOutput("");
    setFinishReason(null);
    setCompletionTokens(null);

    const body: Record<string, unknown> = {
      model,
      prompt,
      max_tokens: maxTokens,
      temperature,
      top_p: topP,
      top_k: topK,
      repetition_penalty: repetitionPenalty,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
      stream,
      echo,
      spec_decode: specDecode,
      enable_thinking: enableThinking,
    };
    if (seed !== null) body.seed = seed;

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      if (stream) {
        await streamSSE(
          "/v1/completions",
          body,
          (data) => {
            const choices = data.choices as
              | { text?: string; finish_reason?: string | null }[]
              | undefined;
            const choice = choices?.[0];
            if (choice?.text) setOutput((o) => o + choice.text);
            if (choice?.finish_reason) setFinishReason(choice.finish_reason);
            const usage = data.usage as { completion_tokens?: number } | undefined;
            if (typeof usage?.completion_tokens === "number") {
              setCompletionTokens(usage.completion_tokens);
            }
          },
          controller.signal,
        );
      } else {
        const res = await api.post<CompletionResult>("/v1/completions", body, controller.signal);
        setOutput(res.choices[0]?.text ?? "");
        setFinishReason(res.choices[0]?.finish_reason ?? null);
        setCompletionTokens(res.usage?.completion_tokens ?? null);
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

  const copyOutput = async () => {
    if (!output) return;
    try {
      await navigator.clipboard.writeText(output);
      toast.success("Copied to clipboard");
    } catch {
      toast.success("Copy failed");
    }
  };

  return (
    <PageShell
      title="Completions"
      description="OpenAI-compatible text completion against the loaded model."
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
              <span className="text-xs text-muted-foreground tabular-nums">
                {fmtNumber(prompt.length)} chars
              </span>
            </div>
            <Textarea
              rows={8}
              placeholder="Write a haiku about tensors…"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="font-mono text-sm"
            />
            <div className="mt-3 flex items-center gap-3">
              {running ? (
                <Button variant="destructive" onClick={stop}>
                  <Square className="h-4 w-4" /> Stop
                </Button>
              ) : (
                <Button onClick={generate} disabled={!model || !prompt.trim()}>
                  <Play className="h-4 w-4" /> Generate
                </Button>
              )}
              {running && stream && <InlineStatus status="processing" label="Streaming…" />}
            </div>
          </Card>

          <Card className="p-5">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <span className="text-sm font-medium">Output</span>
              <div className="flex items-center gap-2">
                {finishReason && (
                  <Badge variant={finishReason === "stop" ? "success" : "warning"}>
                    {finishReason}
                  </Badge>
                )}
                {completionTokens !== null && (
                  <Badge variant="info">{fmtNumber(completionTokens)} tokens</Badge>
                )}
                <Button
                  aria-label="Copy output"
                  variant="ghost"
                  size="icon"
                  onClick={copyOutput}
                  disabled={!output}
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
            </div>
            {output ? (
              <pre className="whitespace-pre-wrap break-words rounded-lg bg-muted/40 p-4 font-mono text-sm">
                {output}
              </pre>
            ) : running ? (
              <div className="flex justify-center py-10">
                <Spinner />
              </div>
            ) : (
              <p className="py-10 text-center text-sm text-muted-foreground">
                Generated text will appear here.
              </p>
            )}
          </Card>
        </div>

        {/* Right: settings panel */}
        <aside className="w-full shrink-0 space-y-4 lg:w-72">
          <Card className="p-5">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium">Model</span>
              <Button
                aria-label="Refresh models"
                variant="ghost"
                size="icon"
                onClick={() => loadModels().catch(() => {})}
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
            <ModelPicker models={models} value={model} onChange={setModel} />
          </Card>

          <Card className="space-y-4 p-5">
            <span className="text-sm font-medium">Parameters</span>
            <SliderRow
              label="Temperature"
              value={temperature}
              onChange={setTemperature}
              min={0}
              max={2}
              step={0.05}
            />
            <NumberRow
              label="Max tokens"
              value={maxTokens}
              onChange={setMaxTokens}
              min={1}
              max={8192}
              step={1}
            />
            <SliderRow label="Top P" value={topP} onChange={setTopP} min={0} max={1} step={0.01} />

            <Separator />

            <SettingRow
              title="Stream"
              description="Server-sent token deltas"
              control={<Switch checked={stream} onCheckedChange={setStream} />}
            />
            <SettingRow
              title="Enable thinking"
              description="Reasoning traces"
              control={<Switch checked={enableThinking} onCheckedChange={setEnableThinking} />}
            />

            <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
              <CollapsibleTrigger className="flex w-full items-center justify-between text-sm font-medium">
                <span>Advanced</span>
                <ChevronDown
                  className={cn("h-4 w-4 transition-transform", advancedOpen && "rotate-180")}
                />
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="space-y-4 pt-4">
                  <NumberRow label="Top K" value={topK} onChange={setTopK} min={0} step={1} />
                  <SliderRow
                    label="Repetition penalty"
                    value={repetitionPenalty}
                    onChange={setRepetitionPenalty}
                    min={0}
                    max={2}
                    step={0.01}
                  />
                  <SliderRow
                    label="Frequency penalty"
                    value={frequencyPenalty}
                    onChange={setFrequencyPenalty}
                    min={-2}
                    max={2}
                    step={0.01}
                  />
                  <SliderRow
                    label="Presence penalty"
                    value={presencePenalty}
                    onChange={setPresencePenalty}
                    min={-2}
                    max={2}
                    step={0.01}
                  />
                  <NumberRow label="Seed" value={seed ?? 0} onChange={(v) => setSeed(v)} step={1} />
                  <div className="flex justify-end">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSeed(null)}
                      disabled={seed === null}
                    >
                      Clear seed
                    </Button>
                  </div>

                  <Separator />

                  <SettingRow
                    title="Echo"
                    description="Include the prompt in output"
                    control={<Switch checked={echo} onCheckedChange={setEcho} />}
                  />
                  <SettingRow
                    title="Speculative decoding"
                    description="Draft-model acceleration"
                    control={<Switch checked={specDecode} onCheckedChange={setSpecDecode} />}
                  />
                </div>
              </CollapsibleContent>
            </Collapsible>
          </Card>
        </aside>
      </div>
    </PageShell>
  );
}
