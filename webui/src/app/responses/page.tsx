"use client";

import { useEffect, useRef, useState, useCallback, type ReactNode } from "react";
import {
  Button,
  IconButton,
  Textarea,
  Input,
  Card,
  Badge,
  Switch,
  Slider,
  Spinner,
  EmptyState,
  CustomSelect,
  Alert,
  ThemeToggle,
  cn,
  toast,
} from "yunui";
import { ShellChrome } from "@/components/app-shell";
import { ThinkingBlock } from "yunui/ai";
import { ChatComposer } from "yunui/chat";
import {
  Plus,
  Trash2,
  Settings2,
  Sparkles,
  User,
  Wrench,
  X,
  Copy,
  Check,
} from "lucide-react";
import { ModelPicker } from "@/components/model-picker";
import { api, streamSSE, ApiError, type SSEChunk } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import { Markdown } from "@/components/markdown";
import { PageShell } from "@/components/page-shell";
import type { Model } from "@/lib/types";

// ---- page-local shapes -----------------------------------------------------
interface FnCall {
  name: string;
  arguments: string;
}
type TurnStatus =
  | "streaming"
  | "queued"
  | "in_progress"
  | "completed"
  | "failed"
  | "incomplete"
  | "cancelled";
interface Usage {
  input_tokens?: number;
  output_tokens?: number;
  reasoning_tokens?: number;
}
interface Turn {
  id: string;
  userText: string;
  responseId?: string;
  status: TurnStatus;
  text: string;
  reasoning: string;
  functionCalls: FnCall[];
  usage?: Usage;
  error?: string;
  background?: boolean;
}

const EFFORTS = [
  { value: "minimal", label: "Minimal" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
];

const uid = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

// ---- unknown-safe readers --------------------------------------------------
const asRec = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" ? (v as Record<string, unknown>) : {};
const asStr = (v: unknown): string | undefined => (typeof v === "string" ? v : undefined);

/** Fold a non-streaming `output[]` array into {text, reasoning, functionCalls}. */
function foldOutput(output: unknown): { text: string; reasoning: string; functionCalls: FnCall[] } {
  let text = "";
  let reasoning = "";
  const functionCalls: FnCall[] = [];
  const items = Array.isArray(output) ? (output as unknown[]) : [];
  for (const raw of items) {
    const item = asRec(raw);
    const type = asStr(item.type);
    if (type === "reasoning") {
      // Reasoning is a separate output item; text lives in `summary[].text`
      // (or occasionally a flat `content` / `text`).
      const summary = Array.isArray(item.summary) ? (item.summary as unknown[]) : [];
      for (const s of summary) reasoning += asStr(asRec(s).text) ?? "";
      reasoning += asStr(item.text) ?? "";
    } else if (type === "message") {
      const content = Array.isArray(item.content) ? (item.content as unknown[]) : [];
      for (const c of content) {
        const part = asRec(c);
        if (asStr(part.type) === "output_text") text += asStr(part.text) ?? "";
      }
    } else if (type === "function_call") {
      functionCalls.push({
        name: asStr(item.name) ?? "",
        arguments: asStr(item.arguments) ?? "",
      });
    }
  }
  return { text, reasoning, functionCalls };
}

const TERMINAL: TurnStatus[] = ["completed", "failed", "incomplete", "cancelled"];

export default function ResponsesPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [prevResponseId, setPrevResponseId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // settings
  const [instructions, setInstructions] = useState("");
  const [effort, setEffort] = useState("medium");
  const [temperature, setTemperature] = useState(1);
  const [topP, setTopP] = useState(1);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [store, setStore] = useState(true);
  const [background, setBackground] = useState(false);
  const [enableThinking, setEnableThinking] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const runningIdRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .get<{ data: Model[] }>("/v1/models")
      .then((d) => {
        setModels(d.data ?? []);
        setModel((m) => m || d.data?.find((x) => x.loaded)?.id || d.data?.[0]?.id || "");
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, running]);

  const updateTurn = useCallback((id: string, fn: (t: Turn) => Turn) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? fn(t) : t)));
  }, []);

  const newThread = () => {
    if (running) return;
    setTurns([]);
    setPrevResponseId(null);
    setError(null);
  };

  const stop = async () => {
    abortRef.current?.abort();
    abortRef.current = null;
    const id = runningIdRef.current;
    const turn = turns.find((t) => t.id === id);
    // Ask the server to cancel a stored/background response too.
    if (turn?.responseId) {
      api.post(`/v1/responses/${turn.responseId}/cancel`).catch(() => {});
    }
    if (id) updateTurn(id, (t) => ({ ...t, status: "cancelled" }));
    setRunning(false);
    runningIdRef.current = null;
  };

  const describe = (e: unknown) =>
    e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Something went wrong.";

  // ---- streaming path (store, non-background) ----
  const runStream = async (turnId: string, body: Record<string, unknown>, signal: AbortSignal) => {
    let text = "";
    let reasoning = "";
    const fnMap: Record<string, FnCall> = {};
    let respId: string | undefined;
    let status: TurnStatus = "streaming";
    let usage: Usage | undefined;

    const flush = () =>
      updateTurn(turnId, (t) => ({
        ...t,
        text,
        reasoning,
        responseId: respId ?? t.responseId,
        functionCalls: Object.values(fnMap),
        usage: usage ?? t.usage,
        status,
      }));

    await streamSSE(
      "/v1/responses",
      { ...body, stream: true },
      (d: SSEChunk) => {
        const ev = asStr(d._event) ?? asStr(d.type) ?? "";
        switch (ev) {
          case "response.created": {
            respId = asStr(asRec(d.response).id) ?? respId;
            break;
          }
          case "response.output_text.delta": {
            text += asStr(d.delta) ?? "";
            break;
          }
          case "response.reasoning_summary_text.delta": {
            reasoning += asStr(d.delta) ?? "";
            break;
          }
          case "response.output_item.added": {
            const it = asRec(d.item);
            const itemId = asStr(it.id);
            if (asStr(it.type) === "function_call" && itemId) {
              fnMap[itemId] = { name: asStr(it.name) ?? "", arguments: asStr(it.arguments) ?? "" };
            }
            break;
          }
          case "response.function_call_arguments.delta": {
            const itemId = asStr(d.item_id);
            if (itemId) {
              fnMap[itemId] = fnMap[itemId] ?? { name: "", arguments: "" };
              fnMap[itemId].arguments += asStr(d.delta) ?? "";
            }
            break;
          }
          case "response.function_call_arguments.done": {
            const itemId = asStr(d.item_id);
            if (itemId) {
              fnMap[itemId] = fnMap[itemId] ?? { name: "", arguments: "" };
              const args = asStr(d.arguments);
              if (args !== undefined) fnMap[itemId].arguments = args;
            }
            break;
          }
          case "response.completed":
          case "response.failed":
          case "response.incomplete": {
            const r = asRec(d.response);
            respId = asStr(r.id) ?? respId;
            status = (asStr(r.status) as TurnStatus) ?? (ev.split(".")[1] as TurnStatus);
            usage = asRec(r.usage) as Usage;
            break;
          }
        }
        flush();
      },
      signal,
    );

    return { respId, status };
  };

  // ---- background path: POST (non-stream) then poll GET until terminal ----
  const runBackground = async (
    turnId: string,
    body: Record<string, unknown>,
    signal: AbortSignal,
  ) => {
    const created = await api.post<Record<string, unknown>>(
      "/v1/responses",
      { ...body, stream: false, background: true },
      signal,
    );
    let respId = asStr(created.id);
    let status = (asStr(created.status) as TurnStatus) ?? "queued";
    updateTurn(turnId, (t) => ({ ...t, responseId: respId, status }));

    while (respId && (status === "queued" || status === "in_progress")) {
      await new Promise((r) => setTimeout(r, 1200));
      if (signal.aborted) throw new DOMException("aborted", "AbortError");
      const r = await api.get<Record<string, unknown>>(`/v1/responses/${respId}`, signal);
      status = (asStr(r.status) as TurnStatus) ?? status;
      respId = asStr(r.id) ?? respId;
      const folded = foldOutput(r.output);
      updateTurn(turnId, (t) => ({
        ...t,
        ...folded,
        responseId: respId,
        usage: asRec(r.usage) as Usage,
        status,
      }));
    }
    return { respId, status };
  };

  // ---- plain non-stream path ----
  const runOnce = async (turnId: string, body: Record<string, unknown>, signal: AbortSignal) => {
    const r = await api.post<Record<string, unknown>>(
      "/v1/responses",
      { ...body, stream: false },
      signal,
    );
    const respId = asStr(r.id);
    const status = (asStr(r.status) as TurnStatus) ?? "completed";
    const folded = foldOutput(r.output);
    updateTurn(turnId, (t) => ({
      ...t,
      ...folded,
      responseId: respId,
      usage: asRec(r.usage) as Usage,
      status,
    }));
    return { respId, status };
  };

  const send = async () => {
    const text = input.trim();
    if (!text || running || !model) return;
    setError(null);

    const turn: Turn = {
      id: uid(),
      userText: text,
      status: background ? "queued" : "streaming",
      text: "",
      reasoning: "",
      functionCalls: [],
      background,
    };
    setTurns((prev) => [...prev, turn]);
    setInput("");
    setRunning(true);
    runningIdRef.current = turn.id;

    const body: Record<string, unknown> = {
      model,
      input: text,
      max_output_tokens: maxTokens,
      temperature,
      top_p: topP,
      store,
      reasoning: { effort },
    };
    if (instructions.trim()) body.instructions = instructions.trim();
    if (prevResponseId) body.previous_response_id = prevResponseId;
    if (enableThinking) body.enable_thinking = true;

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let result: { respId?: string; status: TurnStatus };
      if (background) {
        result = await runBackground(turn.id, body, controller.signal);
      } else if (store) {
        result = await runStream(turn.id, body, controller.signal);
      } else {
        // store:false → server keeps no history; still a single-shot response.
        result = await runStream(turn.id, { ...body, store: false }, controller.signal);
      }
      // Thread forward: the server keeps history when store is on.
      if (store && result.respId && result.status === "completed") {
        setPrevResponseId(result.respId);
      }
    } catch (e) {
      if (controller.signal.aborted) {
        updateTurn(turn.id, (t) => ({ ...t, status: "cancelled" }));
      } else {
        const msg = describe(e);
        setError(msg);
        updateTurn(turn.id, (t) => ({ ...t, status: "failed", error: msg }));
        toast.error("Response failed", msg);
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
      runningIdRef.current = null;
    }
  };

  const deleteResponse = async (turn: Turn) => {
    if (turn.responseId && turn.background !== undefined && store) {
      try {
        await api.delete(`/v1/responses/${turn.responseId}`);
      } catch {
        /* already gone / not stored */
      }
    }
    setTurns((prev) => prev.filter((t) => t.id !== turn.id));
    // If we deleted the response the thread was pinned to, drop the pointer.
    if (prevResponseId && turn.responseId === prevResponseId) setPrevResponseId(null);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2.5">
        <ShellChrome />
        {/* flex-1 on mobile so the picker shares the row with the nav chrome
            instead of w-full forcing it onto its own line. */}
        <ModelPicker models={models} value={model} onChange={setModel} className="min-w-0 flex-1 sm:flex-none sm:w-64" />
        <Badge variant={prevResponseId ? "success" : "default"} className="font-mono text-xs">
          {prevResponseId ? `thread → ${prevResponseId.slice(0, 14)}…` : "new thread"}
        </Badge>
        <div className="ml-auto flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={newThread} disabled={running}>
            <Plus className="h-4 w-4" /> New thread
          </Button>
          <IconButton
            icon={<Settings2 className="h-4 w-4" />}
            label="Settings"
            onClick={() => setShowSettings((s) => !s)}
          />
          <ThemeToggle variant="pill" />
        </div>
      </div>

      {/* Settings drawer */}
      {showSettings && (
        <div className="space-y-4 border-b border-border bg-muted/30 px-4 py-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium">Instructions (system)</label>
            <Textarea aria-label="Instructions (system)"
              rows={2}
              placeholder="Optional system instructions applied to the whole thread…"
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
            />
          </div>
          <div className="grid grid-cols-2 gap-x-8 gap-y-4 lg:grid-cols-3">
            <div className="space-y-1.5">
              <label className="text-xs font-medium">Reasoning effort</label>
              <CustomSelect options={EFFORTS} value={effort} onChange={setEffort} />
            </div>
            <SettingSlider
              label="Temperature"
              value={temperature}
              min={0}
              max={2}
              step={0.05}
              onChange={setTemperature}
            />
            <SettingSlider
              label="Top-p"
              value={topP}
              min={0}
              max={1}
              step={0.01}
              onChange={setTopP}
            />
            <SettingSlider
              label="Max output tokens"
              value={maxTokens}
              min={64}
              max={8192}
              step={64}
              onChange={setMaxTokens}
              format={fmtNumber}
            />
            <div className="flex flex-col justify-center gap-2">
              <SettingSwitch label="Store (stateful history)" checked={store} onChange={setStore} />
              <SettingSwitch label="Background (poll)" checked={background} onChange={setBackground} />
              <SettingSwitch label="Thinking" checked={enableThinking} onChange={setEnableThinking} />
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="px-4 pt-3">
          <Alert
            variant="error"
            title="Request failed"
            icon={
              <IconButton
                icon={<X className="h-4 w-4" />}
                label="Dismiss"
                onClick={() => setError(null)}
                className="p-0 text-current opacity-70 transition-opacity hover:bg-transparent hover:opacity-100"
              />
            }
          >
            {error}
          </Alert>
        </div>
      )}

      {/* Transcript */}
      <div ref={scrollRef} className="flex-1 overflow-auto px-4 py-6">
        <div className="mx-auto max-w-3xl space-y-6">
          {turns.length === 0 ? (
            <div className="pt-20">
              <EmptyState
                icon={<Sparkles className="h-6 w-6" />}
                title="Stateful Responses console"
                description="Each turn sends only your new message plus previous_response_id — the server keeps the thread (store:true). Reasoning, text and function calls all stream in."
              />
            </div>
          ) : (
            turns.map((t) => <TurnView key={t.id} turn={t} onDelete={deleteResponse} />)
          )}
        </div>
      </div>

      {/* Composer */}
      <div className="border-t border-border p-4">
        <div className="mx-auto max-w-3xl space-y-2">
          <ChatComposer
            value={input}
            onChange={setInput}
            onSend={send}
            onStop={stop}
            loading={running}
            sendDisabled={!model}
            placeholder="Continue the thread…  (Enter to send, Shift+Enter for newline)"
          />
        </div>
      </div>
    </div>
  );
}

function TurnView({ turn, onDelete }: { turn: Turn; onDelete: (t: Turn) => void }) {
  const [copied, setCopied] = useState(false);
  const streaming = turn.status === "streaming" || turn.status === "in_progress" || turn.status === "queued";

  const copyId = async () => {
    if (!turn.responseId) return;
    try {
      await navigator.clipboard.writeText(turn.responseId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* ignore */
    }
  };

  const statusVariant =
    turn.status === "completed"
      ? "success"
      : turn.status === "failed"
        ? "error"
        : turn.status === "cancelled" || turn.status === "incomplete"
          ? "warning"
          : "info";

  return (
    <div className="space-y-3">
      {/* user */}
      <div className="flex flex-row-reverse gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent/10 text-accent">
          <User className="h-4 w-4" />
        </div>
        <div className="flex min-w-0 flex-1 flex-col items-end">
          <Card className="w-fit max-w-full bg-accent/10 p-3.5">
            <p className="whitespace-pre-wrap break-words text-sm">{turn.userText}</p>
          </Card>
        </div>
      </div>

      {/* assistant */}
      <div className="flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
          <Sparkles className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1 space-y-2">
          {turn.reasoning ? (
            <ThinkingBlock
              content={turn.reasoning}
              isStreaming={streaming}
              renderContent={(c) => <Markdown>{c}</Markdown>}
            />
          ) : null}

          {turn.functionCalls.map((fc, i) => (
            <Card key={i} className="p-3">
              <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <Wrench className="h-3.5 w-3.5" /> {fc.name || "function_call"}
              </div>
              <pre className="overflow-auto rounded bg-muted p-2 text-xs">{fc.arguments || "{}"}</pre>
            </Card>
          ))}

          {(turn.text || streaming) && (
            <Card className="w-fit max-w-full p-3.5">
              <Markdown>{turn.text || "…"}</Markdown>
            </Card>
          )}

          {turn.error ? <p className="text-xs text-error">{turn.error}</p> : null}

          {/* meta row */}
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant={statusVariant}>
              {streaming && <Spinner size="sm" className="mr-1" />}
              {turn.status}
            </Badge>
            {turn.background ? <Badge variant="default">background</Badge> : null}
            {turn.usage?.output_tokens ? (
              <span>{fmtNumber(turn.usage.output_tokens)} out</span>
            ) : null}
            {turn.usage?.reasoning_tokens ? (
              <span>{fmtNumber(turn.usage.reasoning_tokens)} reasoning</span>
            ) : null}
            {turn.responseId ? (
              <button
                type="button"
                onClick={copyId}
                className="flex items-center gap-1 font-mono hover:text-foreground"
                title="Copy response id"
              >
                {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                {turn.responseId.slice(0, 18)}…
              </button>
            ) : null}
            {!streaming && (
              <IconButton
                icon={<Trash2 className="h-3.5 w-3.5" />}
                label="Delete response"
                onClick={() => onDelete(turn)}
                className="ml-auto p-1"
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function SettingSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-xs">
        <span className="font-medium">{label}</span>
        <span className="tabular-nums text-muted-foreground">{format ? format(value) : value}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={(v) => onChange(v[0] ?? min)} />
    </div>
  );
}

function SettingSwitch({
  label,
  checked,
  onChange,
}: {
  label: ReactNode;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 text-xs">
      <span className="font-medium">{label}</span>
      <Switch checked={checked} onCheckedChange={onChange} />
    </label>
  );
}
