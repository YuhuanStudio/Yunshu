"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  Button,
  IconButton,
  Textarea,
  Slider,
  Switch,
  Card,
  Badge,
  EmptyState,
  Sheet,
  SegmentedSelect,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Separator,
  cn,
} from "yunui";
import { ThinkingBlock } from "yunui/ai";
import { ModelPicker } from "@/components/model-picker";
import {
  Plus,
  Trash2,
  Send,
  Square,
  Settings2,
  MessageSquare,
  Wrench,
  ImagePlus,
  X,
  User,
  Sparkles,
} from "lucide-react";
import { api, streamSSE } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import { guessModelType } from "@/lib/model-type";
import { Markdown } from "@/components/markdown";
import type { ChatMessage, Conversation, Model, ToolCall } from "@/lib/types";

const STORE_KEY = "yunshu_chat_v2";
const uid = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

type ReasoningEffort = "low" | "medium" | "high";
type ResponseFormat = "text" | "json_object" | "json_schema";

interface Settings {
  temperature: number;
  maxTokens: number;
  topP: number;
  enableThinking: boolean;
  reasoningEffort: ReasoningEffort;
  useThinkingBudget: boolean;
  thinkingBudget: number;
  toolsJson: string;
  toolChoice: "auto" | "none" | "required";
  responseFormat: ResponseFormat;
  jsonSchema: string;
}
const DEFAULT_SETTINGS: Settings = {
  temperature: 0.7,
  maxTokens: 1024,
  topP: 1,
  enableThinking: false,
  reasoningEffort: "medium",
  useThinkingBudget: false,
  thinkingBudget: 4096,
  toolsJson: "",
  toolChoice: "auto",
  responseFormat: "text",
  jsonSchema: `{
  "name": "response",
  "schema": {
    "type": "object",
    "properties": {},
    "required": []
  }
}`,
};

/** Parse a JSON string, returning `{ value }` or `{ error }`. Empty → null value. */
function tryParse(raw: string): { value: unknown; error: string | null } {
  const trimmed = raw.trim();
  if (!trimmed) return { value: null, error: null };
  try {
    return { value: JSON.parse(trimmed), error: null };
  } catch (e) {
    return { value: null, error: (e as Error).message };
  }
}

function newConversation(model: string): Conversation {
  const now = Date.now();
  return { id: uid(), title: "New chat", messages: [], model, createdAt: now, updatedAt: now };
}

/** Build the OpenAI message content for a stored message (parts array iff images). */
function messageContent(m: ChatMessage): unknown {
  if (m.images && m.images.length > 0) {
    return [
      ...(m.content ? [{ type: "text", text: m.content }] : []),
      ...m.images.map((url) => ({ type: "image_url", image_url: { url } })),
    ];
  }
  return m.content;
}

export default function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<string[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Load persisted conversations + settings.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as { conversations: Conversation[]; settings?: Settings };
        setConversations(parsed.conversations ?? []);
        setActiveId(parsed.conversations?.[0]?.id ?? null);
        if (parsed.settings) setSettings({ ...DEFAULT_SETTINGS, ...parsed.settings });
      }
    } catch {
      /* ignore corrupt store */
    }
  }, []);

  // Fetch models once.
  useEffect(() => {
    api
      .get<{ data: Model[] }>("/v1/models")
      .then((d) => {
        setModels(d.data ?? []);
        setModel((m) => m || d.data?.find((x) => x.loaded)?.id || d.data?.[0]?.id || "");
      })
      .catch(() => {});
  }, []);

  const persist = useCallback((next: Conversation[], s: Settings) => {
    localStorage.setItem(STORE_KEY, JSON.stringify({ conversations: next, settings: s }));
  }, []);

  // Keep settings changes persisted.
  useEffect(() => {
    persist(conversations, settings);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings]);

  const active = useMemo(() => conversations.find((c) => c.id === activeId) ?? null, [conversations, activeId]);

  // Whether the active model can take images (VLM / omni).
  const isVision = useMemo(() => {
    const meta = models.find((m) => m.id === model);
    const type = (meta?.type ?? (model ? guessModelType(model) : "LLM")) as string;
    return type === "VLM" || type === "OCR";
  }, [models, model]);

  // Parsed tools + response_format validation (surfaced inline in the drawer).
  const tools = useMemo(() => tryParse(settings.toolsJson), [settings.toolsJson]);
  const toolsError =
    tools.error ??
    (settings.toolsJson.trim() && !Array.isArray(tools.value) ? "Tools must be a JSON array." : null);
  const schema = useMemo(() => tryParse(settings.jsonSchema), [settings.jsonSchema]);
  const schemaError =
    settings.responseFormat === "json_schema"
      ? (schema.error ?? (schema.value == null ? "A JSON schema is required." : null))
      : null;
  const settingsInvalid = Boolean(toolsError) || Boolean(schemaError);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [active?.messages.length, streaming]);

  const updateConversation = useCallback(
    (id: string, fn: (c: Conversation) => Conversation) => {
      setConversations((prev) => {
        const next = prev.map((c) => (c.id === id ? fn(c) : c));
        persist(next, settings);
        return next;
      });
    },
    [persist, settings],
  );

  const createChat = () => {
    const conv = newConversation(model);
    setConversations((prev) => {
      const next = [conv, ...prev];
      persist(next, settings);
      return next;
    });
    setActiveId(conv.id);
  };

  const deleteChat = (id: string) => {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      persist(next, settings);
      if (activeId === id) setActiveId(next[0]?.id ?? null);
      return next;
    });
  };

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
  };

  const onPickFiles = (files: FileList | null) => {
    if (!files) return;
    Array.from(files).forEach((f) => {
      if (!f.type.startsWith("image/")) return;
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") setAttachments((prev) => [...prev, reader.result as string]);
      };
      reader.readAsDataURL(f);
    });
  };

  const send = async () => {
    const text = input.trim();
    const imgs = attachments;
    if ((!text && imgs.length === 0) || streaming || !model || settingsInvalid) return;

    // Ensure a conversation exists.
    let convId = activeId;
    if (!convId) {
      const conv = newConversation(model);
      convId = conv.id;
      setConversations((prev) => {
        const next = [conv, ...prev];
        persist(next, settings);
        return next;
      });
      setActiveId(convId);
    }

    const userMsg: ChatMessage = {
      id: uid(),
      role: "user",
      content: text,
      images: imgs.length ? imgs : undefined,
    };
    const assistantMsg: ChatMessage = { id: uid(), role: "assistant", content: "", reasoning: "", streaming: true };

    setInput("");
    setAttachments([]);
    setStreaming(true);

    // Snapshot history for the request (before adding the placeholder).
    const base = conversations.find((c) => c.id === convId);
    const history = [...(base?.messages ?? []), userMsg];

    updateConversation(convId, (c) => ({
      ...c,
      title: c.messages.length === 0 ? (text || "Image message").slice(0, 40) : c.title,
      messages: [...c.messages, userMsg, assistantMsg],
      model,
      updatedAt: Date.now(),
    }));

    const started = performance.now();
    const controller = new AbortController();
    abortRef.current = controller;

    const body: Record<string, unknown> = {
      model,
      messages: history.map((m) => ({ role: m.role, content: messageContent(m) })),
      temperature: settings.temperature,
      top_p: settings.topP,
      max_tokens: settings.maxTokens,
      stream: true,
    };
    if (settings.enableThinking) {
      body.enable_thinking = true;
      body.reasoning_effort = settings.reasoningEffort;
      if (settings.useThinkingBudget) body.thinking_budget = settings.thinkingBudget;
    }
    if (Array.isArray(tools.value) && tools.value.length > 0) {
      body.tools = tools.value;
      body.tool_choice = settings.toolChoice;
    }
    if (settings.responseFormat === "json_object") {
      body.response_format = { type: "json_object" };
    } else if (settings.responseFormat === "json_schema" && schema.value) {
      body.response_format = { type: "json_schema", json_schema: schema.value };
    }

    const toolAcc: Record<number, ToolCall> = {};
    let content = "";
    let reasoning = "";
    let tokens = 0;

    const flush = () =>
      updateConversation(convId!, (c) => ({
        ...c,
        messages: c.messages.map((m) =>
          m.id === assistantMsg.id
            ? {
                ...m,
                content,
                reasoning,
                tokens,
                toolCalls: Object.values(toolAcc).length ? Object.values(toolAcc) : undefined,
              }
            : m,
        ),
      }));

    try {
      await streamSSE(
        "/v1/chat/completions",
        body,
        (chunk) => {
          const choice = (chunk.choices as Array<Record<string, unknown>> | undefined)?.[0];
          const delta = (choice?.delta as Record<string, unknown>) ?? {};
          if (typeof delta.reasoning_content === "string") reasoning += delta.reasoning_content;
          if (typeof delta.content === "string") content += delta.content;
          const usage = chunk.usage as { completion_tokens?: number } | undefined;
          if (usage?.completion_tokens) tokens = usage.completion_tokens;
          const deltaTools = delta.tool_calls as Array<Record<string, unknown>> | undefined;
          if (deltaTools) {
            for (const tc of deltaTools) {
              const i = (tc.index as number) ?? 0;
              const fn = (tc.function as { name?: string; arguments?: string }) ?? {};
              const existing = toolAcc[i] ?? { id: (tc.id as string) ?? `tool-${i}`, type: "function", function: { name: "", arguments: "" } };
              toolAcc[i] = {
                ...existing,
                function: {
                  name: fn.name ?? existing.function.name,
                  arguments: existing.function.arguments + (fn.arguments ?? ""),
                },
              };
            }
          }
          flush();
        },
        controller.signal,
      );
    } catch (e) {
      if (!controller.signal.aborted) content += `\n\n_Error: ${(e as Error).message}_`;
    } finally {
      const latencyMs = Math.round(performance.now() - started);
      updateConversation(convId!, (c) => ({
        ...c,
        messages: c.messages.map((m) =>
          m.id === assistantMsg.id
            ? {
                ...m,
                content,
                reasoning,
                tokens,
                latencyMs,
                streaming: false,
                toolCalls: Object.values(toolAcc).length ? Object.values(toolAcc) : undefined,
              }
            : m,
        ),
      }));
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const set = <K extends keyof Settings>(key: K, value: Settings[K]) =>
    setSettings((s) => ({ ...s, [key]: value }));

  return (
    <div className="flex h-full">
      {/* Conversation list */}
      <div className="flex w-64 shrink-0 flex-col border-r border-border">
        <div className="p-3">
          <Button className="w-full" onClick={createChat}>
            <Plus className="h-4 w-4" /> New chat
          </Button>
        </div>
        <div className="flex-1 space-y-0.5 overflow-auto px-2 pb-2">
          {conversations.map((c) => (
            <div
              key={c.id}
              onClick={() => setActiveId(c.id)}
              className={cn(
                "group flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-sm transition-colors",
                c.id === activeId ? "bg-accent/10 text-accent" : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
            >
              <MessageSquare className="h-4 w-4 shrink-0" />
              <span className="min-w-0 flex-1 truncate">{c.title}</span>
              <IconButton
                icon={<Trash2 className="h-3.5 w-3.5" />}
                label="Delete conversation"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteChat(c.id);
                }}
                className="-mr-1 shrink-0 p-1 opacity-0 transition-opacity group-hover:opacity-100"
              />
            </div>
          ))}
          {conversations.length === 0 && (
            <p className="px-2 py-6 text-center text-xs text-muted-foreground">No conversations yet.</p>
          )}
        </div>
      </div>

      {/* Main */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Header */}
        <div className="flex items-center gap-2 border-b border-border px-4 py-2.5">
          <ModelPicker models={models} value={model} onChange={setModel} className="w-64" />
          {settings.enableThinking && <Badge variant="info">thinking</Badge>}
          {Array.isArray(tools.value) && tools.value.length > 0 && (
            <Badge variant="default">{tools.value.length} tools</Badge>
          )}
          {settings.responseFormat !== "text" && <Badge variant="default">{settings.responseFormat}</Badge>}
          <IconButton
            className="ml-auto"
            icon={<Settings2 className="h-4 w-4" />}
            label="Settings"
            onClick={() => setShowSettings(true)}
          />
        </div>

        {/* Transcript */}
        <div ref={scrollRef} className="flex-1 overflow-auto px-4 py-6">
          <div className="mx-auto max-w-3xl space-y-6">
            {!active || active.messages.length === 0 ? (
              <div className="pt-20">
                <EmptyState
                  icon={<Sparkles className="h-6 w-6" />}
                  title="Start a conversation"
                  description="Pick a model and send a message. Vision, thinking, tool calls and structured output are all supported — configure them in Settings."
                />
              </div>
            ) : (
              active.messages.map((m) => <MessageBubble key={m.id} message={m} />)
            )}
          </div>
        </div>

        {/* Composer */}
        <div className="border-t border-border p-4">
          <div className="mx-auto max-w-3xl space-y-2">
            {attachments.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {attachments.map((src, i) => (
                  <div key={i} className="relative">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={src} alt="attachment" className="h-16 w-16 rounded-lg border border-border object-cover" />
                    <button
                      type="button"
                      onClick={() => setAttachments((prev) => prev.filter((_, j) => j !== i))}
                      className="absolute -right-1.5 -top-1.5 rounded-full bg-foreground/80 p-0.5 text-background hover:bg-foreground"
                      aria-label="Remove image"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="flex items-end gap-2">
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                multiple
                hidden
                onChange={(e) => {
                  onPickFiles(e.target.files);
                  e.target.value = "";
                }}
              />
              <IconButton
                icon={<ImagePlus className="h-4 w-4" />}
                label={isVision ? "Attach image" : "The selected model is not a vision model"}
                onClick={() => fileRef.current?.click()}
                disabled={!isVision || streaming}
              />
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                placeholder="Send a message…  (Enter to send, Shift+Enter for newline)"
                rows={2}
                className="flex-1 resize-none"
                disabled={streaming}
              />
              {streaming ? (
                <Button variant="secondary" onClick={stop}>
                  <Square className="h-4 w-4" /> Stop
                </Button>
              ) : (
                <Button
                  onClick={send}
                  disabled={(!input.trim() && attachments.length === 0) || !model || settingsInvalid}
                >
                  <Send className="h-4 w-4" /> Send
                </Button>
              )}
            </div>
            {settingsInvalid && (
              <p className="text-xs text-error">Fix the invalid Tools / schema JSON in Settings before sending.</p>
            )}
          </div>
        </div>
      </div>

      {/* Settings drawer */}
      <Sheet open={showSettings} onClose={() => setShowSettings(false)} title="Chat settings">
        <div className="space-y-6 pb-6">
          {/* Sampling */}
          <section className="space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Sampling</h3>
            <SettingSlider label="Temperature" value={settings.temperature} min={0} max={2} step={0.05} onChange={(v) => set("temperature", v)} />
            <SettingSlider label="Top P" value={settings.topP} min={0} max={1} step={0.01} onChange={(v) => set("topP", v)} />
            <SettingSlider label="Max tokens" value={settings.maxTokens} min={64} max={8192} step={64} onChange={(v) => set("maxTokens", v)} format={fmtNumber} />
          </section>

          <Separator />

          {/* Thinking */}
          <section className="space-y-4">
            <SettingSwitch label="Enable thinking" checked={settings.enableThinking} onChange={(v) => set("enableThinking", v)} />
            {settings.enableThinking && (
              <>
                <div>
                  <p className="mb-1.5 text-xs font-medium">Reasoning effort</p>
                  <SegmentedSelect
                    value={settings.reasoningEffort}
                    onChange={(v) => set("reasoningEffort", v as ReasoningEffort)}
                    options={[
                      { value: "low", label: "Low" },
                      { value: "medium", label: "Medium" },
                      { value: "high", label: "High" },
                    ]}
                  />
                </div>
                <SettingSwitch label="Set thinking budget" checked={settings.useThinkingBudget} onChange={(v) => set("useThinkingBudget", v)} />
                {settings.useThinkingBudget && (
                  <SettingSlider label="Thinking budget" value={settings.thinkingBudget} min={1} max={32768} step={256} onChange={(v) => set("thinkingBudget", v)} format={fmtNumber} />
                )}
              </>
            )}
          </section>

          <Separator />

          {/* Tools */}
          <section className="space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Tools (JSON)</h3>
            <Textarea
              value={settings.toolsJson}
              onChange={(e) => set("toolsJson", e.target.value)}
              rows={7}
              placeholder={`[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the weather for a city",
      "parameters": { "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] }
    }
  }
]`}
              className="font-mono text-xs"
            />
            {toolsError ? (
              <p className="text-xs text-error">{toolsError}</p>
            ) : Array.isArray(tools.value) && tools.value.length > 0 ? (
              <p className="text-xs text-success">{tools.value.length} tool(s) parsed.</p>
            ) : null}
            {settings.toolsJson.trim() && !toolsError && (
              <div>
                <p className="mb-1.5 text-xs font-medium">Tool choice</p>
                <SegmentedSelect
                  value={settings.toolChoice}
                  onChange={(v) => set("toolChoice", v as Settings["toolChoice"])}
                  options={[
                    { value: "auto", label: "Auto" },
                    { value: "required", label: "Required" },
                    { value: "none", label: "None" },
                  ]}
                />
              </div>
            )}
          </section>

          <Separator />

          {/* Response format */}
          <section className="space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Response format</h3>
            <Select value={settings.responseFormat} onValueChange={(v) => set("responseFormat", v as ResponseFormat)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="text">Text</SelectItem>
                <SelectItem value="json_object">JSON object</SelectItem>
                <SelectItem value="json_schema">JSON schema</SelectItem>
              </SelectContent>
            </Select>
            {settings.responseFormat === "json_schema" && (
              <>
                <Textarea
                  value={settings.jsonSchema}
                  onChange={(e) => set("jsonSchema", e.target.value)}
                  rows={9}
                  className="font-mono text-xs"
                />
                {schemaError && <p className="text-xs text-error">{schemaError}</p>}
              </>
            )}
          </section>
        </div>
      </Sheet>
    </div>
  );
}

function MessageBubble({ message: m }: { message: ChatMessage }) {
  const isUser = m.role === "user";
  return (
    <div className={cn("message-fade-in flex gap-3", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-accent/10 text-accent" : "bg-muted text-muted-foreground",
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
      </div>
      <div className={cn("min-w-0 flex-1", isUser && "flex flex-col items-end")}>
        {m.reasoning ? (
          <div className="mb-2 w-full">
            <ThinkingBlock content={m.reasoning} isStreaming={m.streaming} renderContent={(c) => <Markdown>{c}</Markdown>} />
          </div>
        ) : null}

        {m.images && m.images.length > 0 && (
          <div className={cn("mb-2 flex flex-wrap gap-2", isUser && "justify-end")}>
            {m.images.map((src, i) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img key={i} src={src} alt="attachment" className="max-h-48 rounded-lg border border-border object-cover" />
            ))}
          </div>
        )}

        {m.toolCalls?.map((t) => (
          <Card key={t.id} className="mb-2 w-full p-3">
            <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <Wrench className="h-3.5 w-3.5" /> {t.function.name}
            </div>
            <pre className="overflow-auto rounded bg-muted p-2 text-xs">{t.function.arguments}</pre>
          </Card>
        ))}

        {(m.content || m.streaming) && (
          <Card className={cn("w-fit max-w-full p-3.5", isUser && "bg-accent/10")}>
            {isUser ? (
              <p className="whitespace-pre-wrap break-words text-sm">{m.content}</p>
            ) : (
              <Markdown>{m.content || "…"}</Markdown>
            )}
          </Card>
        )}

        {!m.streaming && (m.tokens || m.latencyMs) ? (
          <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
            {m.tokens ? <Badge variant="default">{fmtNumber(m.tokens)} tok</Badge> : null}
            {m.latencyMs ? <span>{(m.latencyMs / 1000).toFixed(1)}s</span> : null}
          </div>
        ) : null}
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
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={(v) => onChange(v[0])} />
    </div>
  );
}

function SettingSwitch({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 text-xs">
      <span className="font-medium">{label}</span>
      <Switch checked={checked} onCheckedChange={onChange} />
    </label>
  );
}
