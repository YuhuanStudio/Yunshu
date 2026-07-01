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
  Spinner,
  EmptyState,
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
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
  ChevronDown,
  User,
  Sparkles,
} from "lucide-react";
import { api, streamSSE } from "@/lib/api";
import { fmtNumber } from "@/lib/format";
import { Markdown } from "@/components/markdown";
import type { ChatMessage, Conversation, Model, ToolCall } from "@/lib/types";

const STORE_KEY = "yunshu_chat_v2";
const uid = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

interface Settings {
  temperature: number;
  maxTokens: number;
  enableThinking: boolean;
  specDecode: boolean;
  logprobs: boolean;
  jsonMode: boolean;
}
const DEFAULT_SETTINGS: Settings = {
  temperature: 0.7,
  maxTokens: 1024,
  enableThinking: false,
  specDecode: false,
  logprobs: false,
  jsonMode: false,
};

function newConversation(model: string): Conversation {
  const now = Date.now();
  return { id: uid(), title: "New chat", messages: [], model, createdAt: now, updatedAt: now };
}

export default function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

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

  const active = useMemo(() => conversations.find((c) => c.id === activeId) ?? null, [conversations, activeId]);

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

  const send = async () => {
    const text = input.trim();
    if (!text || streaming || !model) return;

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

    const userMsg: ChatMessage = { id: uid(), role: "user", content: text };
    const assistantMsg: ChatMessage = { id: uid(), role: "assistant", content: "", reasoning: "", streaming: true };

    setInput("");
    setStreaming(true);

    // Snapshot history for the request (before adding the placeholder).
    const base = conversations.find((c) => c.id === convId);
    const history = [...(base?.messages ?? []), userMsg];

    updateConversation(convId, (c) => ({
      ...c,
      title: c.messages.length === 0 ? text.slice(0, 40) : c.title,
      messages: [...c.messages, userMsg, assistantMsg],
      model,
      updatedAt: Date.now(),
    }));

    const started = performance.now();
    const controller = new AbortController();
    abortRef.current = controller;

    const body: Record<string, unknown> = {
      model,
      messages: history.map((m) => ({ role: m.role, content: m.content })),
      temperature: settings.temperature,
      max_tokens: settings.maxTokens,
      stream: true,
    };
    if (settings.enableThinking) body.enable_thinking = true;
    if (settings.specDecode) body.spec_decode = true;
    if (settings.logprobs) body.logprobs = true;
    if (settings.jsonMode) body.response_format = { type: "json_object" };

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
          m.id === assistantMsg.id ? { ...m, content, reasoning, tokens, latencyMs, streaming: false } : m,
        ),
      }));
      setStreaming(false);
      abortRef.current = null;
    }
  };

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
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteChat(c.id);
                }}
                className="opacity-0 transition-opacity hover:text-error group-hover:opacity-100"
                aria-label="Delete conversation"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
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
          <IconButton
            className="ml-auto"
            icon={<Settings2 className="h-4 w-4" />}
            label="Settings"
            onClick={() => setShowSettings((s) => !s)}
          />
        </div>

        {/* Settings drawer */}
        {showSettings && (
          <div className="grid grid-cols-2 gap-x-8 gap-y-4 border-b border-border bg-muted/30 px-4 py-4 lg:grid-cols-3">
            <SettingSlider
              label="Temperature"
              value={settings.temperature}
              min={0}
              max={2}
              step={0.05}
              onChange={(v) => setSettings((s) => ({ ...s, temperature: v }))}
            />
            <SettingSlider
              label="Max tokens"
              value={settings.maxTokens}
              min={64}
              max={8192}
              step={64}
              onChange={(v) => setSettings((s) => ({ ...s, maxTokens: v }))}
              format={fmtNumber}
            />
            <div className="flex flex-col justify-center gap-2">
              <SettingSwitch label="Thinking" checked={settings.enableThinking} onChange={(v) => setSettings((s) => ({ ...s, enableThinking: v }))} />
              <SettingSwitch label="Spec decode" checked={settings.specDecode} onChange={(v) => setSettings((s) => ({ ...s, specDecode: v }))} />
              <SettingSwitch label="JSON mode" checked={settings.jsonMode} onChange={(v) => setSettings((s) => ({ ...s, jsonMode: v }))} />
            </div>
          </div>
        )}

        {/* Transcript */}
        <div ref={scrollRef} className="flex-1 overflow-auto px-4 py-6">
          <div className="mx-auto max-w-3xl space-y-6">
            {!active || active.messages.length === 0 ? (
              <div className="pt-20">
                <EmptyState
                  icon={<Sparkles className="h-6 w-6" />}
                  title="Start a conversation"
                  description="Pick a model and send a message. Streaming, thinking and tool calls are all supported."
                />
              </div>
            ) : (
              active.messages.map((m) => <MessageBubble key={m.id} message={m} />)
            )}
          </div>
        </div>

        {/* Composer */}
        <div className="border-t border-border p-4">
          <div className="mx-auto flex max-w-3xl items-end gap-2">
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
              <Button onClick={send} disabled={!input.trim() || !model}>
                <Send className="h-4 w-4" /> Send
              </Button>
            )}
          </div>
        </div>
      </div>
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
          <div className="mt-1 flex items-center gap-2 text-[11px] text-muted-foreground">
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
