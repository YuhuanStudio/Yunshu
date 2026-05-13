"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import {
  Send,
  Square,
  Trash2,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  Bot,
  User,
  Loader2,
  Settings2,
  Sparkles,
  Plus,
  MessageSquare,
  Pencil,
  ImagePlus,
  X,
} from "lucide-react";
import "highlight.js/styles/github-dark.css";
import "katex/dist/katex.min.css";

// ── Types ──

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  imageUrl?: string;
  reasoning?: string;
  thinking?: boolean;
  tokens?: number;
  latencyMs?: number;
  logprobs?: { tokens: string[]; token_logprobs: number[] };
  streaming?: boolean;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  model: string;
  createdAt: number;
  updatedAt: number;
}

interface Model {
  id: string;
  loaded?: boolean;
}

// ── Storage ──

const STORAGE_KEY = "yunshu_conversations";

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveConversations(convs: Conversation[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(convs));
  } catch {
    // Storage full or unavailable
  }
}

// ── Main Component ──

export default function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [model, setModel] = useState("");
  const [models, setModels] = useState<Model[]>([]);
  const [enableThinking, setEnableThinking] = useState(false);
  const [thinkingBudget, setThinkingBudget] = useState(0);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [showSettings, setShowSettings] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [jsonMode, setJsonMode] = useState(false);
  const [specDecode, setSpecDecode] = useState(false);
  const [showLogprobs, setShowLogprobs] = useState(false);
  const [attachedImage, setAttachedImage] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const editingTitleRef = useRef<string | null>(null);

  // Load conversations + models on mount
  useEffect(() => {
    setConversations(loadConversations());

    fetch("/v1/models")
      .then((r) => r.json())
      .then((data) => {
        const all = data.data || [];
        setModels(all);
        const llm = all.find((m: Model) =>
          /qwen|llama|gemma|mistral|phi|deepseek/i.test(m.id)
        );
        if (llm) setModel(llm.id);
        else if (all.length > 0) setModel(all[0].id);
      })
      .catch(() => {});
  }, []);

  // Auto-save on change
  useEffect(() => {
    if (conversations.length > 0 || localStorage.getItem(STORAGE_KEY)) {
      saveConversations(conversations);
    }
  }, [conversations]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversations, activeId]);

  const activeConv = conversations.find((c) => c.id === activeId) || null;
  const messages = activeConv?.messages || [];

  // Create new conversation
  const newConversation = useCallback(() => {
    const id = `conv-${Date.now()}`;
    const conv: Conversation = {
      id,
      title: "New Chat",
      messages: [],
      model: model,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setConversations((prev) => [conv, ...prev]);
    setActiveId(id);
    setInput("");
  }, [model]);

  // Delete conversation
  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) {
        const remaining = conversations.filter((c) => c.id !== id);
        setActiveId(remaining.length > 0 ? remaining[0].id : null);
      }
    },
    [activeId, conversations]
  );

  // Rename conversation
  const renameConversation = useCallback((id: string, title: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title, updatedAt: Date.now() } : c))
    );
  }, []);

  // Auto-title from first user message
  const autoTitle = useCallback(
    (convId: string, msgs: Message[]) => {
      const first = msgs.find((m) => m.role === "user");
      if (first && first.content.length > 0) {
        const title = first.content.slice(0, 40) + (first.content.length > 40 ? "..." : "");
        setConversations((prev) =>
          prev.map((c) =>
            c.id === convId && c.title === "New Chat"
              ? { ...c, title, updatedAt: Date.now() }
              : c
          )
        );
      }
    },
    []
  );

  // Update messages in active conversation
  const updateMessages = useCallback(
    (updater: (msgs: Message[]) => Message[]) => {
      if (!activeId) return;
      setConversations((prev) =>
        prev.map((c) =>
          c.id === activeId
            ? { ...c, messages: updater(c.messages), updatedAt: Date.now() }
            : c
        )
      );
    },
    [activeId]
  );

  // Auto-resize textarea
  const handleInput = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  }, []);

  // Send message
  const sendMessage = useCallback(async () => {
    if (!input.trim() || streaming || !model) return;

    // Ensure we have an active conversation
    let convId = activeId;
    if (!convId) {
      convId = `conv-${Date.now()}`;
      const conv: Conversation = {
        id: convId,
        title: "New Chat",
        messages: [],
        model: model,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      setConversations((prev) => [conv, ...prev]);
      setActiveId(convId);
    }

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: input.trim(),
      imageUrl: attachedImage || undefined,
    };

    const assistantMsg: Message = {
      id: `asst-${Date.now()}`,
      role: "assistant",
      content: "",
      reasoning: "",
      streaming: true,
    };

    const newMessages = [...(conversations.find((c) => c.id === convId)?.messages || []), userMsg, assistantMsg];

    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? { ...c, messages: newMessages, model, updatedAt: Date.now() }
          : c
      )
    );
    setActiveId(convId);
    setInput("");
    setAttachedImage(null);
    if (inputRef.current) inputRef.current.style.height = "auto";

    // Auto-title
    autoTitle(convId, newMessages);

    const startTime = Date.now();
    abortRef.current = new AbortController();

    try {
      const history = newMessages.filter((m) => m.role !== "system" && !m.streaming).slice(0, -1);
      const apiMessages: Array<{ role: string; content: string | Array<{ type: string; text?: string; image_url?: { url: string } }> }> = [];

      // System prompt
      if (systemPrompt.trim()) {
        apiMessages.push({ role: "system", content: systemPrompt });
      }

      // History
      for (const m of [...history, userMsg]) {
        if (m.imageUrl) {
          apiMessages.push({
            role: m.role,
            content: [
              { type: "image_url", image_url: { url: m.imageUrl } },
              { type: "text", text: m.content },
            ],
          });
        } else {
          apiMessages.push({ role: m.role, content: m.content });
        }
      }

      const payload: Record<string, unknown> = {
        model,
        messages: apiMessages,
        temperature,
        max_tokens: maxTokens,
        stream: true,
        enable_thinking: enableThinking || undefined,
        thinking_budget: enableThinking && thinkingBudget > 0 ? thinkingBudget : undefined,
        spec_decode: specDecode || undefined,
        logprobs: showLogprobs || undefined,
      };
      if (jsonMode) {
        payload.response_format = { type: "json_object" };
      }

      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`${response.status}: ${errText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No reader");

      const decoder = new TextDecoder();
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
            const chunk = JSON.parse(data);
            const delta = chunk.choices?.[0]?.delta;
            if (!delta) continue;

            setConversations((prev) =>
              prev.map((c) => {
                if (c.id !== convId) return c;
                const updated = [...c.messages];
                const last = updated[updated.length - 1];
                if (last.role !== "assistant") return c;

                if (delta.reasoning_content) {
                  last.reasoning = (last.reasoning || "") + delta.reasoning_content;
                  last.thinking = true;
                }
                if (delta.content) {
                  last.content += delta.content;
                }

                const usage = chunk.usage;
                if (usage?.completion_tokens) {
                  last.tokens = usage.completion_tokens;
                }

                // Capture logprobs from streaming chunks
                const lp = chunk.choices?.[0]?.logprobs;
                if (lp?.tokens && lp.tokens.length > 0) {
                  if (!last.logprobs) {
                    last.logprobs = { tokens: [], token_logprobs: [] };
                  }
                  last.logprobs.tokens.push(...lp.tokens);
                  if (lp.token_logprobs) {
                    last.logprobs.token_logprobs.push(...lp.token_logprobs);
                  }
                }

                return { ...c, messages: updated, updatedAt: Date.now() };
              })
            );
          } catch {
            // skip malformed
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") return;
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          const updated = [...c.messages];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            last.content = `Error: ${err instanceof Error ? err.message : String(err)}`;
            last.streaming = false;
          }
          return { ...c, messages: updated };
        })
      );
    } finally {
      const elapsed = Date.now() - startTime;
      setStreaming(false);
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          const updated = [...c.messages];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            last.streaming = false;
            last.latencyMs = elapsed;
          }
          return { ...c, messages: updated };
        })
      );
    }
  }, [input, streaming, model, conversations, activeId, temperature, maxTokens, enableThinking, autoTitle]);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const clearChat = useCallback(() => {
    if (!activeId) return;
    setConversations((prev) =>
      prev.map((c) => (c.id === activeId ? { ...c, messages: [], updatedAt: Date.now() } : c))
    );
  }, [activeId]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    },
    [sendMessage]
  );

  return (
    <div className="flex h-full">
      {/* Conversation sidebar */}
      {showSidebar && (
        <div className="w-64 border-r border-[var(--color-border)] bg-[var(--color-bg-secondary)] flex flex-col shrink-0">
          <div className="p-3 border-b border-[var(--color-border)]">
            <button
              onClick={newConversation}
              className="w-full flex items-center justify-center gap-2 bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white rounded-lg px-3 py-2 text-sm font-medium transition-colors"
            >
              <Plus className="w-4 h-4" />
              New Chat
            </button>
          </div>
          <div className="flex-1 overflow-auto p-2 space-y-1">
            {conversations.map((conv) => (
              <ConvItem
                key={conv.id}
                conv={conv}
                active={conv.id === activeId}
                onSelect={() => { setActiveId(conv.id); setModel(conv.model || model); }}
                onDelete={() => deleteConversation(conv.id)}
                onRename={(title) => renameConversation(conv.id, title)}
              />
            ))}
            {conversations.length === 0 && (
              <div className="text-xs text-[var(--color-text-secondary)] text-center py-4">
                No conversations yet
              </div>
            )}
          </div>
        </div>
      )}

      {/* Chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="border-b border-[var(--color-border)] px-4 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="p-1.5 rounded-lg text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] transition-colors"
            >
              <MessageSquare className="w-4 h-4" />
            </button>
            <Sparkles className="w-5 h-5 text-[var(--color-accent)]" />
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-1.5 text-sm min-w-[200px]"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id}
                </option>
              ))}
            </select>
            {streaming && (
              <span className="flex items-center gap-1.5 text-xs text-[var(--color-accent)]">
                <Loader2 className="w-3 h-3 animate-spin" />
                Generating...
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-1.5 rounded-lg transition-colors ${
                showSettings
                  ? "bg-[var(--color-accent)]/20 text-[var(--color-accent)]"
                  : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)]"
              }`}
            >
              <Settings2 className="w-4 h-4" />
            </button>
            <button
              onClick={clearChat}
              className="p-1.5 rounded-lg text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] transition-colors"
              title="Clear current chat"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-auto">
          {!activeConv || messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md">
                <Bot className="w-12 h-12 mx-auto mb-4 text-[var(--color-accent)] opacity-50" />
                <h3 className="text-lg font-medium mb-2">Start a conversation</h3>
                <p className="text-sm text-[var(--color-text-secondary)]">
                  Send a message to begin chatting with the model.
                  Supports Markdown, code blocks, and LaTeX.
                </p>
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} msg={msg} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-[var(--color-border)] p-4 shrink-0">
          <div className="max-w-4xl mx-auto">
            {attachedImage && (
              <div className="flex items-center gap-2 mb-2 px-1">
                <img src={attachedImage} alt="attached" className="h-12 w-12 object-cover rounded-lg border border-[var(--color-border)]" />
                <button onClick={() => setAttachedImage(null)} className="p-1 rounded hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]">
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}
            <div className="relative flex items-end gap-2">
              <button
                onClick={() => imageInputRef.current?.click()}
                className="p-2.5 rounded-lg text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] transition-colors shrink-0"
                title="Attach image (VLM)"
              >
                <ImagePlus className="w-4 h-4" />
              </button>
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    const reader = new FileReader();
                    reader.onload = (ev) => setAttachedImage(ev.target?.result as string);
                    reader.readAsDataURL(file);
                  }
                }}
              />
              <textarea
                ref={inputRef}
                value={input}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                placeholder="Type a message... (Shift+Enter for new line)"
                rows={1}
                className="flex-1 bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl px-4 py-3 pr-12 text-sm resize-none focus:outline-none focus:border-[var(--color-accent)] min-h-[44px] max-h-[200px]"
                disabled={streaming}
              />
              <div className="absolute right-14 bottom-2">
                {streaming ? (
                  <button
                    onClick={stopGeneration}
                    className="p-2 rounded-lg bg-[var(--color-danger)] text-white hover:opacity-90 transition-opacity"
                  >
                    <Square className="w-4 h-4" />
                  </button>
                ) : (
                  <button
                    onClick={sendMessage}
                    disabled={!input.trim() || !model}
                    className="p-2 rounded-lg bg-[var(--color-accent)] text-white hover:bg-[var(--color-accent-hover)] disabled:opacity-30 transition-colors"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Settings sidebar */}
      {showSettings && (
        <div className="w-72 border-l border-[var(--color-border)] p-4 space-y-5 shrink-0 overflow-auto">
          <h3 className="font-semibold text-sm flex items-center gap-2">
            <Settings2 className="w-4 h-4" />
            Parameters
          </h3>

          <ParamSlider
            label="Temperature"
            value={temperature}
            onChange={setTemperature}
            min={0}
            max={2}
            step={0.1}
          />

          <ParamInput
            label="Max Tokens"
            value={maxTokens}
            onChange={setMaxTokens}
            min={1}
            max={32768}
          />

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="thinking"
              checked={enableThinking}
              onChange={(e) => setEnableThinking(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="thinking" className="text-sm">
              Enable Thinking
            </label>
            {enableThinking && (
              <span className="ml-2 text-xs text-[var(--color-text-secondary)]">
                Budget: <input type="number" min={0} max={32768} value={thinkingBudget}
                  onChange={(e) => setThinkingBudget(parseInt(e.target.value) || 0)}
                  className="w-16 px-1 py-0.5 text-xs bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded text-center"
                /> tokens (0 = unlimited)
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="jsonMode"
              checked={jsonMode}
              onChange={(e) => setJsonMode(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="jsonMode" className="text-sm">
              JSON Mode
            </label>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="specDecode"
              checked={specDecode}
              onChange={(e) => setSpecDecode(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="specDecode" className="text-sm">
              Speculative Decode
            </label>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="showLogprobs"
              checked={showLogprobs}
              onChange={(e) => setShowLogprobs(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="showLogprobs" className="text-sm">
              Show Logprobs
            </label>
          </div>

          <div>
            <label className="text-sm text-[var(--color-text-secondary)] block mb-1">
              System Prompt
            </label>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={3}
              placeholder="You are a helpful assistant..."
              className="w-full bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:border-[var(--color-accent)]"
            />
          </div>

          <div className="pt-4 border-t border-[var(--color-border)]">
            <h4 className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">
              About
            </h4>
            <p className="text-xs text-[var(--color-text-secondary)] leading-relaxed">
              Yunshu is a production-grade MLX inference platform for Apple Silicon.
              Based on mlx-lm BatchGenerator with oMLX-pattern continuous batching.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Conversation Item ──

function ConvItem({
  conv,
  active,
  onSelect,
  onDelete,
  onRename,
}: {
  conv: Conversation;
  active: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onRename: (title: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(conv.title);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  const commitRename = () => {
    if (editTitle.trim() && editTitle !== conv.title) {
      onRename(editTitle.trim());
    }
    setEditing(false);
  };

  const msgCount = conv.messages.filter((m) => m.role !== "system").length;

  return (
    <div
      className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
        active
          ? "bg-[var(--color-accent-muted)] text-[var(--color-accent)]"
          : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] hover:text-[var(--color-text-primary)]"
      }`}
      onClick={onSelect}
    >
      <MessageSquare className="w-3.5 h-3.5 shrink-0" />
      <div className="flex-1 min-w-0">
        {editing ? (
          <input
            ref={inputRef}
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              if (e.key === "Escape") setEditing(false);
            }}
            onClick={(e) => e.stopPropagation()}
            className="w-full bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded px-1.5 py-0.5 text-xs text-[var(--color-text-primary)]"
          />
        ) : (
          <>
            <div className="text-xs font-medium truncate">{conv.title}</div>
            <div className="text-[10px] opacity-60">{msgCount} messages</div>
          </>
        )}
      </div>
      <div className="hidden group-hover:flex items-center gap-0.5">
        <button
          onClick={(e) => { e.stopPropagation(); setEditing(true); setEditTitle(conv.title); }}
          className="p-0.5 rounded hover:bg-[var(--color-bg-tertiary)]"
        >
          <Pencil className="w-3 h-3" />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="p-0.5 rounded hover:bg-[var(--color-bg-tertiary)] text-[var(--color-danger)]"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}

// ── Message Bubble ──

function MessageBubble({ msg }: { msg: Message }) {
  const [showThinking, setShowThinking] = useState(false);
  const [copied, setCopied] = useState(false);

  const isUser = msg.role === "user";

  const copyContent = useCallback(() => {
    navigator.clipboard.writeText(msg.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [msg.content]);

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"} message-fade-in`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-[var(--color-accent)]/20 flex items-center justify-center shrink-0 mt-1">
          <Bot className="w-4 h-4 text-[var(--color-accent)]" />
        </div>
      )}

      <div
        className={`max-w-[75%] min-w-0 ${
          isUser
            ? "bg-[var(--color-accent)] text-white rounded-2xl rounded-br-md px-4 py-3"
            : "w-full"
        }`}
      >
        {/* Thinking section */}
        {!isUser && msg.thinking && msg.reasoning && (
          <div className="mb-3">
            <button
              onClick={() => setShowThinking(!showThinking)}
              className="flex items-center gap-1.5 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
            >
              {showThinking ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
              <span className="flex items-center gap-1">
                <Sparkles className="w-3 h-3" />
                {msg.streaming ? "Thinking..." : "Thought process"}
              </span>
            </button>
            {showThinking && (
              <div className="mt-2 p-3 bg-[var(--color-bg-tertiary)] rounded-lg border border-[var(--color-border)] text-sm text-[var(--color-text-secondary)] whitespace-pre-wrap max-h-64 overflow-auto">
                {msg.reasoning}
              </div>
            )}
          </div>
        )}

        {/* Content */}
        {isUser ? (
          <div className="whitespace-pre-wrap break-words">{msg.content}</div>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none">
            {msg.content ? (
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex, rehypeHighlight]}
                components={{
                  pre: ({ children }) => (
                    <div className="relative group">
                      <pre className="bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg overflow-x-auto">
                        {children}
                      </pre>
                      <button
                        onClick={() => {
                          const code = (children as React.ReactNode[])?.[0]
                            ? String((children as React.ReactNode[])[0])
                            : "";
                          navigator.clipboard.writeText(code.replace(/\n$/, ""));
                        }}
                        className="absolute top-2 right-2 p-1 rounded bg-[var(--color-bg-secondary)] border border-[var(--color-border)] opacity-0 group-hover:opacity-100 transition-opacity text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
                      >
                        <Copy className="w-3 h-3" />
                      </button>
                    </div>
                  ),
                  code: ({ className, children, ...props }) => {
                    const isBlock = className?.startsWith("language-");
                    if (isBlock) {
                      return (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    }
                    return (
                      <code
                        className="bg-[var(--color-bg-tertiary)] px-1.5 py-0.5 rounded text-[var(--color-accent)]"
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {msg.content}
              </ReactMarkdown>
            ) : msg.streaming ? (
              <span className="inline-block w-2 h-4 bg-[var(--color-text-secondary)] animate-pulse" />
            ) : null}
          </div>
        )}

        {/* Footer — latency, copy */}
        {!isUser && !msg.streaming && msg.content && (
          <div className="flex items-center gap-3 mt-2 pt-2 border-t border-[var(--color-border)]">
            {msg.latencyMs != null && (
              <span className="text-xs text-[var(--color-text-secondary)]">
                {(msg.latencyMs / 1000).toFixed(1)}s
              </span>
            )}
            {msg.tokens != null && (
              <span className="text-xs text-[var(--color-text-secondary)]">
                {msg.tokens} tokens
              </span>
            )}
            {msg.logprobs && msg.logprobs.tokens.length > 0 && (
              <details className="text-xs text-[var(--color-text-secondary)]">
                <summary className="cursor-pointer hover:text-[var(--color-text-primary)]">
                  Logprobs ({msg.logprobs.tokens.length} tokens)
                </summary>
                <div className="mt-1 max-h-32 overflow-y-auto font-mono text-[10px] leading-relaxed">
                  {msg.logprobs.tokens.map((tok, i) => (
                    <span key={i} title={`logprob: ${msg.logprobs!.token_logprobs[i]?.toFixed(4) ?? "N/A"}`}>
                      {tok}{" "}
                    </span>
                  ))}
                </div>
              </details>
            )}
            <button
              onClick={copyContent}
              className="text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] flex items-center gap-1 transition-colors"
            >
              {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-[var(--color-accent)] flex items-center justify-center shrink-0 mt-1">
          <User className="w-4 h-4 text-white" />
        </div>
      )}
    </div>
  );
}

// ── Parameter Controls ──

function ParamSlider({
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
    <div>
      <div className="flex justify-between text-sm mb-1">
        <label className="text-[var(--color-text-secondary)]">{label}</label>
        <span className="font-mono">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-[var(--color-accent)]"
      />
    </div>
  );
}

function ParamInput({
  label,
  value,
  onChange,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
}) {
  return (
    <div>
      <label className="text-sm text-[var(--color-text-secondary)] block mb-1">
        {label}
      </label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value) || min)}
        className="w-full bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-1.5 text-sm"
        min={min}
        max={max}
      />
    </div>
  );
}
