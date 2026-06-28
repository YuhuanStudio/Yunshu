"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Radio,
  Mic,
  MicOff,
  Send,
  Loader2,
  X,
  Volume2,
  Settings2,
} from "lucide-react";

interface WsMessage {
  id: string;
  direction: "send" | "recv";
  type: string;
  data: unknown;
  timestamp: number;
}

export default function RealtimePage() {
  // WebSocket must connect to the backend directly (Next.js rewrites don't proxy WS).
  // Use NEXT_PUBLIC_BACKEND_URL if set, otherwise derive from current location
  // (works when WebUI is served by the backend itself, not Next.js dev server).
  const backendHost = typeof window !== "undefined"
    ? (process.env.NEXT_PUBLIC_BACKEND_URL
        ? new URL(process.env.NEXT_PUBLIC_BACKEND_URL).host
        : window.location.port === "3000"
          ? `${window.location.hostname}:8000`  // Next.js dev server → backend
          : window.location.host)               // Production (WebUI served by backend)
    : "localhost:8000";
  const defaultWsUrl = `ws://${backendHost}/realtime`;
  const [url, setUrl] = useState(defaultWsUrl);
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [messages, setMessages] = useState<WsMessage[]>([]);
  const [inputJson, setInputJson] = useState('{\n  "type": "session.update",\n  "session": {\n    "model": "default"\n  }\n}');
  const [autoScroll, setAutoScroll] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const msgIdRef = useRef(0);

  useEffect(() => {
    if (autoScroll) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, autoScroll]);

  // Close the WebSocket on unmount so navigating away does not leak the
  // connection or fire setMessages on an unmounted component (Wave 431 fix).
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        try { wsRef.current.close(); } catch { /* ignore */ }
        wsRef.current = null;
      }
    };
  }, []);

  const addMsg = useCallback((direction: WsMessage["direction"], type: string, data: unknown) => {
    const msg: WsMessage = {
      id: `ws-${++msgIdRef.current}`,
      direction,
      type,
      data,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, msg]);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnecting(true);
    setError(null);
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      setConnecting(false);
      addMsg("recv", "system", { event: "connected" });
    };

    ws.onclose = (e) => {
      setConnected(false);
      setConnecting(false);
      addMsg("recv", "system", { event: "disconnected", code: e.code, reason: e.reason });
      wsRef.current = null;
    };

    ws.onerror = () => {
      setError("WebSocket connection failed");
      setConnecting(false);
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        addMsg("recv", data.type || "message", data);
      } catch {
        addMsg("recv", "raw", e.data);
      }
    };

    wsRef.current = ws;
  }, [url, addMsg]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
  }, []);

  const sendJson = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError("Not connected");
      return;
    }

    try {
      const parsed = JSON.parse(inputJson);
      const type = parsed.type || "message";
      ws.send(inputJson);
      addMsg("send", type, parsed);
      setError(null);
    } catch (err) {
      setError(`Invalid JSON: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [inputJson, addMsg]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    msgIdRef.current = 0;
  }, []);

  return (
    <div className="flex h-full">
      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="border-b border-[var(--color-border)] px-4 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <Radio className="w-5 h-5 text-[var(--color-accent)]" />
            <h2 className="text-lg font-bold">Realtime</h2>
            <span
              className={`text-xs px-2 py-0.5 rounded-full ${
                connected
                  ? "bg-[var(--color-success)]/15 text-[var(--color-success)]"
                  : connecting
                    ? "bg-[var(--color-warning)]/15 text-[var(--color-warning)]"
                    : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
              }`}
            >
              {connected ? "Connected" : connecting ? "Connecting..." : "Disconnected"}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {connected ? (
              <button
                onClick={disconnect}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-[var(--color-danger)]/15 text-[var(--color-danger)] hover:bg-[var(--color-danger)]/25 transition-colors"
              >
                <MicOff className="w-3.5 h-3.5" />
                Disconnect
              </button>
            ) : (
              <button
                onClick={connect}
                disabled={connecting}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-[var(--color-accent)] text-white hover:bg-[var(--color-accent-hover)] disabled:opacity-50 transition-colors"
              >
                {connecting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Mic className="w-3.5 h-3.5" />}
                Connect
              </button>
            )}
            <button
              onClick={clearMessages}
              className="p-1.5 rounded-lg text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] transition-colors"
              title="Clear messages"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Messages log */}
        <div className="flex-1 overflow-auto font-mono text-xs">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-[var(--color-text-secondary)]">
              <div className="text-center max-w-sm">
                <Volume2 className="w-10 h-10 mx-auto mb-3 opacity-30" />
                <p>Connect to a Realtime WebSocket endpoint to send and receive messages.</p>
              </div>
            </div>
          ) : (
            <div className="p-4 space-y-2">
              {messages.map((msg) => (
                <WsMessageRow key={msg.id} msg={msg} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-[var(--color-border)] p-4 shrink-0">
          {error && (
            <div className="mb-2 text-xs text-[var(--color-danger)] bg-[var(--color-danger)]/10 rounded-lg px-3 py-2 flex items-center gap-2">
              <X className="w-3 h-3 shrink-0" />
              {error}
              <button onClick={() => setError(null)} className="ml-auto opacity-60 hover:opacity-100">
                <X className="w-3 h-3" />
              </button>
            </div>
          )}
          <div className="flex gap-2">
            <textarea
              value={inputJson}
              onChange={(e) => setInputJson(e.target.value)}
              rows={4}
              spellCheck={false}
              className="flex-1 bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl px-4 py-3 text-xs font-mono resize-none focus:outline-none focus:border-[var(--color-accent)]"
              placeholder="JSON message to send..."
              disabled={!connected}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  sendJson();
                }
              }}
            />
            <button
              onClick={sendJson}
              disabled={!connected}
              className="self-end p-2.5 rounded-lg bg-[var(--color-accent)] text-white hover:bg-[var(--color-accent-hover)] disabled:opacity-30 transition-colors"
              title="Send (Cmd+Enter)"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <div className="mt-1 text-[10px] text-[var(--color-text-secondary)]">
            Cmd+Enter to send
          </div>
        </div>
      </div>

      {/* Config sidebar */}
      <div className="w-72 border-l border-[var(--color-border)] p-4 space-y-5 shrink-0 overflow-auto">
        <h3 className="font-semibold text-sm flex items-center gap-2">
          <Settings2 className="w-4 h-4" />
          Connection
        </h3>

        <div>
          <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wide block mb-1">
            WebSocket URL
          </label>
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={connected}
            className="w-full bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-[var(--color-accent)] disabled:opacity-50"
          />
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="autoScroll"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />
          <label htmlFor="autoScroll" className="text-sm">Auto-scroll</label>
        </div>

        <div className="pt-4 border-t border-[var(--color-border)]">
          <h4 className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">
            Message Templates
          </h4>
          <div className="space-y-1.5">
            <TemplateButton
              label="Session Update"
              onClick={() =>
                setInputJson(
                  JSON.stringify(
                    { type: "session.update", session: { model: "default" } },
                    null,
                    2
                  )
                )
              }
            />
            <TemplateButton
              label="Create Response"
              onClick={() =>
                setInputJson(
                  JSON.stringify(
                    {
                      type: "response.create",
                      response: {
                        modalities: ["text"],
                        input: [{ type: "input_text", text: "Hello!" }],
                      },
                    },
                    null,
                    2
                  )
                )
              }
            />
            <TemplateButton
              label="Cancel Response"
              onClick={() =>
                setInputJson(JSON.stringify({ type: "response.cancel" }, null, 2))
              }
            />
          </div>
        </div>

        <div className="pt-4 border-t border-[var(--color-border)]">
          <h4 className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">
            Stats
          </h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <div className="text-xs text-[var(--color-text-secondary)]">Sent</div>
              <div className="font-medium tabular-nums">
                {messages.filter((m) => m.direction === "send").length}
              </div>
            </div>
            <div>
              <div className="text-xs text-[var(--color-text-secondary)]">Received</div>
              <div className="font-medium tabular-nums">
                {messages.filter((m) => m.direction === "recv").length}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function WsMessageRow({ msg }: { msg: WsMessage }) {
  const [expanded, setExpanded] = useState(false);
  const isSend = msg.direction === "send";
  const time = new Date(msg.timestamp).toLocaleTimeString();

  return (
    <div className={`rounded-lg border ${
      isSend
        ? "border-[var(--color-accent)]/30 bg-[var(--color-accent)]/5"
        : "border-[var(--color-border)] bg-[var(--color-bg-secondary)]"
    }`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left"
      >
        <span className={`text-[10px] font-bold uppercase ${isSend ? "text-[var(--color-accent)]" : "text-[var(--color-success)]"}`}>
          {isSend ? "→" : "←"}
        </span>
        <span className="text-[var(--color-accent)]">{msg.type}</span>
        <span className="ml-auto text-[10px] text-[var(--color-text-secondary)]">{time}</span>
      </button>
      {expanded && (
        <div className="px-3 pb-2 border-t border-[var(--color-border)]">
          <pre className="mt-2 whitespace-pre-wrap break-all text-[var(--color-text-secondary)]">
            {typeof msg.data === "string" ? msg.data : JSON.stringify(msg.data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

function TemplateButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left text-xs px-3 py-1.5 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] hover:bg-[var(--color-bg-secondary)] transition-colors"
    >
      {label}
    </button>
  );
}
