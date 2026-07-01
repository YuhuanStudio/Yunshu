"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Button,
  Input,
  Textarea,
  Card,
  Badge,
  Alert,
  EmptyState,
  StatusIndicator,
  Checkbox,
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
  cn,
  toast,
} from "yunui";
import { PageShell } from "@/components/page-shell";
import type { WsMessage } from "@/lib/types";

type ConnState = "closed" | "connecting" | "open";

const TEMPLATES: { label: string; body: string }[] = [
  {
    label: "session.update",
    body: JSON.stringify({ type: "session.update", session: { model: "" } }, null, 2),
  },
  {
    label: "response.create",
    body: JSON.stringify(
      { type: "response.create", response: { modalities: ["text"], input: [] } },
      null,
      2,
    ),
  },
  {
    label: "response.cancel",
    body: JSON.stringify({ type: "response.cancel" }, null, 2),
  },
];

function defaultWsUrl(): string {
  if (typeof window === "undefined") return "ws://localhost/realtime";
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/realtime`;
}

let msgSeq = 0;
function nextId(): string {
  msgSeq += 1;
  return `ws-${Date.now()}-${msgSeq}`;
}

function typeOf(parsed: unknown): string {
  if (parsed && typeof parsed === "object" && "type" in parsed) {
    return String((parsed as { type: unknown }).type ?? "message");
  }
  return "message";
}

export default function RealtimePage() {
  const [url, setUrl] = useState("");
  const [state, setState] = useState<ConnState>("closed");
  const [messages, setMessages] = useState<WsMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);

  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  // Seed the URL from the current origin once mounted (window is client-only).
  useEffect(() => {
    setUrl(defaultWsUrl());
  }, []);

  const push = useCallback((msg: WsMessage) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const connected = state === "open";

  const connect = useCallback(() => {
    if (wsRef.current) return;
    setState("connecting");
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      setState("closed");
      toast.error("Invalid WebSocket URL");
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => setState("open");
    ws.onerror = () => {
      setState("closed");
      toast.error("WebSocket error");
    };
    ws.onclose = () => {
      wsRef.current = null;
      setState("closed");
    };
    ws.onmessage = (ev) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(ev.data as string);
      } catch {
        parsed = { type: "message", raw: ev.data };
      }
      push({
        id: nextId(),
        direction: "recv",
        type: typeOf(parsed),
        data: parsed,
        timestamp: Date.now(),
      });
    };
  }, [url, push]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setState("closed");
  }, []);

  // Tear the socket down on unmount.
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  // Auto-scroll to the newest message.
  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [messages, autoScroll]);

  const send = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || state !== "open") return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(draft);
    } catch {
      toast.error("Message is not valid JSON");
      return;
    }
    ws.send(JSON.stringify(parsed));
    push({
      id: nextId(),
      direction: "send",
      type: typeOf(parsed),
      data: parsed,
      timestamp: Date.now(),
    });
  }, [draft, state, push]);

  const draftIsValid = (() => {
    if (!draft.trim()) return false;
    try {
      JSON.parse(draft);
      return true;
    } catch {
      return false;
    }
  })();

  const sentCount = messages.filter((m) => m.direction === "send").length;
  const recvCount = messages.filter((m) => m.direction === "recv").length;

  const statusFor: Record<ConnState, "online" | "away" | "offline"> = {
    open: "online",
    connecting: "away",
    closed: "offline",
  };
  const statusLabel: Record<ConnState, string> = {
    open: "Connected",
    connecting: "Connecting…",
    closed: "Disconnected",
  };

  return (
    <PageShell
      title="Realtime"
      description="WebSocket playground for the realtime API."
      width="wide"
    >
      <div className="flex flex-col gap-6 lg:flex-row">
        {/* Message log */}
        <div className="flex min-w-0 flex-1 flex-col gap-4">
          <div className="flex items-center gap-3">
            <StatusIndicator status={statusFor[state]} pulse={connected}>
              <span className="text-sm">{statusLabel[state]}</span>
            </StatusIndicator>
            <div className="ml-auto flex items-center gap-2">
              {connected ? (
                <Button variant="secondary" size="sm" onClick={disconnect}>
                  Disconnect
                </Button>
              ) : (
                <Button
                  size="sm"
                  onClick={connect}
                  disabled={state === "connecting" || !url.trim()}
                >
                  Connect
                </Button>
              )}
            </div>
          </div>

          <Card className="p-0">
            <div
              ref={logRef}
              className="max-h-[28rem] min-h-[16rem] overflow-y-auto p-2 font-mono text-xs"
            >
              {messages.length === 0 ? (
                <div className="py-10">
                  <EmptyState
                    title="No messages yet"
                    description="Connect and send a frame to see traffic here."
                  />
                </div>
              ) : (
                <ul className="flex flex-col gap-1">
                  {messages.map((m) => {
                    const isSend = m.direction === "send";
                    return (
                      <li key={m.id}>
                        <Collapsible>
                          <CollapsibleTrigger
                            className={cn(
                              "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left",
                              "hover:bg-muted/60",
                            )}
                          >
                            <span
                              className={cn(
                                "font-semibold",
                                isSend ? "text-accent" : "text-success",
                              )}
                              aria-hidden
                            >
                              {isSend ? "→" : "←"}
                            </span>
                            <span className="min-w-0 flex-1 truncate">{m.type}</span>
                            <span className="shrink-0 text-muted-foreground">
                              {new Date(m.timestamp).toLocaleTimeString()}
                            </span>
                          </CollapsibleTrigger>
                          <CollapsibleContent>
                            <pre className="mt-1 overflow-x-auto rounded-md bg-muted/50 p-2 text-muted-foreground">
                              {JSON.stringify(m.data, null, 2)}
                            </pre>
                          </CollapsibleContent>
                        </Collapsible>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </Card>

          {/* Send area */}
          <Card className="flex flex-col gap-3 p-4">
            <Textarea
              className="font-mono text-xs"
              rows={6}
              placeholder='{"type":"response.create"}'
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
            />
            {draft.trim() && !draftIsValid && (
              <Alert variant="error" title="Invalid JSON">
                Fix the payload before sending.
              </Alert>
            )}
            <div className="flex items-center justify-end">
              <Button size="sm" onClick={send} disabled={!connected || !draftIsValid}>
                Send
              </Button>
            </div>
          </Card>
        </div>

        {/* Config panel */}
        <div className="flex w-full shrink-0 flex-col gap-4 lg:w-72">
          <Card className="flex flex-col gap-4 p-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                WebSocket URL
              </label>
              <Input
                className="font-mono text-xs"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                disabled={connected || state === "connecting"}
                placeholder="ws://host/realtime"
              />
            </div>

            <label className="flex items-center gap-2 text-sm">
              <Checkbox checked={autoScroll} onCheckedChange={setAutoScroll} />
              <span>Auto-scroll</span>
            </label>
          </Card>

          <Card className="flex flex-col gap-2 p-4">
            <span className="text-xs font-medium text-muted-foreground">Templates</span>
            {TEMPLATES.map((t) => (
              <Button
                key={t.label}
                variant="secondary"
                size="sm"
                className="justify-start font-mono text-xs"
                onClick={() => setDraft(t.body)}
              >
                {t.label}
              </Button>
            ))}
          </Card>

          <Card className="p-4">
            <span className="text-xs font-medium text-muted-foreground">Stats</span>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <span className="text-2xl font-semibold text-accent">{sentCount}</span>
                <span className="text-xs text-muted-foreground">Sent</span>
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-2xl font-semibold text-success">{recvCount}</span>
                <span className="text-xs text-muted-foreground">Received</span>
              </div>
            </div>
            <div className="mt-3">
              <Badge variant={connected ? "success" : "default"}>
                {connected ? "Live" : "Idle"}
              </Badge>
            </div>
          </Card>
        </div>
      </div>
    </PageShell>
  );
}
