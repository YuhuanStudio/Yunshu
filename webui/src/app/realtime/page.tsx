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
  Switch,
  Slider,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  SegmentedSelect,
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
  cn,
  toast,
} from "yunui";
import { PageShell } from "@/components/page-shell";
import { withTokenParam } from "@/lib/auth";
import {
  Plug,
  Unplug,
  Send,
  Mic,
  MicOff,
  Square,
  Trash2,
  ChevronDown,
  Radio,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Page-local types (per task rules — do NOT touch @/lib/types).
// ---------------------------------------------------------------------------

type ConnState = "closed" | "connecting" | "open";

/** A raw wire event in either direction, as shown in the event log. */
interface LogEntry {
  id: string;
  direction: "send" | "recv";
  type: string;
  data: unknown;
  timestamp: number;
}

/** A rendered conversation turn in the transcript. */
interface Turn {
  id: string;
  role: "user" | "assistant";
  /** Text produced by `response.text.delta` (or the user's typed text). */
  text: string;
  /** Spoken transcript from `response.audio_transcript.delta`. */
  transcript: string;
  streaming: boolean;
}

type Modalities = "text" | "text_audio";

/** Loosely-typed Realtime event — narrowed at each call site. */
type RtEvent = { type: string; [k: string]: unknown };

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"];

const MODALITY_OPTIONS = [
  { value: "text" as const, label: "Text" },
  { value: "text_audio" as const, label: "Text + Audio" },
];

/** Output PCM16 sample rate the backend streams at. */
const OUTPUT_SAMPLE_RATE = 24000;
/** Rate we down-sample the mic to before appending (server accepts pcm16). */
const INPUT_SAMPLE_RATE = 24000;

/** High-frequency streaming events kept OUT of the raw log so it stays legible. */
const NOISY_EVENTS = new Set<string>([
  "response.audio.delta",
  "response.text.delta",
  "response.audio_transcript.delta",
  "input_audio_buffer.append",
]);

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

let seq = 0;
function nextId(): string {
  seq += 1;
  return `rt-${Date.now()}-${seq}`;
}

function str(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function realtimeUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return withTokenParam(`${proto}//${window.location.host}/v1/realtime`);
}

/** base64 (of little-endian PCM16 bytes) → Int16Array. */
function base64ToInt16(b64: string): Int16Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  // Trim to an even byte count so the Int16 view is well-formed.
  const usable = bytes.length - (bytes.length % 2);
  return new Int16Array(bytes.buffer, 0, usable / 2);
}

/** Int16Array → base64 of its raw little-endian bytes. */
function int16ToBase64(int16: Int16Array): string {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** Clamp Float32 [-1,1] samples into Int16. */
function floatTo16(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/** Naive averaging down-sampler (no up-sampling). */
function downsample(buffer: Float32Array, inRate: number, outRate: number): Float32Array {
  if (outRate >= inRate) return buffer;
  const ratio = inRate / outRate;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  let iOut = 0;
  let iIn = 0;
  while (iOut < newLen) {
    const nextIn = Math.round((iOut + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = iIn; i < nextIn && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[iOut] = count > 0 ? accum / count : 0;
    iOut++;
    iIn = nextIn;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function RealtimePage() {
  const [state, setState] = useState<ConnState>("closed");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [transcript, setTranscript] = useState<Turn[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [logOpen, setLogOpen] = useState(false);

  const [draft, setDraft] = useState("");
  const [responding, setResponding] = useState(false);
  const [listening, setListening] = useState(false);
  const [micOn, setMicOn] = useState(false);

  // Session config
  const [voice, setVoice] = useState("alloy");
  const [modalities, setModalities] = useState<Modalities>("text_audio");
  const [instructions, setInstructions] = useState(
    "You are a helpful, concise voice assistant.",
  );
  const [temperature, setTemperature] = useState(0.8);
  const [vad, setVad] = useState(true);

  // --- refs (imperative, no re-render) --------------------------------------
  const wsRef = useRef<WebSocket | null>(null);
  const currentRespRef = useRef<string | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  // playback
  const playCtxRef = useRef<AudioContext | null>(null);
  const playHeadRef = useRef(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  // mic
  const micStreamRef = useRef<MediaStream | null>(null);
  const micCtxRef = useRef<AudioContext | null>(null);
  const micNodeRef = useRef<ScriptProcessorNode | null>(null);

  // config snapshot so event handlers read fresh values without re-subscribing
  const configRef = useRef({ voice, modalities, instructions, temperature, vad });
  useEffect(() => {
    configRef.current = { voice, modalities, instructions, temperature, vad };
  }, [voice, modalities, instructions, temperature, vad]);

  const connected = state === "open";

  // --- logging --------------------------------------------------------------
  const pushLog = useCallback((direction: "send" | "recv", event: RtEvent) => {
    setLog((prev) => {
      const entry: LogEntry = {
        id: nextId(),
        direction,
        type: String(event.type ?? "message"),
        data: event,
        timestamp: Date.now(),
      };
      const next = [...prev, entry];
      // Keep the log bounded so long sessions stay responsive.
      return next.length > 400 ? next.slice(next.length - 400) : next;
    });
  }, []);

  const sendEvent = useCallback(
    (event: RtEvent, opts?: { log?: boolean }): boolean => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return false;
      ws.send(JSON.stringify(event));
      if (opts?.log !== false) pushLog("send", event);
      return true;
    },
    [pushLog],
  );

  // --- audio playback -------------------------------------------------------
  const ensurePlayCtx = useCallback((): AudioContext | null => {
    if (playCtxRef.current) return playCtxRef.current;
    if (typeof window === "undefined") return null;
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctor) return null;
    let ctx: AudioContext;
    try {
      ctx = new Ctor({ sampleRate: OUTPUT_SAMPLE_RATE });
    } catch {
      ctx = new Ctor();
    }
    playCtxRef.current = ctx;
    return ctx;
  }, []);

  const playPcm16 = useCallback(
    (b64: string) => {
      const ctx = ensurePlayCtx();
      if (!ctx) return;
      if (ctx.state === "suspended") void ctx.resume();
      const int16 = base64ToInt16(b64);
      if (int16.length === 0) return;
      const f32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 0x8000;
      // Buffer is authored at 24k; if the context runs at a different rate the
      // AudioBufferSourceNode resamples it on playback.
      const buffer = ctx.createBuffer(1, f32.length, OUTPUT_SAMPLE_RATE);
      buffer.copyToChannel(f32, 0);
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      const startAt = Math.max(ctx.currentTime, playHeadRef.current);
      src.start(startAt);
      playHeadRef.current = startAt + buffer.duration;
      activeSourcesRef.current.add(src);
      src.onended = () => activeSourcesRef.current.delete(src);
    },
    [ensurePlayCtx],
  );

  const stopPlayback = useCallback(() => {
    for (const src of activeSourcesRef.current) {
      try {
        src.stop();
      } catch {
        /* already stopped */
      }
    }
    activeSourcesRef.current.clear();
    playHeadRef.current = 0;
  }, []);

  // --- server event handling ------------------------------------------------
  const buildSessionUpdate = useCallback((): RtEvent => {
    const c = configRef.current;
    return {
      type: "session.update",
      session: {
        modalities: c.modalities === "text_audio" ? ["text", "audio"] : ["text"],
        voice: c.voice,
        instructions: c.instructions,
        temperature: c.temperature,
        turn_detection: c.vad ? { type: "server_vad" } : null,
        input_audio_format: "pcm16",
        output_audio_format: "pcm16",
        max_response_output_tokens: "inf",
      },
    };
  }, []);

  const handleServerEvent = useCallback(
    (ev: RtEvent) => {
      switch (ev.type) {
        case "session.created": {
          const s = ev.session as { id?: string } | undefined;
          if (s?.id) setSessionId(s.id);
          // Push our config as soon as the session exists.
          sendEvent(buildSessionUpdate());
          break;
        }
        case "session.updated": {
          const s = ev.session as { id?: string } | undefined;
          if (s?.id) setSessionId(s.id);
          break;
        }
        case "conversation.created": {
          const c = ev.conversation as { id?: string } | undefined;
          if (c?.id) setConversationId(c.id);
          break;
        }
        case "response.created": {
          const r = ev.response as { id?: string } | undefined;
          const id = r?.id ?? nextId();
          currentRespRef.current = id;
          setResponding(true);
          setTranscript((prev) => [
            ...prev,
            { id, role: "assistant", text: "", transcript: "", streaming: true },
          ]);
          break;
        }
        case "response.text.delta": {
          const d = str(ev.delta);
          const id = currentRespRef.current;
          if (id && d) {
            setTranscript((prev) =>
              prev.map((m) => (m.id === id ? { ...m, text: m.text + d } : m)),
            );
          }
          break;
        }
        case "response.audio_transcript.delta": {
          const d = str(ev.delta);
          const id = currentRespRef.current;
          if (id && d) {
            setTranscript((prev) =>
              prev.map((m) =>
                m.id === id ? { ...m, transcript: m.transcript + d } : m,
              ),
            );
          }
          break;
        }
        case "response.audio.delta": {
          const d = str(ev.delta);
          if (d) playPcm16(d);
          break;
        }
        case "response.done": {
          const id = currentRespRef.current;
          if (id) {
            setTranscript((prev) =>
              prev.map((m) => (m.id === id ? { ...m, streaming: false } : m)),
            );
          }
          currentRespRef.current = null;
          setResponding(false);
          break;
        }
        case "input_audio_buffer.speech_started":
          setListening(true);
          break;
        case "input_audio_buffer.speech_stopped":
        case "input_audio_buffer.committed":
          setListening(false);
          break;
        case "conversation.item.input_audio_transcription.completed": {
          const tr = str(ev.transcript);
          if (tr) {
            setTranscript((prev) => [
              ...prev,
              { id: nextId(), role: "user", text: tr, transcript: "", streaming: false },
            ]);
          }
          break;
        }
        case "error": {
          const e = ev.error as { message?: string } | undefined;
          setErrorMsg(e?.message ?? "Realtime error");
          setResponding(false);
          break;
        }
        default:
          break;
      }
    },
    [buildSessionUpdate, playPcm16, sendEvent],
  );

  // --- connect / disconnect -------------------------------------------------
  const disconnectMic = useCallback(() => {
    micNodeRef.current?.disconnect();
    micNodeRef.current = null;
    micStreamRef.current?.getTracks().forEach((t) => t.stop());
    micStreamRef.current = null;
    if (micCtxRef.current) {
      void micCtxRef.current.close().catch(() => {});
      micCtxRef.current = null;
    }
    setMicOn(false);
    setListening(false);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current) return;
    setState("connecting");
    setErrorMsg(null);
    let ws: WebSocket;
    try {
      ws = new WebSocket(realtimeUrl());
    } catch {
      setState("closed");
      toast.error("Could not open the realtime socket");
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => setState("open");
    ws.onerror = () => {
      toast.error("Realtime socket error");
    };
    ws.onclose = () => {
      wsRef.current = null;
      setState("closed");
      setResponding(false);
    };
    ws.onmessage = (msgEv) => {
      let parsed: RtEvent;
      try {
        parsed = JSON.parse(msgEv.data as string) as RtEvent;
      } catch {
        return;
      }
      if (!NOISY_EVENTS.has(parsed.type)) pushLog("recv", parsed);
      handleServerEvent(parsed);
    };
  }, [handleServerEvent, pushLog]);

  const disconnect = useCallback(() => {
    disconnectMic();
    stopPlayback();
    wsRef.current?.close();
    wsRef.current = null;
    setState("closed");
    setSessionId(null);
    setConversationId(null);
    setResponding(false);
  }, [disconnectMic, stopPlayback]);

  // Full cleanup on unmount.
  useEffect(() => {
    return () => {
      disconnectMic();
      wsRef.current?.close();
      wsRef.current = null;
      if (playCtxRef.current) {
        void playCtxRef.current.close().catch(() => {});
        playCtxRef.current = null;
      }
    };
  }, [disconnectMic]);

  // Auto-scroll the transcript.
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ block: "end" });
  }, [transcript]);

  // --- actions --------------------------------------------------------------
  const applyConfig = useCallback(() => {
    if (sendEvent(buildSessionUpdate())) {
      toast.success("Session updated");
    }
  }, [buildSessionUpdate, sendEvent]);

  const sendText = useCallback(() => {
    const text = draft.trim();
    if (!text || !connected) return;
    const ok = sendEvent({
      type: "conversation.item.create",
      item: { type: "message", role: "user", content: [{ type: "input_text", text }] },
    });
    if (!ok) return;
    sendEvent({ type: "response.create" });
    setTranscript((prev) => [
      ...prev,
      { id: nextId(), role: "user", text, transcript: "", streaming: false },
    ]);
    setDraft("");
  }, [draft, connected, sendEvent]);

  const cancelResponse = useCallback(() => {
    sendEvent({ type: "response.cancel" });
    stopPlayback();
    setResponding(false);
  }, [sendEvent, stopPlayback]);

  const startMic = useCallback(async () => {
    if (!connected || typeof navigator === "undefined" || !navigator.mediaDevices) {
      toast.error("Microphone unavailable");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      micStreamRef.current = stream;
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!Ctor) throw new Error("no AudioContext");
      const ctx = new Ctor();
      micCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      micNodeRef.current = processor;
      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const down = downsample(input, ctx.sampleRate, INPUT_SAMPLE_RATE);
        const int16 = floatTo16(down);
        // Append silently — logging every ~85ms chunk would drown the log.
        sendEvent(
          { type: "input_audio_buffer.append", audio: int16ToBase64(int16) },
          { log: false },
        );
      };
      // Route through a muted gain node so the processor runs without echoing
      // the mic back out of the speakers.
      const mute = ctx.createGain();
      mute.gain.value = 0;
      source.connect(processor);
      processor.connect(mute);
      mute.connect(ctx.destination);
      setMicOn(true);
    } catch {
      disconnectMic();
      toast.error("Microphone permission denied");
    }
  }, [connected, sendEvent, disconnectMic]);

  const stopMic = useCallback(() => {
    // In manual mode, commit what we captured and ask for a response.
    if (!configRef.current.vad) {
      sendEvent({ type: "input_audio_buffer.commit" });
      sendEvent({ type: "response.create" });
    }
    disconnectMic();
  }, [sendEvent, disconnectMic]);

  const commitAudio = useCallback(() => {
    sendEvent({ type: "input_audio_buffer.commit" });
    sendEvent({ type: "response.create" });
  }, [sendEvent]);

  const clearTranscript = useCallback(() => {
    setTranscript([]);
    stopPlayback();
  }, [stopPlayback]);

  // --- derived --------------------------------------------------------------
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
      description="OpenAI-Realtime WebSocket console — text, streaming audio and mic."
      width="wide"
    >
      {errorMsg && (
        <Alert
          variant="error"
          title="Realtime error"
          className="mb-4"
        >
          <div className="flex items-center justify-between gap-3">
            <span className="min-w-0 break-words">{errorMsg}</span>
            <Button variant="ghost" size="sm" onClick={() => setErrorMsg(null)}>
              Dismiss
            </Button>
          </div>
        </Alert>
      )}

      {/* Connection bar */}
      <Card className="mb-6 flex flex-wrap items-center gap-3 p-4">
        <StatusIndicator status={statusFor[state]} pulse={connected}>
          <span className="text-sm">{statusLabel[state]}</span>
        </StatusIndicator>
        {sessionId && (
          <Badge variant="info" className="font-mono text-xs">
            {sessionId}
          </Badge>
        )}
        {conversationId && (
          <Badge variant="default" className="font-mono text-xs">
            {conversationId}
          </Badge>
        )}
        {listening && (
          <Badge variant="success">
            <Radio className="mr-1 inline size-3 animate-pulse" />
            Listening
          </Badge>
        )}
        <div className="ml-auto flex items-center gap-2">
          {connected ? (
            <Button variant="secondary" size="sm" onClick={disconnect}>
              <Unplug className="mr-1.5 size-4" />
              Disconnect
            </Button>
          ) : (
            <Button size="sm" onClick={connect} disabled={state === "connecting"}>
              <Plug className="mr-1.5 size-4" />
              Connect
            </Button>
          )}
        </div>
      </Card>

      <div className="flex flex-col gap-6 lg:flex-row">
        {/* Transcript + composer */}
        <div className="flex min-w-0 flex-1 flex-col gap-4">
          <Card className="p-0">
            <div
              className="overflow-y-auto p-4"
              style={{ maxHeight: "32rem", minHeight: "18rem" }}
            >
              {transcript.length === 0 ? (
                <div className="py-10">
                  <EmptyState
                    title="No turns yet"
                    description="Connect, then type a message or hold the mic to start talking."
                  />
                </div>
              ) : (
                <ul className="flex flex-col gap-3">
                  {transcript.map((t) => {
                    const isUser = t.role === "user";
                    const body = t.text || t.transcript;
                    return (
                      <li
                        key={t.id}
                        className={cn("flex", isUser ? "justify-end" : "justify-start")}
                      >
                        <div
                          className={cn(
                            "max-w-[85%] rounded-lg px-3 py-2 text-sm",
                            isUser
                              ? "bg-accent text-accent-foreground"
                              : "bg-muted text-foreground",
                          )}
                        >
                          <div className="mb-0.5 text-xs font-medium opacity-70">
                            {isUser ? "You" : "Assistant"}
                          </div>
                          <div className="whitespace-pre-wrap break-words">
                            {body || (t.streaming ? "…" : "")}
                            {t.streaming && body && (
                              <span className="ml-0.5 inline-block animate-pulse">▋</span>
                            )}
                          </div>
                          {t.text && t.transcript && (
                            <div className="mt-1 border-t border-current/10 pt-1 text-xs opacity-60">
                              🔊 {t.transcript}
                            </div>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
              <div ref={transcriptEndRef} />
            </div>
          </Card>

          {/* Composer */}
          <Card className="flex flex-col gap-3 p-4">
            <Textarea
              rows={3}
              placeholder="Type a message…"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  sendText();
                }
              }}
              disabled={!connected}
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                onClick={sendText}
                disabled={!connected || !draft.trim()}
              >
                <Send className="mr-1.5 size-4" />
                Send
              </Button>
              {responding && (
                <Button variant="secondary" size="sm" onClick={cancelResponse}>
                  <Square className="mr-1.5 size-4" />
                  Cancel
                </Button>
              )}
              <div className="ml-auto flex items-center gap-2">
                {micOn ? (
                  <Button variant="secondary" size="sm" onClick={stopMic}>
                    <MicOff className="mr-1.5 size-4" />
                    Stop mic
                  </Button>
                ) : (
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => void startMic()}
                    disabled={!connected}
                  >
                    <Mic className="mr-1.5 size-4" />
                    Mic
                  </Button>
                )}
                {micOn && !vad && (
                  <Button size="sm" onClick={commitAudio}>
                    Commit + respond
                  </Button>
                )}
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              {vad
                ? "Server VAD is on — the model replies automatically when you stop speaking."
                : "Manual turns — stop the mic (or press Commit) to send captured audio."}{" "}
              Cmd/Ctrl+Enter sends text.
            </p>
          </Card>
        </div>

        {/* Session config */}
        <div className="flex w-full shrink-0 flex-col gap-4 lg:w-80">
          <Card className="flex flex-col gap-4 p-4">
            <span className="text-sm font-semibold">Session</span>

            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">Modalities</label>
              <SegmentedSelect<Modalities>
                options={MODALITY_OPTIONS}
                value={modalities}
                onChange={setModalities}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">Voice</label>
              <Select value={voice} onValueChange={setVoice}>
                <SelectTrigger aria-label="Voice">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {VOICES.map((v) => (
                    <SelectItem key={v} value={v}>
                      {v}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">Instructions</label>
              <Textarea aria-label="Instructions"
                rows={4}
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
              />
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-muted-foreground">Temperature</span>
                <Badge variant="info">{temperature.toFixed(2)}</Badge>
              </div>
              <Slider label="Temperature"
                value={[temperature]}
                onValueChange={(v) => setTemperature(v[0] ?? 0.8)}
                min={0.6}
                max={1.2}
                step={0.05}
              />
            </div>

            <label className="flex items-center justify-between gap-2 text-sm">
              <span>
                <span className="font-medium">Server VAD</span>
                <span className="block text-xs text-muted-foreground">
                  Auto-detect end of speech
                </span>
              </span>
              <Switch label="Auto-detect end of speech" checked={vad} onCheckedChange={setVad} />
            </label>

            <Button
              variant="secondary"
              size="sm"
              onClick={applyConfig}
              disabled={!connected}
            >
              Apply session
            </Button>
          </Card>

          <Card className="flex flex-col gap-2 p-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">Transcript</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearTranscript}
                disabled={transcript.length === 0}
              >
                <Trash2 className="mr-1 size-3.5" />
                Clear
              </Button>
            </div>
          </Card>
        </div>
      </div>

      {/* Raw event log */}
      <Card className="mt-6 p-0">
        <Collapsible open={logOpen} onOpenChange={setLogOpen}>
          <CollapsibleTrigger
            className={cn(
              "flex w-full items-center gap-2 rounded-md px-4 py-3 text-left text-sm font-medium",
              "hover:bg-muted/60",
            )}
          >
            <ChevronDown
              className={cn(
                "size-4 transition-transform",
                logOpen ? "rotate-0" : "-rotate-90",
              )}
            />
            <span>Raw event log</span>
            <Badge variant="default" className="ml-1">
              {log.length}
            </Badge>
            <span className="ml-auto text-xs font-normal text-muted-foreground">
              streaming deltas omitted
            </span>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div
              className="overflow-y-auto border-t border-border p-2 font-mono text-xs"
              style={{ maxHeight: "24rem" }}
            >
              {log.length === 0 ? (
                <div className="px-2 py-6 text-center text-muted-foreground">
                  No events yet.
                </div>
              ) : (
                <ul className="flex flex-col gap-1">
                  {log.map((m) => {
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
          </CollapsibleContent>
        </Collapsible>
      </Card>
    </PageShell>
  );
}
