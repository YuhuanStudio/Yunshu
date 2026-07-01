"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  Button,
  Input,
  Textarea,
  Card,
  Badge,
  Alert,
  IconButton,
  Spinner,
  Slider,
  CustomSelect,
  FileDropzone,
  cn,
  toast,
} from "yunui";
import { Mic, AudioLines, Square, X, Image as ImageIcon, FileAudio, Volume2 } from "lucide-react";
import { streamSSE, ApiError, type SSEChunk } from "@/lib/api";
import { fmtBytes } from "@/lib/format";
import { PageShell } from "@/components/page-shell";

// Qwen3-Omni ships a small roster of named speakers.
const SPEAKERS = ["Ethan", "Chelsie", "Aiden"];
const SAMPLE_RATE = 24000;

// ---- binary helpers --------------------------------------------------------
/** Base64 → Uint8Array. */
function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** File → `data:` URI (base64), the form the backend accepts for image/audio. */
function fileToDataUri(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/** Wrap raw PCM16 mono samples in a 44-byte WAV container (for a replayable clip). */
function pcmToWav(pcm: Uint8Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + pcm.length);
  const view = new DataView(buffer);
  const w = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
  };
  w(0, "RIFF");
  view.setUint32(4, 36 + pcm.length, true);
  w(8, "WAVE");
  w(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  w(36, "data");
  view.setUint32(40, pcm.length, true);
  new Uint8Array(buffer, 44).set(pcm);
  return new Blob([buffer], { type: "audio/wav" });
}

function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">
        {label}
        {hint && <span className="ml-1 font-normal text-muted-foreground">{hint}</span>}
      </label>
      {children}
    </div>
  );
}

export default function OmniPage() {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState("Ethan");
  const [thinkerMaxTokens, setThinkerMaxTokens] = useState(256);

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);

  const [running, setRunning] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  audioUrlRef.current = audioUrl;

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      sourceRef.current?.stop();
      audioCtxRef.current?.close().catch(() => {});
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    };
  }, []);

  const describe = (e: unknown) =>
    e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Something went wrong.";

  /** Concatenate PCM16 chunks, decode to Float32, and play via WebAudio. */
  const playPcm = async (chunks: Uint8Array[], sampleRate: number): Promise<Blob | null> => {
    const total = chunks.reduce((n, c) => n + c.length, 0);
    if (total < 2) return null;
    const pcm = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) {
      pcm.set(c, off);
      off += c.length;
    }
    // Interpret the little-endian PCM16 bytes as Int16, then normalise to [-1, 1).
    const sampleCount = Math.floor(pcm.length / 2);
    const view = new DataView(pcm.buffer, pcm.byteOffset, sampleCount * 2);
    const f32 = new Float32Array(sampleCount);
    for (let i = 0; i < sampleCount; i++) f32[i] = view.getInt16(i * 2, true) / 32768;

    const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new AC({ sampleRate });
    audioCtxRef.current = ctx;
    const buffer = ctx.createBuffer(1, sampleCount, sampleRate);
    buffer.copyToChannel(f32, 0);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    sourceRef.current = src;
    src.onended = () => {
      if (sourceRef.current === src) sourceRef.current = null;
    };
    await ctx.resume().catch(() => {});
    src.start();

    return pcmToWav(pcm.subarray(0, sampleCount * 2), sampleRate);
  };

  const submit = async () => {
    const value = text.trim();
    if (!value || running) return;
    setRunning(true);
    setError(null);
    setTranscript("");
    // Drop any previous clip.
    sourceRef.current?.stop();
    sourceRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);

    const controller = new AbortController();
    abortRef.current = controller;

    const chunks: Uint8Array[] = [];
    let sampleRate = SAMPLE_RATE;
    let live = "";

    try {
      const body: Record<string, unknown> = {
        text: value,
        speaker,
        thinker_max_new_tokens: thinkerMaxTokens,
      };
      if (imageFile) body.image_path = await fileToDataUri(imageFile);
      if (audioFile) body.audio_path = await fileToDataUri(audioFile);

      await streamSSE(
        "/v1/omni/speech/stream",
        body,
        (d: SSEChunk) => {
          const type = d.type as string | undefined;
          if (type === "text") {
            if (typeof d.delta === "string") {
              live += d.delta;
              setTranscript(live);
            }
          } else if (type === "audio") {
            if (typeof d.delta === "string") chunks.push(b64ToBytes(d.delta));
            if (typeof d.sr === "number") sampleRate = d.sr;
          } else if (type === "error") {
            throw new ApiError(String(d.message ?? "Omni stream error"), 500);
          }
          // type "done" needs no handling; finalisation happens after [DONE].
        },
        controller.signal,
      );

      // Concat-on-done: assemble the PCM16 stream and play it back.
      const wav = await playPcm(chunks, sampleRate);
      if (wav) {
        const url = URL.createObjectURL(wav);
        setAudioUrl(url);
      }
    } catch (e) {
      if (!controller.signal.aborted) {
        const msg = describe(e);
        setError(msg);
        toast.error("Omni failed", msg);
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    sourceRef.current?.stop();
    sourceRef.current = null;
    setRunning(false);
  };

  const fileRow = (file: File | null, onClear: () => void, Icon: typeof FileAudio) =>
    file && (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Icon className="h-4 w-4 shrink-0" />
        <span className="min-w-0 flex-1 truncate">{file.name}</span>
        <Badge variant="info">{fmtBytes(file.size)}</Badge>
        <IconButton icon={<X className="h-3.5 w-3.5" />} label="Remove" onClick={onClear} className="p-1" />
      </div>
    );

  return (
    <PageShell
      title="Omni"
      description="Qwen3-Omni voice: stream a spoken reply (text + PCM16 audio) from text, optionally grounded on an image or audio clip."
      width="narrow"
    >
      <div className="space-y-6">
        {error && (
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
        )}

        <Card className="space-y-4 p-5">
          <Field label="Text" hint="(1–8000 chars)">
            <Textarea
              rows={4}
              maxLength={8000}
              placeholder="What should the model say?"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </Field>

          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Speaker">
              <CustomSelect
                options={SPEAKERS.map((s) => ({ value: s, label: s }))}
                value={speaker}
                onChange={setSpeaker}
              />
            </Field>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <label className="font-medium">Thinker max new tokens</label>
                <Badge variant="info">{thinkerMaxTokens}</Badge>
              </div>
              <Slider
                value={[thinkerMaxTokens]}
                onValueChange={(v) => setThinkerMaxTokens(v[0] ?? 0)}
                min={0}
                max={512}
                step={8}
              />
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Image" hint="(optional grounding)">
              <FileDropzone
                accept="image/*"
                icon={<ImageIcon className="h-6 w-6" />}
                label="Drop an image"
                hint="Sent as a data URI"
                onFiles={(files) => {
                  if (files[0]) setImageFile(files[0]);
                }}
              />
              {fileRow(imageFile, () => setImageFile(null), ImageIcon)}
            </Field>
            <Field label="Audio" hint="(optional grounding)">
              <FileDropzone
                accept="audio/*"
                icon={<FileAudio className="h-6 w-6" />}
                label="Drop an audio clip"
                hint="Sent as a data URI"
                onFiles={(files) => {
                  if (files[0]) setAudioFile(files[0]);
                }}
              />
              {fileRow(audioFile, () => setAudioFile(null), FileAudio)}
            </Field>
          </div>

          <div className="flex justify-end gap-2">
            {running && (
              <Button variant="outline" onClick={stop}>
                <Square className="h-4 w-4" /> Stop
              </Button>
            )}
            <Button onClick={submit} disabled={!text.trim() || running}>
              {running ? <Spinner size="sm" /> : <Mic className="h-4 w-4" />}
              Speak
            </Button>
          </div>
        </Card>

        {(transcript || running) && (
          <Card className="space-y-2 p-5">
            <div className="flex items-center gap-2 text-sm font-medium">
              <AudioLines className="h-4 w-4 text-muted-foreground" />
              Transcript
              {running && <Spinner size="sm" />}
            </div>
            <p className={cn("whitespace-pre-wrap text-sm", !transcript && "text-muted-foreground")}>
              {transcript || "Streaming…"}
            </p>
          </Card>
        )}

        {audioUrl && (
          <Card className="space-y-2 p-5">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Volume2 className="h-4 w-4 text-muted-foreground" />
              Audio
              <span className="font-normal text-muted-foreground">PCM16 mono · {SAMPLE_RATE / 1000} kHz</span>
            </div>
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <audio src={audioUrl} controls className="w-full" />
          </Card>
        )}
      </div>
    </PageShell>
  );
}
