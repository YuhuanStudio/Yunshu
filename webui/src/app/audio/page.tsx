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
  Switch,
  Slider,
  NumberInput,
  CustomSelect,
  SegmentedSelect,
  FileDropzone,
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
  cn,
  toast,
} from "yunui";
import { MediaGallery, type MediaResult } from "yunui/patterns";
import {
  Mic,
  Volume2,
  Copy,
  Check,
  FileAudio,
  X,
  Wand2,
  AudioLines,
  Languages,
  ChevronDown,
  Settings2,
  Square,
} from "lucide-react";
import { ModelPicker } from "@/components/model-picker";
import { api, streamSSE, ApiError, type SSEChunk } from "@/lib/api";
import { fmtBytes } from "@/lib/format";
import { PageShell } from "@/components/page-shell";
import type { Model } from "@/lib/types";

type Mode = "tts" | "stt" | "pipeline" | "sts";
type StsOp = "enhance" | "separate" | "transform";

const MODES = [
  { value: "tts" as const, label: "Speech", icon: Volume2 },
  { value: "stt" as const, label: "Transcribe", icon: Mic },
  { value: "pipeline" as const, label: "Voice pipeline", icon: AudioLines },
  { value: "sts" as const, label: "Enhance", icon: Wand2 },
];

const FALLBACK_VOICES = ["alloy", "chelsie", "ethan", "aiden"];
const TTS_FORMATS = ["wav", "mp3", "opus", "aac", "flac", "pcm"];
const STT_FORMATS = ["json", "text", "srt", "vtt", "verbose_json"];
const STS_FORMATS = [
  { value: "wav", label: "WAV" },
  { value: "mp3", label: "MP3" },
  { value: "json", label: "JSON (base64)" },
];

// ---- page-local result shapes ---------------------------------------------
interface TranscriptSegment {
  id?: number;
  start?: number;
  end?: number;
  text: string;
}
interface TranscriptionResult {
  text: string;
  language?: string;
  duration?: number;
  segments?: TranscriptSegment[];
}
interface PipelineResult {
  text: string;
  audio: string;
  transcription: string;
}

// ---- binary helpers -------------------------------------------------------
/** Base64 → Uint8Array. */
function b64ToBytes(b64: string): Uint8Array<ArrayBuffer> {
  const bin = atob(b64);
  const out = new Uint8Array(new ArrayBuffer(bin.length));
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** File → base64 (chunked to avoid call-stack limits on large inputs). */
async function fileToBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** Wrap raw PCM16 mono samples in a 44-byte WAV container. */
function pcmToWav(pcm: Uint8Array, sampleRate: number, channels = 1, bits = 16): Blob {
  const blockAlign = (channels * bits) / 8;
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
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bits, true);
  w(36, "data");
  view.setUint32(40, pcm.length, true);
  new Uint8Array(buffer, 44).set(pcm);
  return new Blob([buffer], { type: "audio/wav" });
}

// ---- small field helpers --------------------------------------------------
function Field({
  label,
  hint,
  children,
}: {
  label: ReactNode;
  hint?: ReactNode;
  children: ReactNode;
}) {
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
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <label className="font-medium">{label}</label>
        <Badge variant="info">{value}</Badge>
      </div>
      <Slider
        value={[value]}
        onValueChange={(v) => onChange(v[0] ?? min)}
        min={min}
        max={max}
        step={step}
      />
    </div>
  );
}

export default function AudioPage() {
  const [mode, setMode] = useState<Mode>("tts");
  const [models, setModels] = useState<Model[]>([]);
  const [voices, setVoices] = useState<string[]>(FALLBACK_VOICES);
  const [error, setError] = useState<string | null>(null);

  // Extension per generated item, keyed by id, for download filenames.
  const extRef = useRef<Record<string, string>>({});

  // --- TTS state ---
  const [ttsModel, setTtsModel] = useState("");
  const [text, setText] = useState("");
  const [voice, setVoice] = useState("alloy");
  const [speed, setSpeed] = useState(1);
  const [format, setFormat] = useState("wav");
  const [stream, setStream] = useState(false);
  const [advOpen, setAdvOpen] = useState(false);
  const [instruct, setInstruct] = useState("");
  const [temperature, setTemperature] = useState(1);
  const [topP, setTopP] = useState(0.95);
  const [topK, setTopK] = useState(50);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [language, setLanguage] = useState("");
  const [seed, setSeed] = useState<number | null>(null);
  const [refAudio, setRefAudio] = useState("");
  const [refText, setRefText] = useState("");
  const [speech, setSpeech] = useState<MediaResult[]>([]);
  const [speechView, setSpeechView] = useState<"grid" | "list">("list");
  const [generating, setGenerating] = useState(false);
  const streamAbort = useRef<AbortController | null>(null);
  const speechRef = useRef<MediaResult[]>([]);
  speechRef.current = speech;

  // --- STT state ---
  const [sttModel, setSttModel] = useState("");
  const [sttFile, setSttFile] = useState<File | null>(null);
  const [sttLanguage, setSttLanguage] = useState("");
  const [sttFormat, setSttFormat] = useState("json");
  const [sttPrompt, setSttPrompt] = useState("");
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [subtitle, setSubtitle] = useState<string | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [copied, setCopied] = useState(false);

  // --- Voice pipeline state ---
  const [pipeFile, setPipeFile] = useState<File | null>(null);
  const [llmModel, setLlmModel] = useState("");
  const [pipeVoice, setPipeVoice] = useState("alloy");
  const [pipeSpeed, setPipeSpeed] = useState(1);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [llmTemperature, setLlmTemperature] = useState(0.7);
  const [pipeResult, setPipeResult] = useState<PipelineResult | null>(null);
  const [pipeMedia, setPipeMedia] = useState<MediaResult[]>([]);
  const [pipeMediaView, setPipeMediaView] = useState<"grid" | "list">("list");
  const [pipeRunning, setPipeRunning] = useState(false);
  const pipeMediaRef = useRef<MediaResult[]>([]);
  pipeMediaRef.current = pipeMedia;

  // --- STS state ---
  const [stsFile, setStsFile] = useState<File | null>(null);
  const [stsOp, setStsOp] = useState<StsOp>("enhance");
  const [stsFormat, setStsFormat] = useState("wav");
  const [noiseFloor, setNoiseFloor] = useState(-40);
  const [sourceText, setSourceText] = useState("");
  const [pitchShift, setPitchShift] = useState(0);
  const [formantRatio, setFormantRatio] = useState(1);
  const [stsMedia, setStsMedia] = useState<MediaResult[]>([]);
  const [stsMediaView, setStsMediaView] = useState<"grid" | "list">("list");
  const [stsRunning, setStsRunning] = useState(false);
  const stsMediaRef = useRef<MediaResult[]>([]);
  stsMediaRef.current = stsMedia;

  // Fetch models + voices once.
  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ data: Model[] }>("/v1/models", controller.signal)
      .then((res) => setModels(res.data ?? []))
      .catch(() => {
        /* a request will surface its own error */
      });
    api
      .get<{ data: { id: string }[] }>("/v1/audio/voices", controller.signal)
      .then((res) => {
        const ids = (res.data ?? []).map((v) => v.id).filter(Boolean);
        if (ids.length) setVoices(ids);
      })
      .catch(() => {
        /* keep FALLBACK_VOICES */
      });
    return () => controller.abort();
  }, []);

  // Revoke every generated object URL on unmount.
  useEffect(() => {
    return () => {
      streamAbort.current?.abort();
      [speechRef, pipeMediaRef, stsMediaRef].forEach((ref) =>
        ref.current.forEach((r) => r.url && URL.revokeObjectURL(r.url)),
      );
    };
  }, []);

  const describe = (e: unknown) =>
    e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Something went wrong.";

  const voiceOptions = voices.map((v) => ({ value: v, label: v }));

  const addSpeech = (blob: Blob, ext: string, meta?: string) => {
    const id = crypto.randomUUID();
    extRef.current[id] = ext;
    setSpeech((r) => [
      {
        id,
        url: URL.createObjectURL(blob),
        kind: "audio",
        prompt: text,
        model: ttsModel,
        meta: meta ?? voice,
        status: "completed",
      },
      ...r,
    ]);
  };

  // ---- TTS: non-streaming (binary) ----
  const generateSpeech = async () => {
    if (!ttsModel || !text.trim() || generating) return;
    setGenerating(true);
    setError(null);
    try {
      const body: Record<string, unknown> = {
        model: ttsModel,
        input: text,
        voice,
        speed,
        response_format: format,
        temperature,
        top_p: topP,
        top_k: topK,
        repetition_penalty: repetitionPenalty,
        max_tokens: maxTokens,
      };
      if (instruct.trim()) body.instruct = instruct.trim();
      if (language.trim()) body.language = language.trim();
      if (seed !== null) body.seed = seed;
      if (refAudio.trim()) body.ref_audio = refAudio.trim();
      if (refText.trim()) body.ref_text = refText.trim();

      const blob = await api.postBlob("/v1/audio/speech", body);
      addSpeech(blob, format === "pcm" ? "pcm" : format);
    } catch (e) {
      const msg = describe(e);
      setError(msg);
      toast.error("Speech generation failed", msg);
    } finally {
      setGenerating(false);
    }
  };

  // ---- TTS: streaming (SSE of PCM16 chunks, assembled into one WAV) ----
  const streamSpeech = async () => {
    if (!ttsModel || !text.trim() || generating) return;
    setGenerating(true);
    setError(null);
    const controller = new AbortController();
    streamAbort.current = controller;
    let sampleRate = 24000;
    const chunks: Uint8Array[] = [];
    let cancelled = false;
    try {
      const body: Record<string, unknown> = {
        model: ttsModel,
        input: text,
        voice,
        speed,
        temperature,
        top_p: topP,
        top_k: topK,
        repetition_penalty: repetitionPenalty,
        max_tokens: maxTokens,
      };
      if (instruct.trim()) body.instruct = instruct.trim();
      if (language.trim()) body.language = language.trim();
      if (seed !== null) body.seed = seed;

      await streamSSE(
        "/v1/audio/speech/stream",
        body,
        (data: SSEChunk) => {
          const type = data.type as string | undefined;
          if (type === "header") {
            if (typeof data.sample_rate === "number") sampleRate = data.sample_rate;
          } else if (type === "audio") {
            if (typeof data.audio === "string") chunks.push(b64ToBytes(data.audio));
          } else if (type === "cancelled") {
            cancelled = true;
          } else if (type === "error") {
            throw new ApiError(String(data.message ?? "Stream error"), 500);
          }
        },
        controller.signal,
      );

      if (!cancelled && chunks.length) {
        const total = chunks.reduce((n, c) => n + c.length, 0);
        const pcm = new Uint8Array(total);
        let off = 0;
        for (const c of chunks) {
          pcm.set(c, off);
          off += c.length;
        }
        addSpeech(pcmToWav(pcm, sampleRate), "wav", `${voice} · streamed`);
      }
    } catch (e) {
      if (controller.signal.aborted) {
        // user-initiated stop; keep whatever we captured
        if (chunks.length) {
          const total = chunks.reduce((n, c) => n + c.length, 0);
          const pcm = new Uint8Array(total);
          let off = 0;
          for (const c of chunks) {
            pcm.set(c, off);
            off += c.length;
          }
          addSpeech(pcmToWav(pcm, sampleRate), "wav", `${voice} · partial`);
        }
      } else {
        const msg = describe(e);
        setError(msg);
        toast.error("Speech stream failed", msg);
      }
    } finally {
      streamAbort.current = null;
      setGenerating(false);
    }
  };

  const stopStream = () => streamAbort.current?.abort();

  const downloadMedia = (item: MediaResult) => {
    const a = document.createElement("a");
    a.href = item.url;
    a.download = `speech-${item.id}.${extRef.current[item.id] ?? "wav"}`;
    a.click();
  };

  const makeRemove =
    (setter: React.Dispatch<React.SetStateAction<MediaResult[]>>) => (item: MediaResult) => {
      if (item.url) URL.revokeObjectURL(item.url);
      setter((r) => r.filter((x) => x.id !== item.id));
    };
  const removeSpeech = makeRemove(setSpeech);
  const removePipeMedia = makeRemove(setPipeMedia);
  const removeStsMedia = makeRemove(setStsMedia);

  // ---- STT: transcribe / translate ----
  const runStt = async (translate: boolean) => {
    if (!sttModel || !sttFile || transcribing) return;
    setTranscribing(true);
    setError(null);
    setResult(null);
    setSubtitle(null);
    try {
      const form = new FormData();
      form.append("file", sttFile);
      form.append("model", sttModel);
      form.append("response_format", translate ? "json" : sttFormat);
      if (sttLanguage.trim() && !translate) form.append("language", sttLanguage.trim());
      if (sttPrompt.trim()) form.append("prompt", sttPrompt.trim());
      const path = translate ? "/v1/audio/translations" : "/v1/audio/transcriptions";

      // srt/vtt/text return plain strings; json/verbose_json return objects.
      if (!translate && (sttFormat === "srt" || sttFormat === "vtt" || sttFormat === "text")) {
        const blob = await api.postBlob(path, form);
        setSubtitle(await blob.text());
      } else {
        const res = await api.postForm<TranscriptionResult>(path, form);
        setResult(res);
      }
    } catch (e) {
      const msg = describe(e);
      setError(msg);
      toast.error(translate ? "Translation failed" : "Transcription failed", msg);
    } finally {
      setTranscribing(false);
    }
  };

  const copyText = async (value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Copy failed");
    }
  };

  // ---- Voice pipeline ----
  const runPipeline = async () => {
    if (!pipeFile || !llmModel || pipeRunning) return;
    setPipeRunning(true);
    setError(null);
    setPipeResult(null);
    try {
      const form = new FormData();
      form.append("file", pipeFile);
      form.append("llm_model", llmModel);
      form.append("voice", pipeVoice);
      form.append("speed", String(pipeSpeed));
      form.append("llm_temperature", String(llmTemperature));
      if (systemPrompt.trim()) form.append("system_prompt", systemPrompt.trim());
      form.append("stream", "false");
      const res = await api.postForm<PipelineResult>("/v1/audio/voice-pipeline", form);
      setPipeResult(res);
      if (res.audio) {
        const id = crypto.randomUUID();
        extRef.current[id] = "wav";
        setPipeMedia((r) => [
          {
            id,
            url: URL.createObjectURL(new Blob([b64ToBytes(res.audio)], { type: "audio/wav" })),
            kind: "audio",
            prompt: res.text,
            model: llmModel,
            meta: pipeVoice,
            status: "completed",
          },
          ...r,
        ]);
      }
    } catch (e) {
      const msg = describe(e);
      setError(msg);
      toast.error("Voice pipeline failed", msg);
    } finally {
      setPipeRunning(false);
    }
  };

  // ---- Speech-to-speech ----
  const runSts = async () => {
    if (!stsFile || stsRunning) return;
    setStsRunning(true);
    setError(null);
    try {
      const audio = await fileToBase64(stsFile);
      const body: Record<string, unknown> = { audio, response_format: stsFormat };
      if (stsOp === "enhance") {
        body.method = "spectral";
        body.noise_floor_db = noiseFloor;
      } else if (stsOp === "separate") {
        body.method = "vocal";
        if (sourceText.trim()) body.source_text = sourceText.trim();
      } else {
        body.pitch_shift = pitchShift;
        body.formant_ratio = formantRatio;
      }
      const path = `/v1/audio/speech-to-speech/${stsOp}`;

      let blob: Blob;
      let ext: string;
      if (stsFormat === "json") {
        const res = await api.post<{ audio?: string; data?: string }>(path, body);
        const b64 = res.audio ?? res.data;
        if (!b64) throw new ApiError("No audio returned by the server.", 500);
        blob = new Blob([b64ToBytes(b64)], { type: "audio/wav" });
        ext = "wav";
      } else {
        blob = await api.postBlob(path, body);
        ext = stsFormat;
      }
      const id = crypto.randomUUID();
      extRef.current[id] = ext;
      setStsMedia((r) => [
        {
          id,
          url: URL.createObjectURL(blob),
          kind: "audio",
          prompt: stsFile.name,
          meta: stsOp,
          status: "completed",
        },
        ...r,
      ]);
    } catch (e) {
      const msg = describe(e);
      setError(msg);
      toast.error("Enhance failed", msg);
    } finally {
      setStsRunning(false);
    }
  };

  // ---- shared bits ----
  const fileRow = (file: File | null) =>
    file && (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <FileAudio className="h-4 w-4 shrink-0" />
        <span className="min-w-0 flex-1 truncate">{file.name}</span>
        <Badge variant="info">{fmtBytes(file.size)}</Badge>
      </div>
    );

  return (
    <PageShell
      title="Audio"
      description="Synthesize speech, transcribe or translate audio, run a full voice pipeline, or enhance a recording."
      width="narrow"
    >
      <div className="space-y-6">
        <SegmentedSelect<Mode> options={MODES} value={mode} onChange={setMode} />

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

        {/* ---------------------------------------------------------------- TTS */}
        {mode === "tts" && (
          <Card className="space-y-4 p-5">
            <Field label="Model">
              <ModelPicker models={models} value={ttsModel} onChange={setTtsModel} />
            </Field>

            <Field label="Text to speak">
              <Textarea
                rows={5}
                maxLength={32768}
                placeholder="Enter the text you want spoken aloud…"
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </Field>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Voice">
                <CustomSelect
                  options={voiceOptions}
                  value={voice}
                  onChange={setVoice}
                  searchable
                />
              </Field>
              <Field label="Format">
                <CustomSelect
                  options={TTS_FORMATS.map((f) => ({ value: f, label: f.toUpperCase() }))}
                  value={format}
                  onChange={setFormat}
                />
              </Field>
            </div>

            <SliderRow label="Speed" value={speed} onChange={setSpeed} min={0.25} max={4} step={0.05} />

            <label className="flex items-center justify-between gap-3 rounded-lg border border-border p-3">
              <div className="space-y-0.5">
                <span className="text-sm font-medium">Stream</span>
                <p className="text-xs text-muted-foreground">
                  Receive PCM16 chunks over SSE (assembled into one clip; format is forced to WAV).
                </p>
              </div>
              <Switch checked={stream} onCheckedChange={setStream} disabled={generating} />
            </label>

            <Collapsible open={advOpen} onOpenChange={setAdvOpen}>
              <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg border border-border px-3 py-2 text-sm font-medium hover:bg-muted/50">
                <span className="flex items-center gap-2">
                  <Settings2 className="h-4 w-4" /> Advanced
                </span>
                <ChevronDown
                  className={cn("h-4 w-4 transition-transform", advOpen && "rotate-180")}
                />
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-4 pt-4">
                <Field
                  label="Instruct"
                  hint="(VoiceDesign style / tone, optional)"
                >
                  <Input
                    placeholder="e.g. warm and cheerful, speaking slowly"
                    value={instruct}
                    onChange={(e) => setInstruct(e.target.value)}
                  />
                </Field>

                <div className="grid gap-4 sm:grid-cols-2">
                  <SliderRow
                    label="Temperature"
                    value={temperature}
                    onChange={setTemperature}
                    min={0}
                    max={2}
                    step={0.05}
                  />
                  <SliderRow
                    label="Top-p"
                    value={topP}
                    onChange={setTopP}
                    min={0}
                    max={1}
                    step={0.01}
                  />
                </div>

                <div className="grid gap-4 sm:grid-cols-3">
                  <Field label="Top-k">
                    <NumberInput value={topK} onChange={setTopK} min={0} step={1} />
                  </Field>
                  <Field label="Repetition penalty">
                    <NumberInput
                      value={repetitionPenalty}
                      onChange={setRepetitionPenalty}
                      min={0}
                      step={0.05}
                    />
                  </Field>
                  <Field label="Max tokens">
                    <NumberInput value={maxTokens} onChange={setMaxTokens} min={1} step={64} />
                  </Field>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <Field label="Language" hint="(optional)">
                    <Input
                      placeholder="e.g. en, zh"
                      value={language}
                      onChange={(e) => setLanguage(e.target.value)}
                    />
                  </Field>
                  <Field label="Seed" hint="(optional)">
                    <NumberInput
                      value={seed ?? 0}
                      onChange={setSeed}
                      min={0}
                      step={1}
                      placeholder="Random"
                    />
                  </Field>
                </div>

                <div className="space-y-2 rounded-lg border border-dashed border-border p-3">
                  <p className="text-xs text-muted-foreground">
                    Voice cloning — these are paths on the server host, not uploads.
                  </p>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <Field label="Ref audio path" hint="(server-side)">
                      <Input
                        placeholder="/data/voices/ref.wav"
                        value={refAudio}
                        onChange={(e) => setRefAudio(e.target.value)}
                      />
                    </Field>
                    <Field label="Ref text" hint="(server-side)">
                      <Input
                        placeholder="Transcript of the ref audio"
                        value={refText}
                        onChange={(e) => setRefText(e.target.value)}
                      />
                    </Field>
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>

            <div className="flex justify-end gap-2">
              {stream && generating && (
                <Button variant="outline" onClick={stopStream}>
                  <Square className="h-4 w-4" /> Stop
                </Button>
              )}
              <Button
                onClick={stream ? streamSpeech : generateSpeech}
                disabled={!ttsModel || !text.trim() || generating}
              >
                {generating ? <Spinner size="sm" /> : <Volume2 className="h-4 w-4" />}
                {stream ? "Stream" : "Generate"}
              </Button>
            </div>

            {speech.length > 0 && (
              <MediaGallery
                items={speech}
                viewMode={speechView}
                onViewModeChange={setSpeechView}
                onDownload={downloadMedia}
                onDelete={removeSpeech}
              />
            )}
          </Card>
        )}

        {/* --------------------------------------------------------------- STT */}
        {mode === "stt" && (
          <Card className="space-y-4 p-5">
            <Field label="Model">
              <ModelPicker models={models} value={sttModel} onChange={setSttModel} />
            </Field>

            <Field label="Audio file">
              <FileDropzone
                accept="audio/*,video/*"
                icon={<FileAudio className="h-6 w-6" />}
                label="Drop an audio file or click to browse"
                hint="Audio or video, up to 25 MB"
                onFiles={(files) => {
                  if (files[0]) {
                    setSttFile(files[0]);
                    setResult(null);
                    setSubtitle(null);
                  }
                }}
              />
              {fileRow(sttFile)}
            </Field>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Language" hint="(optional)">
                <Input
                  placeholder="Auto-detect"
                  value={sttLanguage}
                  onChange={(e) => setSttLanguage(e.target.value)}
                />
              </Field>
              <Field label="Response format">
                <CustomSelect
                  options={STT_FORMATS.map((f) => ({ value: f, label: f }))}
                  value={sttFormat}
                  onChange={setSttFormat}
                />
              </Field>
            </div>

            <Field label="Prompt" hint="(optional context / spelling hints)">
              <Input
                placeholder="e.g. proper nouns, jargon"
                value={sttPrompt}
                onChange={(e) => setSttPrompt(e.target.value)}
              />
            </Field>

            <div className="flex flex-wrap justify-end gap-2">
              <Button
                variant="outline"
                onClick={() => runStt(true)}
                disabled={!sttModel || !sttFile || transcribing}
              >
                <Languages className="h-4 w-4" /> Translate to English
              </Button>
              <Button
                onClick={() => runStt(false)}
                disabled={!sttModel || !sttFile || transcribing}
              >
                {transcribing ? <Spinner size="sm" /> : <Mic className="h-4 w-4" />}
                Transcribe
              </Button>
            </div>

            {subtitle !== null && (
              <Card className="space-y-3 p-4">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium">
                    Output <Badge variant="info">{sttFormat}</Badge>
                  </span>
                  <Button variant="ghost" size="sm" onClick={() => copyText(subtitle)}>
                    {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    {copied ? "Copied" : "Copy"}
                  </Button>
                </div>
                <pre className="max-h-96 select-text overflow-auto whitespace-pre-wrap text-sm">
                  {subtitle || "No output."}
                </pre>
              </Card>
            )}

            {result && (
              <Card className="space-y-3 p-4">
                <div className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-2 text-sm font-medium">
                    Transcription
                    {result.language && <Badge variant="info">{result.language}</Badge>}
                    {typeof result.duration === "number" && (
                      <Badge variant="default">{result.duration.toFixed(1)}s</Badge>
                    )}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyText(result.text)}
                    disabled={!result.text}
                  >
                    {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    {copied ? "Copied" : "Copy"}
                  </Button>
                </div>
                <p
                  className={cn(
                    "select-text whitespace-pre-wrap text-sm",
                    result.text ? "" : "text-muted-foreground",
                  )}
                >
                  {result.text || "No speech detected."}
                </p>
                {result.segments && result.segments.length > 0 && (
                  <div className="space-y-1 border-t border-border pt-3">
                    {result.segments.map((s, i) => (
                      <div key={s.id ?? i} className="flex gap-3 text-xs">
                        {typeof s.start === "number" && (
                          <span className="shrink-0 font-mono text-muted-foreground">
                            {s.start.toFixed(1)}s
                          </span>
                        )}
                        <span className="min-w-0 flex-1">{s.text}</span>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            )}
          </Card>
        )}

        {/* ---------------------------------------------------------- pipeline */}
        {mode === "pipeline" && (
          <Card className="space-y-4 p-5">
            <p className="text-sm text-muted-foreground">
              Transcribe an utterance, answer it with an LLM, and speak the reply back.
            </p>

            <Field label="Audio file">
              <FileDropzone
                accept="audio/*,video/*"
                icon={<FileAudio className="h-6 w-6" />}
                label="Drop a spoken question"
                hint="Audio or video"
                onFiles={(files) => {
                  if (files[0]) {
                    setPipeFile(files[0]);
                    setPipeResult(null);
                  }
                }}
              />
              {fileRow(pipeFile)}
            </Field>

            <Field label="LLM model">
              <ModelPicker models={models} value={llmModel} onChange={setLlmModel} />
            </Field>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Voice">
                <CustomSelect
                  options={voiceOptions}
                  value={pipeVoice}
                  onChange={setPipeVoice}
                  searchable
                />
              </Field>
              <SliderRow
                label="Speed"
                value={pipeSpeed}
                onChange={setPipeSpeed}
                min={0.25}
                max={4}
                step={0.05}
              />
            </div>

            <SliderRow
              label="LLM temperature"
              value={llmTemperature}
              onChange={setLlmTemperature}
              min={0}
              max={2}
              step={0.05}
            />

            <Field label="System prompt" hint="(optional)">
              <Textarea
                rows={3}
                placeholder="You are a helpful voice assistant…"
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
              />
            </Field>

            <div className="flex justify-end">
              <Button onClick={runPipeline} disabled={!pipeFile || !llmModel || pipeRunning}>
                {pipeRunning ? <Spinner size="sm" /> : <AudioLines className="h-4 w-4" />}
                Run pipeline
              </Button>
            </div>

            {pipeResult && (
              <div className="space-y-3">
                <Card className="space-y-1 p-4">
                  <span className="text-xs font-medium text-muted-foreground">Heard</span>
                  <p className="select-text whitespace-pre-wrap text-sm">
                    {pipeResult.transcription || "—"}
                  </p>
                </Card>
                <Card className="space-y-1 p-4">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium text-muted-foreground">Reply</span>
                    <Button variant="ghost" size="sm" onClick={() => copyText(pipeResult.text)}>
                      {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                      {copied ? "Copied" : "Copy"}
                    </Button>
                  </div>
                  <p className="select-text whitespace-pre-wrap text-sm">{pipeResult.text || "—"}</p>
                </Card>
              </div>
            )}

            {pipeMedia.length > 0 && (
              <MediaGallery
                items={pipeMedia}
                viewMode={pipeMediaView}
                onViewModeChange={setPipeMediaView}
                onDownload={downloadMedia}
                onDelete={removePipeMedia}
              />
            )}
          </Card>
        )}

        {/* --------------------------------------------------------------- STS */}
        {mode === "sts" && (
          <Card className="space-y-4 p-5">
            <Field label="Audio file" hint="(WAV recommended)">
              <FileDropzone
                accept="audio/*"
                icon={<FileAudio className="h-6 w-6" />}
                label="Drop a WAV recording"
                hint="Encoded to base64 in the browser"
                onFiles={(files) => {
                  if (files[0]) setStsFile(files[0]);
                }}
              />
              {fileRow(stsFile)}
            </Field>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Operation">
                <SegmentedSelect<StsOp>
                  options={[
                    { value: "enhance", label: "Enhance" },
                    { value: "separate", label: "Separate" },
                    { value: "transform", label: "Transform" },
                  ]}
                  value={stsOp}
                  onChange={setStsOp}
                />
              </Field>
              <Field label="Response format">
                <CustomSelect options={STS_FORMATS} value={stsFormat} onChange={setStsFormat} />
              </Field>
            </div>

            {stsOp === "enhance" && (
              <SliderRow
                label="Noise floor (dB)"
                value={noiseFloor}
                onChange={setNoiseFloor}
                min={-80}
                max={0}
                step={1}
              />
            )}

            {stsOp === "separate" && (
              <Field label="Source text" hint="(optional guidance)">
                <Input
                  placeholder="What to isolate, e.g. lead vocal"
                  value={sourceText}
                  onChange={(e) => setSourceText(e.target.value)}
                />
              </Field>
            )}

            {stsOp === "transform" && (
              <div className="grid gap-4 sm:grid-cols-2">
                <SliderRow
                  label="Pitch shift (semitones)"
                  value={pitchShift}
                  onChange={setPitchShift}
                  min={-12}
                  max={12}
                  step={1}
                />
                <SliderRow
                  label="Formant ratio"
                  value={formantRatio}
                  onChange={setFormantRatio}
                  min={0.5}
                  max={2}
                  step={0.05}
                />
              </div>
            )}

            <div className="flex justify-end">
              <Button onClick={runSts} disabled={!stsFile || stsRunning}>
                {stsRunning ? <Spinner size="sm" /> : <Wand2 className="h-4 w-4" />}
                Process
              </Button>
            </div>

            {stsMedia.length > 0 && (
              <MediaGallery
                items={stsMedia}
                viewMode={stsMediaView}
                onViewModeChange={setStsMediaView}
                onDownload={downloadMedia}
                onDelete={removeStsMedia}
              />
            )}
          </Card>
        )}
      </div>
    </PageShell>
  );
}
