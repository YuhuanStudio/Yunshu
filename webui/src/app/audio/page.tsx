"use client";

import { useEffect, useRef, useState } from "react";
import {
  Button,
  Input,
  Textarea,
  Card,
  Badge,
  Alert,
  IconButton,
  Spinner,
  SegmentedSelect,
  FileDropzone,
  cn,
  toast,
} from "yunui";
import { MediaGallery, type MediaResult } from "yunui/patterns";
import { Mic, Volume2, Copy, Check, FileAudio, X } from "lucide-react";
import { ModelPicker } from "@/components/model-picker";
import { api, ApiError } from "@/lib/api";
import { fmtBytes } from "@/lib/format";
import { PageShell } from "@/components/page-shell";
import type { Model } from "@/lib/types";

type Mode = "tts" | "asr";

const MODES = [
  { value: "tts" as const, label: "Text to speech", icon: Volume2 },
  { value: "asr" as const, label: "Speech to text", icon: Mic },
];

export default function AudioPage() {
  const [mode, setMode] = useState<Mode>("tts");
  const [model, setModel] = useState("");
  const [models, setModels] = useState<Model[]>([]);
  const [error, setError] = useState<string | null>(null);

  // TTS state
  const [text, setText] = useState("");
  const [voice, setVoice] = useState("");
  const [instruct, setInstruct] = useState("");
  const [speech, setSpeech] = useState<MediaResult[]>([]);
  const [speechView, setSpeechView] = useState<"grid" | "list">("list");
  const [generating, setGenerating] = useState(false);
  // Track object URLs so they can be revoked on unmount.
  const speechRef = useRef<MediaResult[]>([]);
  speechRef.current = speech;

  // ASR state
  const [file, setFile] = useState<File | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [copied, setCopied] = useState(false);

  // Fetch models once.
  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ data: Model[] }>("/v1/models", controller.signal)
      .then((res) => setModels(res.data ?? []))
      .catch(() => {
        /* leave the select empty; a request will surface its own error */
      });
    return () => controller.abort();
  }, []);

  // Revoke every generated object URL on unmount.
  useEffect(() => {
    return () => {
      speechRef.current.forEach((r) => r.url && URL.revokeObjectURL(r.url));
    };
  }, []);

  const describe = (e: unknown) =>
    e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Something went wrong.";

  const generateSpeech = async () => {
    if (!model || !text.trim() || generating) return;
    setGenerating(true);
    setError(null);
    try {
      const blob = await api.postBlob("/v1/audio/speech", {
        model,
        input: text,
        voice: voice.trim() || undefined,
        instruct: instruct.trim() || undefined,
        response_format: "wav",
      });
      setSpeech((r) => [
        {
          id: crypto.randomUUID(),
          url: URL.createObjectURL(blob),
          kind: "audio",
          prompt: text,
          model,
          meta: voice.trim() || undefined,
          status: "completed",
        },
        ...r,
      ]);
    } catch (e) {
      const msg = describe(e);
      setError(msg);
      toast.error("Speech generation failed", msg);
    } finally {
      setGenerating(false);
    }
  };

  const downloadSpeech = (item: MediaResult) => {
    const a = document.createElement("a");
    a.href = item.url;
    a.download = `speech-${item.id}.wav`;
    a.click();
  };

  const removeSpeech = (item: MediaResult) => {
    if (item.url) URL.revokeObjectURL(item.url);
    setSpeech((r) => r.filter((x) => x.id !== item.id));
  };

  const transcribe = async () => {
    if (!model || !file || transcribing) return;
    setTranscribing(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("model", model);
      const res = await api.postForm<{ text: string }>("/v1/audio/transcriptions", form);
      setTranscript(res.text ?? "");
    } catch (e) {
      const msg = describe(e);
      setError(msg);
      toast.error("Transcription failed", msg);
    } finally {
      setTranscribing(false);
    }
  };

  const copyTranscript = async () => {
    if (!transcript) return;
    try {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Copy failed");
    }
  };

  const modelSelect = (
    <div className="space-y-2">
      <label className="text-sm font-medium">Model</label>
      <ModelPicker models={models} value={model} onChange={setModel} />
    </div>
  );

  return (
    <PageShell
      title="Audio"
      description="Synthesize speech from text or transcribe an audio file to text."
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

        {mode === "tts" ? (
          <Card className="space-y-4 p-5">
            {modelSelect}

            <div className="space-y-2">
              <label className="text-sm font-medium">Text to speak</label>
              <Textarea
                rows={5}
                placeholder="Enter the text you want spoken aloud…"
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Voice</label>
                <Input
                  placeholder="e.g. alloy"
                  value={voice}
                  onChange={(e) => setVoice(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Instruct <span className="text-muted-foreground">(optional)</span>
                </label>
                <Input
                  placeholder="Speaking style / tone"
                  value={instruct}
                  onChange={(e) => setInstruct(e.target.value)}
                />
              </div>
            </div>

            <div className="flex justify-end">
              <Button onClick={generateSpeech} disabled={!model || !text.trim() || generating}>
                {generating ? <Spinner size="sm" /> : <Volume2 className="h-4 w-4" />}
                Generate
              </Button>
            </div>

            {speech.length > 0 && (
              <MediaGallery
                items={speech}
                viewMode={speechView}
                onViewModeChange={setSpeechView}
                onDownload={downloadSpeech}
                onDelete={removeSpeech}
              />
            )}
          </Card>
        ) : (
          <Card className="space-y-4 p-5">
            {modelSelect}

            <div className="space-y-2">
              <label className="text-sm font-medium">Audio file</label>
              <FileDropzone
                accept="audio/*"
                icon={<FileAudio className="h-6 w-6" />}
                label="Drop an audio file or click to browse"
                hint="Any audio format supported by the model"
                onFiles={(files) => {
                  if (files[0]) {
                    setFile(files[0]);
                    setTranscript(null);
                  }
                }}
              />
              {file && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <FileAudio className="h-4 w-4 shrink-0" />
                  <span className="min-w-0 flex-1 truncate">{file.name}</span>
                  <Badge variant="info">{fmtBytes(file.size)}</Badge>
                </div>
              )}
            </div>

            <div className="flex justify-end">
              <Button onClick={transcribe} disabled={!model || !file || transcribing}>
                {transcribing ? <Spinner size="sm" /> : <Mic className="h-4 w-4" />}
                Transcribe
              </Button>
            </div>

            {transcript !== null && (
              <Card className="space-y-3 p-4">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium">Transcription</span>
                  <Button variant="ghost" size="sm" onClick={copyTranscript} disabled={!transcript}>
                    {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    {copied ? "Copied" : "Copy"}
                  </Button>
                </div>
                <p
                  className={cn(
                    "select-text whitespace-pre-wrap text-sm",
                    transcript ? "" : "text-muted-foreground",
                  )}
                >
                  {transcript || "No speech detected."}
                </p>
              </Card>
            )}
          </Card>
        )}
      </div>
    </PageShell>
  );
}
