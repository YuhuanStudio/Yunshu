"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Volume2,
  Mic,
  Upload,
  Play,
  Download,
  Loader2,
  Music,
  Copy,
  Check,
  X,
} from "lucide-react";

type Mode = "tts" | "asr";

export default function AudioPage() {
  const [mode, setMode] = useState<Mode>("tts");
  const [models, setModels] = useState<{ id: string }[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [ttsText, setTtsText] = useState("Hello, welcome to Yunshu!");
  const [ttsVoice, setTtsVoice] = useState("Chelsie");
  const [ttsInstruct, setTtsInstruct] = useState("Speak in a friendly and warm tone.");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [genTime, setGenTime] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [asrResult, setAsrResult] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamProgress, setStreamProgress] = useState("");
  const [streamChunks, setStreamChunks] = useState<number>(0);
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioUrlRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  // Revoke previous object URL when a new one is set or component unmounts
  const revokeAudioUrl = useCallback(() => {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
  }, []);

  const setAudioUrlSafe = useCallback((url: string) => {
    revokeAudioUrl();
    audioUrlRef.current = url;
    setAudioUrl(url);
  }, [revokeAudioUrl]);

  useEffect(() => {
    return () => {
      revokeAudioUrl();
      timersRef.current.forEach(clearTimeout);
      abortRef.current?.abort();
    };
  }, [revokeAudioUrl]);

  useEffect(() => {
    fetch("/v1/models")
      .then((r) => r.json())
      .then((data) => {
        const all = data.data || [];
        setModels(all);
        const target = mode === "tts"
          ? all.find((m: { id: string }) => /tts|voice/i.test(m.id))
          : all.find((m: { id: string }) => /asr|whisper/i.test(m.id));
        if (target) setSelectedModel(target.id);
      })
      .catch(() => {});
  }, [mode]);

  const handleTts = async () => {
    if (!ttsText.trim() || !selectedModel) return;
    setLoading(true);
    setError(null);
    revokeAudioUrl();
    const start = performance.now();
    try {
      const resp = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          input: ttsText,
          voice: ttsVoice,
          instruct: ttsInstruct || undefined,
          response_format: "wav",
        }),
      });
      if (resp.ok) {
        const blob = await resp.blob();
        setAudioUrlSafe(URL.createObjectURL(blob));
        setGenTime((performance.now() - start) / 1000);
      } else {
        setError(`TTS failed: ${resp.status} ${await resp.text()}`);
      }
    } catch (err) {
      setError(`Error: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const handleTtsStream = async () => {
    if (!ttsText.trim() || !selectedModel) return;
    setStreaming(true);
    setError(null);
    setStreamProgress("Connecting...");
    setStreamChunks(0);
    revokeAudioUrl();
    const start = performance.now();
    try {
      const resp = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          input: ttsText,
          voice: ttsVoice,
          instruct: ttsInstruct || undefined,
          response_format: "wav",
          stream: true,
        }),
      });
      if (!resp.ok) {
        setError(`Streaming TTS failed: ${resp.status}`);
        return;
      }
      const reader = resp.body?.getReader();
      if (!reader) { setError("No stream body"); return; }
      const chunks: Uint8Array[] = [];
      let chunkCount = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        chunkCount++;
        setStreamChunks(chunkCount);
        setStreamProgress(`Streaming... ${chunkCount} chunks received`);
      }
      const blob = new Blob(chunks as BlobPart[], { type: "audio/wav" });
      setAudioUrlSafe(URL.createObjectURL(blob));
      setGenTime((performance.now() - start) / 1000);
      setStreamProgress(`Completed — ${chunkCount} chunks in ${((performance.now() - start) / 1000).toFixed(1)}s`);
    } catch (err) {
      setError(`Stream error: ${err}`);
    } finally {
      setStreaming(false);
    }
  };

  const selectedModelRef = useRef(selectedModel);
  selectedModelRef.current = selectedModel;

  const handleAsr = useCallback(async (file: File) => {
    if (!selectedModelRef.current) return;
    setLoading(true);
    setError(null);
    setAsrResult("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model", selectedModelRef.current);
      const resp = await fetch("/v1/audio/transcriptions", {
        method: "POST",
        body: formData,
      });
      if (resp.ok) {
        const data = await resp.json();
        setAsrResult(data.text || "(no transcription)");
      } else {
        setError(`ASR failed: ${resp.status}`);
      }
    } catch (err) {
      setError(`Error: ${err}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("audio/")) handleAsr(file);
  }, [handleAsr]);

  const filteredModels = models.filter((m) =>
    mode === "tts" ? /tts|voice/i.test(m.id) : /asr|whisper/i.test(m.id)
  );

  return (
    <div className="p-6 space-y-6 max-w-3xl page-enter">
      <h2 className="text-2xl font-bold">Audio</h2>

      {/* Mode Tabs */}
      <div className="flex gap-1 bg-[var(--color-bg-tertiary)] rounded-lg p-1 w-fit">
        {(["tts", "asr"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === m
                ? "bg-[var(--color-accent)] text-white shadow-sm"
                : "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
            }`}
          >
            {m === "tts" ? <Volume2 className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            {m === "tts" ? "Text to Speech" : "Speech to Text"}
          </button>
        ))}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-[var(--color-danger)]/10 border border-[var(--color-danger)]/30 rounded-lg px-4 py-3 text-sm text-[var(--color-danger)] flex items-center gap-2">
          <X className="w-4 h-4 shrink-0" />
          {error}
          <button onClick={() => setError(null)} className="ml-auto opacity-60 hover:opacity-100">
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

      {mode === "tts" ? (
        <div className="space-y-4">
          <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-4">
            {/* Model */}
            <div>
              <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wide">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full mt-1 bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm"
              >
                {filteredModels.map((m) => (
                  <option key={m.id} value={m.id}>{m.id}</option>
                ))}
                {filteredModels.length === 0 && <option disabled>No TTS models found</option>}
              </select>
            </div>

            {/* Text */}
            <div>
              <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wide">Text</label>
              <textarea
                value={ttsText}
                onChange={(e) => setTtsText(e.target.value)}
                rows={3}
                className="w-full mt-1 bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-[var(--color-accent)] resize-none"
              />
            </div>

            {/* Voice + Instruction */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wide">Voice</label>
                <input
                  type="text"
                  value={ttsVoice}
                  onChange={(e) => setTtsVoice(e.target.value)}
                  className="w-full mt-1 bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wide">Instruction</label>
                <input
                  type="text"
                  value={ttsInstruct}
                  onChange={(e) => setTtsInstruct(e.target.value)}
                  className="w-full mt-1 bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-1.5 text-sm"
                  placeholder="Describe the voice..."
                />
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleTts}
                disabled={loading || streaming || !ttsText.trim() || !selectedModel}
                className="flex items-center gap-2 bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Volume2 className="w-4 h-4" />}
                {loading ? "Generating..." : "Generate"}
              </button>
              <button
                onClick={handleTtsStream}
                disabled={loading || streaming || !ttsText.trim() || !selectedModel}
                className="flex items-center gap-2 bg-[var(--color-bg-tertiary)] hover:bg-[var(--color-bg-secondary)] border border-[var(--color-border)] disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              >
                {streaming ? <Loader2 className="w-4 h-4 animate-spin" /> : <Music className="w-4 h-4" />}
                {streaming ? "Streaming..." : "Stream"}
              </button>
            </div>

            {streamProgress && (
              <div className="text-xs text-[var(--color-text-secondary)] mt-1">
                {streamProgress} {streamChunks > 0 && `(${streamChunks} chunks)`}
              </div>
            )}
          </div>

          {/* Audio Output */}
          {audioUrl && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <Music className="w-4 h-4 text-[var(--color-accent)]" />
                  Generated Audio
                </h3>
                {genTime != null && (
                  <span className="text-xs text-[var(--color-text-secondary)]">
                    {genTime.toFixed(1)}s
                  </span>
                )}
              </div>
              <audio ref={audioRef} src={audioUrl} controls className="w-full" />
              <a
                href={audioUrl}
                download="yunshu_output.wav"
                className="inline-flex items-center gap-1.5 mt-3 text-sm text-[var(--color-accent)] hover:underline"
              >
                <Download className="w-3.5 h-3.5" />
                Download WAV
              </a>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4 space-y-4">
            {/* Model */}
            <div>
              <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wide">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full mt-1 bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm"
              >
                {filteredModels.map((m) => (
                  <option key={m.id} value={m.id}>{m.id}</option>
                ))}
                {filteredModels.length === 0 && <option disabled>No ASR models found</option>}
              </select>
            </div>

            {/* Drop Zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
                dragOver
                  ? "border-[var(--color-accent)] bg-[var(--color-accent-muted)]"
                  : "border-[var(--color-border)] hover:border-[var(--color-text-secondary)]"
              }`}
            >
              <Upload className={`w-8 h-8 mx-auto mb-3 ${dragOver ? "text-[var(--color-accent)]" : "text-[var(--color-text-secondary)]"}`} />
              <p className="text-sm text-[var(--color-text-secondary)]">
                {dragOver ? "Drop audio file here" : "Drop audio file or click to browse"}
              </p>
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                WAV, MP3, FLAC, M4A supported
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleAsr(file);
                }}
              />
            </div>

            {loading && (
              <div className="flex items-center gap-2 text-sm text-[var(--color-accent)]">
                <Loader2 className="w-4 h-4 animate-spin" />
                Transcribing...
              </div>
            )}
          </div>

          {/* ASR Result */}
          {asrResult && (
            <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border)] p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <Mic className="w-4 h-4 text-[var(--color-accent)]" />
                  Transcription
                </h3>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(asrResult);
                    setCopied(true);
                    const id = setTimeout(() => setCopied(false), 2000);
                    timersRef.current.push(id);
                  }}
                  className="flex items-center gap-1 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
                >
                  {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
              <p className="whitespace-pre-wrap text-sm leading-relaxed">{asrResult}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
