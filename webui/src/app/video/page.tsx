"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Button,
  Textarea,
  Input,
  NumberInput,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  SegmentedSelect,
  Slider,
  Switch,
  Card,
  Badge,
  Alert,
  IconButton,
  Spinner,
  FileDropzone,
  EmptyState,
} from "yunui";
import { MediaGallery, type MediaResult } from "yunui/patterns";
import { Clapperboard, Film, Sparkles, Upload, X } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, ApiError, streamSSE, type SSEChunk } from "@/lib/api";
import type { Model } from "@/lib/types";

/**
 * Video generation playground — text-to-video (T2V) or image-to-video (I2V,
 * triggered by uploading a reference image). Two delivery paths:
 *
 *  - live preview OFF → `POST /v1/video/generations` (non-stream). The backend
 *    returns a base64 MP4 which we decode into a Blob → object URL → a
 *    `kind: "video"` gallery item.
 *  - live preview ON → the same endpoint with `stream: true`, consumed via SSE.
 *    Per-frame PNG events update a live preview panel and drive the gallery
 *    item's progress; on `done` we finalize with the last frame as a still
 *    `kind: "image"` (no MP4 is produced in stream mode).
 */

// ---- request/response shapes (page-local; backend contract) --------------

interface VideoGenerateRequest {
  model: string;
  prompt: string;
  negative_prompt: string;
  image?: string;
  width: number;
  height: number;
  num_frames: number;
  num_inference_steps: number;
  guide_scale: number;
  fps: number;
  seed: number | null;
  scheduler: string;
  tiling: string;
  response_format: "mp4" | "frames";
  stream: boolean;
}

interface VideoMp4Response {
  data: {
    video: string;
    num_frames?: number;
    fps?: number;
    width?: number;
    height?: number;
    method?: string;
  }[];
}

/** A per-frame SSE event during a streamed generation. */
interface FrameEvent {
  data?: { frame: string; index: number; width: number; height: number; type: "frame" }[];
  type?: "done";
  frames_delivered?: number;
  error?: { type: string; message: string };
}

const SCHEDULERS = ["unipc", "euler", "dpm++"] as const;
const TILING = ["auto", "none", "default", "aggressive", "conservative", "spatial", "temporal"] as const;

const FORMATS = [
  { value: "mp4" as const, label: "MP4" },
  { value: "frames" as const, label: "Frames" },
];

/** Decode a base64 payload into a Blob of the given MIME type. */
function base64ToBlob(b64: string, type: string): Blob {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}

export default function VideoPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("wan-2.2-t2v");

  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [width, setWidth] = useState(1280);
  const [height, setHeight] = useState(704);
  const [numFrames, setNumFrames] = useState(81);
  const [steps, setSteps] = useState(20);
  const [guideScale, setGuideScale] = useState(5.0);
  const [fps, setFps] = useState(16);
  const [seed, setSeed] = useState<number | undefined>(undefined);
  const [scheduler, setScheduler] = useState<string>("unipc");
  const [tiling, setTiling] = useState<string>("auto");
  const [format, setFormat] = useState<"mp4" | "frames">("mp4");
  const [livePreview, setLivePreview] = useState(false);

  // I2V reference image (base64 data URL); its presence switches T2V → I2V.
  const [image, setImage] = useState<string | null>(null);

  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveFrame, setLiveFrame] = useState<string | null>(null);
  const [results, setResults] = useState<MediaResult[]>([]);

  // Track object URLs so we can revoke them on unmount.
  const resultsRef = useRef<MediaResult[]>([]);
  resultsRef.current = results;

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ data: Model[] }>("/v1/models", controller.signal)
      .then((res) => setModels(res.data ?? []))
      .catch(() => {
        /* model list unavailable — a request will surface its own error */
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    return () => {
      resultsRef.current.forEach((r) => {
        if (r.url.startsWith("blob:")) URL.revokeObjectURL(r.url);
      });
    };
  }, []);

  const describe = (e: unknown) =>
    e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Something went wrong.";

  const onImageFiles = useCallback((files: File[]) => {
    const file = files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImage(reader.result as string);
    reader.readAsDataURL(file);
  }, []);

  const isI2V = image !== null;

  const generate = useCallback(async () => {
    if (!model || !prompt.trim() || generating) return;
    setGenerating(true);
    setError(null);
    setLiveFrame(null);

    const id = crypto.randomUUID();
    const meta = `${width}×${height} · ${numFrames}f`;
    setResults((r) => [
      { id, url: "", kind: "video", prompt, model, meta, status: "processing", progress: 1 },
      ...r,
    ]);

    const body: VideoGenerateRequest = {
      model,
      prompt,
      negative_prompt: negativePrompt,
      image: isI2V ? image!.replace(/^data:[^,]+,/, "") : undefined,
      width,
      height,
      num_frames: numFrames,
      num_inference_steps: steps,
      guide_scale: guideScale,
      fps,
      seed: seed ?? null,
      scheduler,
      tiling,
      response_format: livePreview ? "frames" : format,
      stream: livePreview,
    };

    try {
      if (livePreview) {
        // Streamed: per-frame PNGs; the backend reports errors as SSE events
        // (`{ error: {...} }`), not HTTP status codes.
        let lastFrame: string | null = null;
        let delivered = 0;
        let streamError: string | null = null;

        await streamSSE("/v1/video/generations", body, (chunk: SSEChunk) => {
          const ev = chunk as FrameEvent;
          if (ev.error) {
            streamError = ev.error.message;
            return;
          }
          if (ev.type === "done") {
            delivered = ev.frames_delivered ?? delivered;
            return;
          }
          const frame = ev.data?.[0];
          if (frame?.frame) {
            const url = `data:image/png;base64,${frame.frame}`;
            lastFrame = url;
            delivered = frame.index + 1;
            setLiveFrame(url);
            const pct =
              numFrames > 0 ? Math.min(99, Math.round(((frame.index + 1) / numFrames) * 100)) : 50;
            setResults((r) => r.map((x) => (x.id === id ? { ...x, progress: pct } : x)));
          }
        });

        if (streamError) throw new Error(streamError);
        if (!lastFrame) throw new Error("The stream produced no frames.");

        // No MP4 in stream mode — finalize with the last frame as a still image.
        setResults((r) =>
          r.map((x) =>
            x.id === id
              ? {
                  ...x,
                  url: lastFrame!,
                  kind: "image",
                  status: "completed",
                  progress: 100,
                  meta: `${meta} · ${delivered} frames`,
                }
              : x,
          ),
        );
      } else {
        // Non-stream: JSON with a base64 MP4 → Blob → object URL.
        const res = await api.post<VideoMp4Response>("/v1/video/generations", body);
        const clip = res.data?.[0];
        if (!clip?.video) throw new Error("The response contained no video data.");
        const url = URL.createObjectURL(base64ToBlob(clip.video, "video/mp4"));
        const detail = `${clip.width ?? width}×${clip.height ?? height} · ${clip.num_frames ?? numFrames}f @ ${clip.fps ?? fps}fps`;
        setResults((r) =>
          r.map((x) =>
            x.id === id ? { ...x, url, status: "completed", progress: 100, meta: detail } : x,
          ),
        );
      }
    } catch (e) {
      const msg = describe(e);
      setError(msg);
      setResults((r) => r.map((x) => (x.id === id ? { ...x, status: "failed", error: msg } : x)));
    } finally {
      setGenerating(false);
      setLiveFrame(null);
    }
  }, [
    model,
    prompt,
    negativePrompt,
    image,
    isI2V,
    width,
    height,
    numFrames,
    steps,
    guideScale,
    fps,
    seed,
    scheduler,
    tiling,
    format,
    livePreview,
    generating,
  ]);

  const download = useCallback((item: MediaResult) => {
    const a = document.createElement("a");
    a.href = item.url;
    a.download = `yunshu-${item.id}.${item.kind === "video" ? "mp4" : "png"}`;
    a.click();
  }, []);

  const remove = useCallback((item: MediaResult) => {
    if (item.url.startsWith("blob:")) URL.revokeObjectURL(item.url);
    setResults((r) => r.filter((x) => x.id !== item.id));
  }, []);

  return (
    <PageShell
      title="Video"
      description="Generate video from a prompt (T2V), or animate a reference image (I2V)."
      width="wide"
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Controls */}
        <div className="space-y-6">
          {error && (
            <Alert
              variant="error"
              title="Generation failed"
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
            <div className="flex items-center justify-between gap-3">
              <label className="text-sm font-medium">Model</label>
              <Badge variant={isI2V ? "info" : "default"}>{isI2V ? "Image → Video" : "Text → Video"}</Badge>
            </div>
            <ModelPicker models={models} value={model} onChange={setModel} />

            <div className="space-y-2">
              <label className="text-sm font-medium">Prompt</label>
              <Textarea aria-label="Prompt"
                rows={3}
                placeholder="Describe the motion and scene you want…"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">
                Negative prompt <span className="text-muted-foreground">(optional)</span>
              </label>
              <Textarea
                rows={2}
                placeholder="What to avoid…"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
              />
            </div>
          </Card>

          {/* I2V reference image */}
          <Card className="space-y-4 p-5">
            <div className="flex items-center justify-between gap-3">
              <div className="space-y-0.5">
                <span className="text-sm font-medium">Reference image</span>
                <p className="text-xs text-muted-foreground">
                  Upload an image to animate it (image-to-video). Leave empty for text-to-video.
                </p>
              </div>
              {image && (
                <Button variant="ghost" size="sm" onClick={() => setImage(null)}>
                  <X className="h-4 w-4" /> Clear
                </Button>
              )}
            </div>

            {image ? (
              <div className="flex items-center gap-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={image}
                  alt="Reference"
                  className="h-20 w-20 shrink-0 rounded-lg border border-border object-cover"
                />
                <span className="text-sm text-muted-foreground">
                  This frame seeds the first frame of the generated clip.
                </span>
              </div>
            ) : (
              <FileDropzone
                accept="image/*"
                onFiles={onImageFiles}
                icon={<Upload className="h-6 w-6" />}
                label="Drop a reference image"
                hint="PNG or JPG"
              />
            )}
          </Card>

          {/* Dimensions + frames */}
          <Card className="space-y-4 p-5">
            {/* 2×2, not 4-across: in the lg two-column layout this form is only
                half-width, and 4 steppers here squeezed the NumberInput so a
                4-digit value like 1280 clipped to "128". */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Width</label>
                <NumberInput aria-label="Width" value={width} onChange={setWidth} min={64} max={2048} step={64} />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Height</label>
                <NumberInput aria-label="Height" value={height} onChange={setHeight} min={64} max={2048} step={64} />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Frames</label>
                <NumberInput aria-label="Frames" value={numFrames} onChange={setNumFrames} min={1} max={257} step={4} />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">FPS</label>
                <NumberInput aria-label="FPS" value={fps} onChange={setFps} min={1} max={60} step={1} />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Frames are auto-corrected to 4n+1 by the backend (e.g. 81, 85, …).
            </p>

            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Steps</label>
                <NumberInput aria-label="Steps" value={steps} onChange={setSteps} min={1} max={100} step={1} />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Seed</label>
                <NumberInput aria-label="Seed"
                  value={seed ?? 0}
                  onChange={setSeed}
                  min={0}
                  step={1}
                  placeholder="Random"
                />
              </div>
              <div className="space-y-2 sm:col-span-1 col-span-2">
                <div className="flex items-center justify-between text-sm">
                  <label className="font-medium">Guidance</label>
                  <Badge variant="info">{guideScale.toFixed(1)}</Badge>
                </div>
                <Slider label="Guidance"
                  value={[guideScale]}
                  onValueChange={(v) => setGuideScale(v[0] ?? 0)}
                  min={1}
                  max={20}
                  step={0.5}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Scheduler</label>
                <Select value={scheduler} onValueChange={setScheduler}>
                  <SelectTrigger aria-label="Scheduler">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {SCHEDULERS.map((s) => (
                      <SelectItem key={s} value={s}>
                        {s}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Tiling</label>
                <Select value={tiling} onValueChange={setTiling}>
                  <SelectTrigger aria-label="Tiling">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TILING.map((t) => (
                      <SelectItem key={t} value={t}>
                        {t}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Output format</label>
                <SegmentedSelect<"mp4" | "frames">
                  options={FORMATS}
                  value={livePreview ? "frames" : format}
                  onChange={setFormat}
                  disabled={livePreview}
                />
              </div>
              <label className="flex items-center gap-2 text-sm">
                <Switch label="Output format" checked={livePreview} onCheckedChange={setLivePreview} />
                Live preview (stream frames)
              </label>
            </div>
          </Card>

          <Button
            className="w-full"
            onClick={generate}
            disabled={generating || !model || !prompt.trim()}
          >
            {generating ? <Spinner size="sm" /> : <Sparkles className="h-4 w-4" />}
            {generating ? "Generating…" : "Generate video"}
          </Button>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {livePreview && liveFrame && (
            <Card className="space-y-2 overflow-hidden p-3">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Film className="h-4 w-4" /> Live preview
              </div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={liveFrame}
                alt="Latest streamed frame"
                className="w-full rounded-lg border border-border object-contain"
              />
            </Card>
          )}

          <MediaGallery
            items={results}
            title="Results"
            onDownload={download}
            onDelete={remove}
            empty={
              <Card className="flex aspect-video w-full items-center justify-center overflow-hidden bg-muted p-0">
                <EmptyState
                  icon={<Clapperboard className="h-8 w-8" />}
                  title="No videos yet"
                  description="Enter a prompt and generate to see your clips here."
                />
              </Card>
            }
          />
        </div>
      </div>
    </PageShell>
  );
}
