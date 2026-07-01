"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Button,
  Textarea,
  NumberInput,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  Slider,
  Card,
  Badge,
  Spinner,
  FileDropzone,
  EmptyState,
} from "yunui";
import { MediaGallery, type MediaResult } from "yunui/patterns";
import { ImageIcon, Sparkles, Upload, X } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, ApiError } from "@/lib/api";
import type { Model } from "@/lib/types";

const SIZES = ["512x512", "768x768", "1024x1024"] as const;

interface ImageGenResponse {
  data: { b64_json?: string; url?: string }[];
}

export default function ImagesPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [size, setSize] = useState<string>("1024x1024");
  const [steps, setSteps] = useState(20);
  const [seed, setSeed] = useState<number | undefined>(undefined);

  const [controlImage, setControlImage] = useState<string | null>(null);
  const [controlScale, setControlScale] = useState(0.5);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<MediaResult[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ data: Model[] }>("/v1/models", controller.signal)
      .then((res) => {
        const list = res.data ?? [];
        setModels(list);
        setModel((cur) => cur || list[0]?.id || "");
      })
      .catch(() => {
        /* model list unavailable — surfaced via the global connection status */
      });
    return () => controller.abort();
  }, []);

  const onControlFiles = useCallback((files: File[]) => {
    const file = files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setControlImage(reader.result as string);
    reader.readAsDataURL(file);
  }, []);

  const generate = useCallback(async () => {
    if (!model || !prompt.trim()) return;
    const id = crypto.randomUUID();
    // Prepend a processing placeholder so the gallery shows a spinner card.
    setResults((r) => [
      { id, url: "", kind: "image", prompt, model, meta: size, status: "processing" },
      ...r,
    ]);
    setLoading(true);
    const start = performance.now();
    try {
      const res = await api.post<ImageGenResponse>("/v1/images/generations", {
        model,
        prompt,
        n: 1,
        size,
        num_inference_steps: steps,
        seed,
        // ControlNet: send the raw base64 (strip the `data:` URL prefix).
        control_image: controlImage
          ? controlImage.replace(/^data:[^,]+,/, "")
          : undefined,
        control_scale: controlImage ? controlScale : undefined,
      });
      const img = res.data?.[0];
      const url = img?.b64_json ? `data:image/png;base64,${img.b64_json}` : img?.url;
      if (!url) throw new Error("The response contained no image data.");
      const secs = ((performance.now() - start) / 1000).toFixed(1);
      setResults((r) =>
        r.map((x) =>
          x.id === id ? { ...x, url, status: "completed", meta: `${size} · ${secs}s` } : x,
        ),
      );
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : (e as Error).message;
      setResults((r) => r.map((x) => (x.id === id ? { ...x, status: "failed", error: msg } : x)));
    } finally {
      setLoading(false);
    }
  }, [model, prompt, size, steps, seed, controlImage, controlScale]);

  const download = useCallback((item: MediaResult) => {
    const a = document.createElement("a");
    a.href = item.url;
    a.download = `yunshu-${item.id}.png`;
    a.click();
  }, []);

  const remove = useCallback((item: MediaResult) => {
    setResults((r) => r.filter((x) => x.id !== item.id));
  }, []);

  return (
    <PageShell
      title="Images"
      description="Generate images from a text prompt, with optional ControlNet guidance."
      width="wide"
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Controls */}
        <div className="space-y-6">
          <Card className="space-y-4 p-5">
            <div className="space-y-2">
              <label className="text-sm font-medium">Model</label>
              <ModelPicker models={models} value={model} onChange={setModel} />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Prompt</label>
              <Textarea
                rows={4}
                placeholder="Describe the image you want to generate…"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Size</label>
                <Select value={size} onValueChange={setSize}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {SIZES.map((s) => (
                      <SelectItem key={s} value={s}>
                        {s}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Steps</label>
                <NumberInput value={steps} onChange={setSteps} min={1} max={100} step={1} />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Seed</label>
                <NumberInput
                  value={seed ?? 0}
                  onChange={setSeed}
                  min={0}
                  step={1}
                  placeholder="Random"
                />
              </div>
            </div>
          </Card>

          {/* ControlNet */}
          <Card className="space-y-4 p-5">
            <div className="flex items-center justify-between gap-3">
              <div className="space-y-0.5">
                <span className="text-sm font-medium">ControlNet</span>
                <p className="text-xs text-muted-foreground">
                  Optional control map (canny / depth / pose) to guide the layout.
                </p>
              </div>
              {controlImage && (
                <Button variant="ghost" size="sm" onClick={() => setControlImage(null)}>
                  <X className="h-4 w-4" /> Clear
                </Button>
              )}
            </div>

            {controlImage ? (
              <div className="flex items-center gap-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={controlImage}
                  alt="Control reference"
                  className="h-20 w-20 shrink-0 rounded-lg border border-border object-cover"
                />
                <div className="min-w-0 flex-1 space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <label className="font-medium">Control strength</label>
                    <Badge variant="info">{controlScale.toFixed(2)}</Badge>
                  </div>
                  <Slider
                    value={[controlScale]}
                    onValueChange={(v) => setControlScale(v[0] ?? 0)}
                    min={0}
                    max={1}
                    step={0.05}
                  />
                </div>
              </div>
            ) : (
              <FileDropzone
                accept="image/*"
                onFiles={onControlFiles}
                icon={<Upload className="h-6 w-6" />}
                label="Drop a control image"
                hint="PNG or JPG"
              />
            )}
          </Card>

          <Button className="w-full" onClick={generate} disabled={loading || !model || !prompt.trim()}>
            {loading ? <Spinner size="sm" /> : <Sparkles className="h-4 w-4" />}
            {loading ? "Generating…" : "Generate"}
          </Button>

          {loading && (
            <p className="animate-pulse text-center text-sm text-muted-foreground">
              Rendering your image — this can take a moment…
            </p>
          )}
        </div>

        {/* Results */}
        <MediaGallery
          items={results}
          title="Results"
          onDownload={download}
          onDelete={remove}
          empty={
            <Card className="flex aspect-square w-full items-center justify-center overflow-hidden bg-muted p-0">
              <EmptyState
                icon={<ImageIcon className="h-8 w-8" />}
                title="No images yet"
                description="Enter a prompt and generate to see your results here."
              />
            </Card>
          }
        />
      </div>
    </PageShell>
  );
}
