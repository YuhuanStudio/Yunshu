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
  Alert,
  Spinner,
  FileDropzone,
  EmptyState,
  cn,
} from "yunui";
import {
  ImageIcon,
  Download,
  RefreshCw,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
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
  const [error, setError] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [genTime, setGenTime] = useState<number | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ data: Model[] }>("/v1/models", controller.signal)
      .then((res) => {
        const list = res.data ?? [];
        setModels(list);
        setModel((cur) => cur || list[0]?.id || "");
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError((e as Error).message);
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
    setLoading(true);
    setError(null);
    setImageSrc(null);
    setGenTime(null);
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
      if (img?.b64_json) {
        setImageSrc(`data:image/png;base64,${img.b64_json}`);
      } else if (img?.url) {
        setImageSrc(img.url);
      } else {
        setError("The response contained no image data.");
      }
      setGenTime((performance.now() - start) / 1000);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : (e as Error).message;
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [model, prompt, size, steps, seed, controlImage, controlScale]);

  const download = useCallback(() => {
    if (!imageSrc) return;
    const a = document.createElement("a");
    a.href = imageSrc;
    a.download = `yunshu-${Date.now()}.png`;
    a.click();
  }, [imageSrc]);

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

          {error && (
            <Alert variant="error" title="Generation failed">
              {error}
            </Alert>
          )}
        </div>

        {/* Preview */}
        <div className="space-y-4">
          <Card className="overflow-hidden p-0">
            <div
              className={cn(
                "flex aspect-square w-full items-center justify-center bg-muted",
              )}
            >
              {imageSrc ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={imageSrc}
                  alt="Generated"
                  className="h-full w-full object-contain"
                />
              ) : loading ? (
                <Spinner />
              ) : (
                <EmptyState
                  icon={<ImageIcon className="h-8 w-8" />}
                  title="No image yet"
                  description="Enter a prompt and generate to see your result here."
                />
              )}
            </div>
          </Card>

          {imageSrc && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              {genTime != null && (
                <span className="text-sm text-muted-foreground">
                  Generated in {genTime.toFixed(1)}s
                </span>
              )}
              <div className="ml-auto flex gap-2">
                <Button variant="secondary" size="sm" onClick={generate} disabled={loading}>
                  <RefreshCw className="h-4 w-4" /> Regenerate
                </Button>
                <a href={imageSrc} download={`yunshu-${Date.now()}.png`} onClick={download}>
                  <Button variant="secondary" size="sm">
                    <Download className="h-4 w-4" /> Download
                  </Button>
                </a>
              </div>
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}
