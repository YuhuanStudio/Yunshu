"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  FileDropzone,
  NumberInput,
  SegmentedSelect,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Slider,
  Spinner,
  Switch,
  Textarea,
} from "yunui";
import { MediaGallery, type MediaResult } from "yunui/patterns";
import {
  Blend,
  Boxes,
  Brush,
  ImageIcon,
  Layers,
  Mountain,
  Sparkles,
  SquarePen,
  Upload,
  Wand2,
  X,
} from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { ModelPicker } from "@/components/model-picker";
import { api, ApiError, streamSSE, type SSEChunk } from "@/lib/api";
import type { Model } from "@/lib/types";

const SIZES = ["512x512", "768x768", "1024x1024"] as const;

/** The seven backend image endpoints, driven off a single `mode`. */
type Mode = "generate" | "variations" | "edits" | "inpaint" | "controlnet" | "depth";

interface ModeDef {
  value: Mode;
  label: string;
  icon: typeof Sparkles;
  desc: string;
}

const MODES: ModeDef[] = [
  { value: "generate", label: "Generate", icon: Sparkles, desc: "Text → image (T2I)" },
  { value: "variations", label: "Variations", icon: Boxes, desc: "Reimagine an image (no prompt)" },
  { value: "edits", label: "Edit", icon: SquarePen, desc: "Image + prompt (img2img)" },
  { value: "inpaint", label: "Inpaint", icon: Brush, desc: "Repaint a masked region" },
  { value: "controlnet", label: "ControlNet", icon: Layers, desc: "Structure-guided generation" },
  { value: "depth", label: "Depth", icon: Mountain, desc: "Depth-map guided generation" },
];

/** Per-mode default step / denoise values, applied when the mode changes. */
const MODE_DEFAULTS: Record<Mode, { steps: number; denoise: number }> = {
  generate: { steps: 4, denoise: 0.75 },
  variations: { steps: 4, denoise: 0.45 },
  edits: { steps: 4, denoise: 0.8 },
  inpaint: { steps: 8, denoise: 0.75 },
  controlnet: { steps: 4, denoise: 0.75 },
  depth: { steps: 4, denoise: 0.75 },
};

interface ImageData {
  b64_json?: string;
  url?: string;
}
interface ImageGenResponse {
  data: ImageData[];
}

/** Strip the `data:<mime>;base64,` prefix so the backend receives raw base64. */
const rawB64 = (d: string | null): string | undefined =>
  d ? d.replace(/^data:[^,]+,/, "") : undefined;

/** A labelled image drop target with a thumbnail preview + clear control. */
function ImageDrop({
  value,
  onChange,
  label,
  hint,
}: {
  value: string | null;
  onChange: (dataUrl: string | null) => void;
  label: string;
  hint?: string;
}) {
  const onFiles = useCallback(
    (files: File[]) => {
      const file = files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => onChange(reader.result as string);
      reader.readAsDataURL(file);
    },
    [onChange],
  );

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">{label}</label>
        {value && (
          <Button variant="ghost" size="sm" onClick={() => onChange(null)}>
            <X className="h-4 w-4" /> Clear
          </Button>
        )}
      </div>
      {value ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={value}
          alt={label}
          className="h-28 w-28 rounded-lg border border-border object-cover"
        />
      ) : (
        <FileDropzone
          accept="image/*"
          onFiles={onFiles}
          icon={<Upload className="h-6 w-6" />}
          label={`Drop ${label.toLowerCase()}`}
          hint={hint ?? "PNG or JPG"}
        />
      )}
    </div>
  );
}

export default function ImagesPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("");

  const [mode, setMode] = useState<Mode>("generate");

  // Shared controls.
  const [prompt, setPrompt] = useState("");
  const [size, setSize] = useState<string>("1024x1024");
  const [seed, setSeed] = useState<number | undefined>(undefined);

  // Per-mode numeric controls.
  const [steps, setSteps] = useState(4);
  const [denoise, setDenoise] = useState(0.75);
  const [controlScale, setControlScale] = useState(0.8);
  const [conditionType, setConditionType] = useState<"canny" | "depth" | "raw">("canny");
  const [controlnetStrength, setControlnetStrength] = useState(1.0);
  const [cannyLow, setCannyLow] = useState(100);
  const [cannyHigh, setCannyHigh] = useState(200);
  const [depthStrength, setDepthStrength] = useState(1.0);

  // Live preview (streaming) — only for Generate.
  const [livePreview, setLivePreview] = useState(false);
  const [previewInterval, setPreviewInterval] = useState(5);

  // Image inputs (data URLs; base64 stripped at send time).
  const [image, setImage] = useState<string | null>(null); // variations/edits/inpaint/controlnet
  const [controlImage, setControlImage] = useState<string | null>(null); // generate (optional)
  const [depthImage, setDepthImage] = useState<string | null>(null); // depth
  const [mask, setMask] = useState<string | null>(null); // inpaint (optional)

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<MediaResult[]>([]);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .get<{ data: Model[] }>("/v1/models", controller.signal)
      .then((res) => {
        const list = res.data ?? [];
        setModels(list);
        setModel((cur) => cur || list[0]?.id || "Z-Image-Turbo-MLX-4bit");
      })
      .catch(() => {
        setModel((cur) => cur || "Z-Image-Turbo-MLX-4bit");
      });
    return () => controller.abort();
  }, []);

  const changeMode = useCallback((m: Mode) => {
    setMode(m);
    const d = MODE_DEFAULTS[m];
    setSteps(d.steps);
    setDenoise(d.denoise);
  }, []);

  // Which controls apply to the active mode.
  const showPrompt = mode !== "variations";
  const showImage = mode === "variations" || mode === "edits" || mode === "inpaint" || mode === "controlnet";
  const showSteps = mode === "generate" || mode === "variations" || mode === "edits" || mode === "inpaint";
  const showDenoise = mode === "variations" || mode === "edits" || mode === "inpaint";

  const download = useCallback((item: MediaResult) => {
    const a = document.createElement("a");
    a.href = item.url;
    a.download = `yunshu-${item.id}.png`;
    a.click();
  }, []);

  const remove = useCallback((item: MediaResult) => {
    setResults((r) => r.filter((x) => x.id !== item.id));
  }, []);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  const run = useCallback(async () => {
    if (!model) return;
    if (showPrompt && !prompt.trim()) return;
    if (showImage && !image) return;
    if (mode === "depth" && !depthImage) return;

    const def = MODES.find((m) => m.value === mode)!;
    const id = crypto.randomUUID();
    const streaming = mode === "generate" && livePreview;

    setResults((r) => [
      {
        id,
        url: "",
        kind: "image",
        prompt: prompt.trim() || def.label,
        model,
        meta: `${def.label} · ${size}`,
        status: "processing",
        progress: 0,
      },
      ...r,
    ]);
    setLoading(true);
    const start = performance.now();

    const base = { model, size, n: 1, response_format: "b64_json", seed };

    try {
      if (streaming) {
        controllerRef.current = new AbortController();
        await streamSSE(
          "/v1/images/generations/stream",
          {
            ...base,
            prompt,
            num_inference_steps: steps,
            control_image: rawB64(controlImage),
            control_scale: controlImage ? controlScale : undefined,
            preview_interval: previewInterval,
          },
          (chunk: SSEChunk) => handleStreamFrame(id, chunk, start, size),
          controllerRef.current.signal,
        );
        return;
      }

      let path = "";
      let body: Record<string, unknown> = {};
      switch (mode) {
        case "generate":
          path = "/v1/images/generations";
          body = {
            ...base,
            prompt,
            num_inference_steps: steps,
            control_image: rawB64(controlImage),
            control_scale: controlImage ? controlScale : undefined,
          };
          break;
        case "variations":
          path = "/v1/images/variations";
          body = { ...base, image: rawB64(image), denoise_strength: denoise, steps };
          break;
        case "edits":
          path = "/v1/images/edits";
          body = { ...base, prompt, image: rawB64(image), denoise_strength: denoise, steps };
          break;
        case "inpaint":
          path = "/v1/images/inpaint";
          body = {
            ...base,
            prompt,
            image: rawB64(image),
            mask: rawB64(mask),
            denoise_strength: denoise,
            steps,
          };
          break;
        case "controlnet":
          path = "/v1/images/controlnet";
          body = {
            ...base,
            prompt,
            image: rawB64(image),
            condition_type: conditionType,
            controlnet_strength: controlnetStrength,
            canny_low: conditionType === "canny" ? cannyLow : undefined,
            canny_high: conditionType === "canny" ? cannyHigh : undefined,
          };
          break;
        case "depth":
          path = "/v1/images/depth-guided";
          body = { ...base, prompt, depth_image: rawB64(depthImage), depth_strength: depthStrength };
          break;
      }

      const res = await api.post<ImageGenResponse>(path, body);
      const img0 = res.data?.[0];
      const url = img0?.b64_json ? `data:image/png;base64,${img0.b64_json}` : img0?.url;
      if (!url) throw new Error("The response contained no image data.");
      const secs = ((performance.now() - start) / 1000).toFixed(1);
      setResults((r) =>
        r.map((x) =>
          x.id === id
            ? { ...x, url, status: "completed", progress: 100, meta: `${def.label} · ${size} · ${secs}s` }
            : x,
        ),
      );
    } catch (e) {
      const cancelled = (e as Error)?.name === "AbortError";
      const msg = cancelled
        ? "Cancelled."
        : e instanceof ApiError
          ? e.message
          : (e as Error).message;
      setResults((r) =>
        r.map((x) => (x.id === id ? { ...x, status: "failed", error: msg } : x)),
      );
    } finally {
      setLoading(false);
      controllerRef.current = null;
    }
  }, [
    model,
    mode,
    prompt,
    size,
    seed,
    steps,
    denoise,
    controlScale,
    conditionType,
    controlnetStrength,
    cannyLow,
    cannyHigh,
    depthStrength,
    livePreview,
    previewInterval,
    image,
    controlImage,
    depthImage,
    mask,
    showPrompt,
    showImage,
  ]);

  /** Apply one SSE frame from `/generations/stream` to the processing item. */
  const handleStreamFrame = useCallback(
    (id: string, chunk: SSEChunk, start: number, sizeStr: string) => {
      if (chunk.type === "cancelled") {
        setResults((r) =>
          r.map((x) => (x.id === id ? { ...x, status: "failed", error: "Cancelled." } : x)),
        );
        return;
      }
      const b64 = typeof chunk.image === "string" ? chunk.image : undefined;
      if (chunk.is_final && b64) {
        const secs = ((performance.now() - start) / 1000).toFixed(1);
        setResults((r) =>
          r.map((x) =>
            x.id === id
              ? {
                  ...x,
                  url: `data:image/png;base64,${b64}`,
                  status: "completed",
                  progress: 100,
                  meta: `Live · ${sizeStr} · ${secs}s`,
                }
              : x,
          ),
        );
        return;
      }
      if (chunk.is_preview && b64) {
        // Swap in the preview thumbnail while keeping the processing state.
        setResults((r) =>
          r.map((x) => (x.id === id ? { ...x, url: `data:image/png;base64,${b64}` } : x)),
        );
        return;
      }
      // Progress frame: {step, total_steps, progress, is_final}.
      const step = typeof chunk.step === "number" ? chunk.step : undefined;
      const total = typeof chunk.total_steps === "number" ? chunk.total_steps : undefined;
      let pct: number | undefined;
      if (total && total > 0 && step !== undefined) pct = (step / total) * 100;
      else if (typeof chunk.progress === "number")
        pct = chunk.progress <= 1 ? chunk.progress * 100 : chunk.progress;
      if (pct !== undefined) {
        const clamped = Math.max(0, Math.min(100, pct));
        setResults((r) => r.map((x) => (x.id === id ? { ...x, progress: clamped } : x)));
      }
    },
    [],
  );

  const activeMode = MODES.find((m) => m.value === mode)!;
  const canRun =
    !loading &&
    !!model &&
    (!showPrompt || !!prompt.trim()) &&
    (!showImage || !!image) &&
    (mode !== "depth" || !!depthImage);

  return (
    <PageShell
      title="Images"
      description="Generate and transform images across all seven modes — text-to-image, variations, edits, inpainting, ControlNet, and depth-guided."
      width="wide"
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Controls */}
        <div className="space-y-6">
          {/* Mode switcher */}
          <Card className="space-y-3 p-5">
            <label className="text-sm font-medium">Mode</label>
            <SegmentedSelect<Mode>
              options={MODES.map((m) => ({ value: m.value, label: m.label, desc: m.desc, icon: m.icon }))}
              value={mode}
              onChange={changeMode}
            />
            <p className="text-xs text-muted-foreground">{activeMode.desc}</p>
          </Card>

          {/* Shared: model / size / seed */}
          <Card className="space-y-4 p-5">
            <div className="space-y-2">
              <label className="text-sm font-medium">Model</label>
              <ModelPicker models={models} value={model} onChange={setModel} />
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Size</label>
                <Select value={size} onValueChange={setSize}>
                  <SelectTrigger aria-label="Size">
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
                <label className="text-sm font-medium">Seed</label>
                <NumberInput aria-label="Seed"
                  value={seed ?? 0}
                  onChange={setSeed}
                  min={0}
                  step={1}
                  placeholder="Random"
                />
              </div>
            </div>
          </Card>

          {/* Prompt (all modes except Variations) */}
          {showPrompt && (
            <Card className="space-y-2 p-5">
              <label className="text-sm font-medium">Prompt</label>
              <Textarea aria-label="Prompt"
                rows={4}
                placeholder="Describe the image you want…"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </Card>
          )}

          {/* Image inputs */}
          {(showImage || mode === "generate" || mode === "depth") && (
            <Card className="space-y-4 p-5">
              {showImage && (
                <ImageDrop
                  value={image}
                  onChange={setImage}
                  label={mode === "controlnet" ? "Source image" : "Input image"}
                  hint="Required · PNG or JPG"
                />
              )}
              {mode === "generate" && (
                <ImageDrop
                  value={controlImage}
                  onChange={setControlImage}
                  label="Control image"
                  hint="Optional · guides the layout"
                />
              )}
              {mode === "depth" && (
                <ImageDrop
                  value={depthImage}
                  onChange={setDepthImage}
                  label="Depth image"
                  hint="Required · a depth map"
                />
              )}
              {mode === "inpaint" && (
                <ImageDrop
                  value={mask}
                  onChange={setMask}
                  label="Mask"
                  hint="Optional · white = repaint, black = keep"
                />
              )}
            </Card>
          )}

          {/* Per-mode parameters */}
          <Card className="space-y-4 p-5">
            {/* Live preview (Generate only) */}
            {mode === "generate" && (
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="space-y-0.5">
                    <span className="text-sm font-medium">Live preview</span>
                    <p className="text-xs text-muted-foreground">
                      Stream step-by-step previews while the image renders.
                    </p>
                  </div>
                  <Switch label="Live preview" checked={livePreview} onCheckedChange={setLivePreview} />
                </div>
                {livePreview && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Preview interval (steps)</label>
                    <NumberInput aria-label="Preview interval (steps)"
                      value={previewInterval}
                      onChange={(v) => setPreviewInterval(v ?? 1)}
                      min={1}
                      max={50}
                      step={1}
                    />
                  </div>
                )}
              </div>
            )}

            {showSteps && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Steps</label>
                <NumberInput aria-label="Steps"
                  value={steps}
                  onChange={(v) => setSteps(v ?? 1)}
                  min={1}
                  max={100}
                  step={1}
                />
              </div>
            )}

            {showDenoise && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <label className="font-medium">Denoise strength</label>
                  <Badge variant="info">{denoise.toFixed(2)}</Badge>
                </div>
                <Slider label="Denoise strength"
                  value={[denoise]}
                  onValueChange={(v) => setDenoise(v[0] ?? 0)}
                  min={0}
                  max={1}
                  step={0.05}
                />
              </div>
            )}

            {mode === "generate" && controlImage && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <label className="font-medium">Control scale</label>
                  <Badge variant="info">{controlScale.toFixed(2)}</Badge>
                </div>
                <Slider label="Control scale"
                  value={[controlScale]}
                  onValueChange={(v) => setControlScale(v[0] ?? 0)}
                  min={0}
                  max={2}
                  step={0.05}
                />
              </div>
            )}

            {mode === "controlnet" && (
              <>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Condition type</label>
                  <SegmentedSelect<"canny" | "depth" | "raw">
                    options={[
                      { value: "canny", label: "Canny" },
                      { value: "depth", label: "Depth" },
                      { value: "raw", label: "Raw" },
                    ]}
                    value={conditionType}
                    onChange={setConditionType}
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <label className="font-medium">ControlNet strength</label>
                    <Badge variant="info">{controlnetStrength.toFixed(2)}</Badge>
                  </div>
                  <Slider label="ControlNet strength"
                    value={[controlnetStrength]}
                    onValueChange={(v) => setControlnetStrength(v[0] ?? 0)}
                    min={0}
                    max={2}
                    step={0.05}
                  />
                </div>
                {conditionType === "canny" && (
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Canny low</label>
                      <NumberInput aria-label="Canny low"
                        value={cannyLow}
                        onChange={(v) => setCannyLow(v ?? 0)}
                        min={0}
                        max={255}
                        step={1}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Canny high</label>
                      <NumberInput aria-label="Canny high"
                        value={cannyHigh}
                        onChange={(v) => setCannyHigh(v ?? 0)}
                        min={0}
                        max={255}
                        step={1}
                      />
                    </div>
                  </div>
                )}
              </>
            )}

            {mode === "depth" && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <label className="font-medium">Depth strength</label>
                  <Badge variant="info">{depthStrength.toFixed(2)}</Badge>
                </div>
                <Slider label="Depth strength"
                  value={[depthStrength]}
                  onValueChange={(v) => setDepthStrength(v[0] ?? 0)}
                  min={0}
                  max={2}
                  step={0.05}
                />
              </div>
            )}
          </Card>

          <div className="flex gap-3">
            <Button className="flex-1" onClick={run} disabled={!canRun}>
              {loading ? <Spinner size="sm" /> : <Wand2 className="h-4 w-4" />}
              {loading ? "Working…" : `Run ${activeMode.label}`}
            </Button>
            {loading && mode === "generate" && livePreview && (
              <Button variant="outline" onClick={cancel}>
                <X className="h-4 w-4" /> Cancel
              </Button>
            )}
          </div>

          {loading && (
            <p className="animate-pulse text-center text-sm text-muted-foreground">
              <Blend className="mr-1 inline h-4 w-4" />
              Rendering — this can take a moment…
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
                description="Pick a mode, fill in the inputs, and run to see your results here."
              />
            </Card>
          }
        />
      </div>
    </PageShell>
  );
}
