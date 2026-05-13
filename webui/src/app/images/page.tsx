"use client";

import { useEffect, useState, useCallback } from "react";
import {
  ImageIcon,
  Download,
  Loader2,
  RefreshCw,
  Sparkles,
  X,
} from "lucide-react";

interface Model {
  id: string;
}

export default function ImagesPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [prompt, setPrompt] = useState(
    "A majestic mountain landscape at golden hour with dramatic clouds"
  );
  const [size, setSize] = useState("1024x1024");
  const [steps, setSteps] = useState(4);
  const [seed, setSeed] = useState<string>("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [genTime, setGenTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);

  useEffect(() => {
    fetch("/v1/models")
      .then((r) => r.json())
      .then((data) => {
        const all = data.data || [];
        setModels(all);
        const img = all.find((m: Model) =>
          /image|z-image|turbo|flux|diffus/i.test(m.id)
        );
        if (img) setSelectedModel(img.id);
      })
      .catch(() => {});
  }, []);

  const generate = useCallback(async () => {
    if (!prompt.trim() || !selectedModel) return;
    setLoading(true);
    setError(null);
    setImageUrl(null);
    setProgress("Starting generation...");
    const start = performance.now();

    try {
      const resp = await fetch("/v1/images/generations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          prompt,
          n: 1,
          size,
          num_inference_steps: steps,
          seed: seed ? parseInt(seed) : undefined,
        }),
      });

      if (resp.ok) {
        const data = await resp.json();
        const img = data.data?.[0];
        if (img?.b64_json) {
          setImageUrl(`data:image/png;base64,${img.b64_json}`);
          setGenTime((performance.now() - start) / 1000);
          setProgress(`Completed in ${((performance.now() - start) / 1000).toFixed(1)}s`);
        } else if (img?.url) {
          setImageUrl(img.url);
          setGenTime((performance.now() - start) / 1000);
          setProgress(`Completed in ${((performance.now() - start) / 1000).toFixed(1)}s`);
        }
      } else {
        setError(`Generation failed: ${resp.status} ${await resp.text()}`);
      }
    } catch (err) {
      setError(`Error: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [prompt, selectedModel, size, steps, seed]);

  const download = useCallback(() => {
    if (!imageUrl) return;
    const a = document.createElement("a");
    a.href = imageUrl;
    a.download = `yunshu-${Date.now()}.png`;
    a.click();
  }, [imageUrl]);

  const imgModels = models.filter((m) =>
    /image|z-image|turbo|flux|diffus/i.test(m.id)
  );

  return (
    <div className="p-6 space-y-6 max-w-4xl page-enter">
      <div className="flex items-center gap-3">
        <ImageIcon className="w-6 h-6 text-rose-400" />
        <h2 className="text-2xl font-bold">Image Generation</h2>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-[var(--color-danger)]/10 border border-[var(--color-danger)]/30 rounded-lg px-4 py-3 text-sm text-[var(--color-danger)] flex items-center gap-2">
          <X className="w-4 h-4 shrink-0" />
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-auto opacity-60 hover:opacity-100"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* Controls */}
      <div className="space-y-4">
        {/* Prompt */}
        <div>
          <label className="text-sm text-[var(--color-text-secondary)] block mb-1.5">
            Prompt
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            className="w-full bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:border-[var(--color-accent)]"
            placeholder="Describe the image you want to generate..."
          />
        </div>

        {/* Parameters row */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Model */}
          <div>
            <label className="text-sm text-[var(--color-text-secondary)] block mb-1.5">
              Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm"
            >
              {imgModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id}
                </option>
              ))}
              {imgModels.length === 0 && (
                <option value="">No image models</option>
              )}
            </select>
          </div>

          {/* Size */}
          <div>
            <label className="text-sm text-[var(--color-text-secondary)] block mb-1.5">
              Size
            </label>
            <select
              value={size}
              onChange={(e) => setSize(e.target.value)}
              className="w-full bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm"
            >
              <option value="512x512">512 x 512</option>
              <option value="768x768">768 x 768</option>
              <option value="1024x1024">1024 x 1024</option>
              <option value="1024x768">1024 x 768</option>
              <option value="768x1024">768 x 1024</option>
            </select>
          </div>

          {/* Steps */}
          <div>
            <label className="text-sm text-[var(--color-text-secondary)] block mb-1.5">
              Steps
            </label>
            <input
              type="number"
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value) || 4)}
              min={1}
              max={50}
              className="w-full bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm"
            />
          </div>

          {/* Seed */}
          <div>
            <label className="text-sm text-[var(--color-text-secondary)] block mb-1.5">
              Seed
            </label>
            <input
              type="text"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="Random"
              className="w-full bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm"
            />
          </div>
        </div>

        {/* Generate button */}
        <button
          onClick={generate}
          disabled={loading || !prompt.trim() || !selectedModel}
          className="w-full flex items-center justify-center gap-2 bg-[var(--color-accent)] text-white rounded-xl px-4 py-3 font-medium hover:bg-[var(--color-accent-hover)] disabled:opacity-40 transition-colors"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              Generate Image
            </>
          )}
        </button>
        {progress && !imageUrl && (
          <div className="text-xs text-[var(--color-text-secondary)] mt-2 animate-pulse">
            {progress}
          </div>
        )}
      </div>

      {/* Result */}
      {imageUrl && (
        <div className="space-y-3">
          <div className="relative rounded-xl overflow-hidden border border-[var(--color-border)] bg-[var(--color-bg-tertiary)]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl}
              alt="Generated"
              className="w-full max-h-[600px] object-contain"
            />
          </div>
          <div className="flex items-center justify-between">
            {genTime != null && (
              <span className="text-sm text-[var(--color-text-secondary)]">
                Generated in {genTime.toFixed(1)}s
              </span>
            )}
            <div className="flex gap-2 ml-auto">
              <button
                onClick={generate}
                disabled={loading}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] hover:bg-[var(--color-bg-secondary)] transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Regenerate
              </button>
              <button
                onClick={download}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] hover:bg-[var(--color-bg-secondary)] transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                Download
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
