import { Brain, Eye, Volume2, Mic, ImageIcon, Bot, type LucideIcon } from "lucide-react";
import { cn } from "yunui";

/**
 * Yunshu's model taxonomy. The glyph color uses core Tailwind palette hues
 * (mirroring YunUI's own ModelTypeIcon convention) — a fixed categorical
 * palette, distinct from the semantic tokens used for chrome. The chip chrome
 * stays neutral (muted) so it reads the same in every theme.
 */
export type ModelType = "LLM" | "VLM" | "TTS" | "ASR" | "IMAGE_GEN";

export const MODEL_TYPES: Record<ModelType, { label: string; icon: LucideIcon; color: string }> = {
  LLM: { label: "LLM", icon: Brain, color: "text-blue-500" },
  VLM: { label: "VLM", icon: Eye, color: "text-purple-500" },
  TTS: { label: "TTS", icon: Volume2, color: "text-emerald-500" },
  ASR: { label: "ASR", icon: Mic, color: "text-amber-500" },
  IMAGE_GEN: { label: "Image", icon: ImageIcon, color: "text-rose-500" },
};

/** Best-effort model-type inference from a model id. */
export function guessModelType(id: string): ModelType {
  const l = id.toLowerCase();
  if (l.includes("tts") || l.includes("voice") || l.includes("cosyvoice")) return "TTS";
  if (l.includes("asr") || l.includes("whisper")) return "ASR";
  if (l.includes("vlm") || l.includes("omni") || l.includes("vision") || l.includes("qwen2-vl")) return "VLM";
  if (
    l.includes("image") || l.includes("turbo") || l.includes("flux") ||
    l.includes("sd-") || l.includes("diffusion") || l.includes("dalle")
  )
    return "IMAGE_GEN";
  return "LLM";
}

/** Just the colored type glyph (falls back to a bot for unknown types). */
export function ModelTypeGlyph({ type, size = 16 }: { type: string; size?: number }) {
  const cfg = MODEL_TYPES[type as ModelType];
  if (!cfg) return <Bot size={size} className="text-muted-foreground" />;
  const Icon = cfg.icon;
  return <Icon size={size} className={cfg.color} />;
}

/** A compact type chip: neutral chrome + colored glyph + label. */
export function ModelTypeChip({ type, className }: { type: string; className?: string }) {
  const cfg = MODEL_TYPES[type as ModelType];
  const label = cfg?.label ?? type;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground",
        className,
      )}
    >
      <ModelTypeGlyph type={type} size={13} />
      {label}
    </span>
  );
}

/** A larger square icon tile for cards (colored glyph on a tinted-muted tile). */
export function ModelTypeTile({ type, size = 40 }: { type: string; size?: number }) {
  return (
    <span
      className="inline-flex items-center justify-center rounded-lg bg-muted"
      style={{ width: size, height: size }}
    >
      <ModelTypeGlyph type={type} size={Math.round(size * 0.5)} />
    </span>
  );
}
