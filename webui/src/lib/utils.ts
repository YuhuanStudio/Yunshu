/**
 * Shared utility functions for the Yunshu WebUI.
 */

export function fmtBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${u[i]}`;
}

export function guessModelType(id: string): string {
  const l = id.toLowerCase();
  if (l.includes("tts") || l.includes("voice") || l.includes("cosyvoice")) return "TTS";
  if (l.includes("asr") || l.includes("whisper")) return "ASR";
  if (l.includes("vlm") || l.includes("omni") || l.includes("vision") || l.includes("qwen2-vl")) return "VLM";
  if (l.includes("image") || l.includes("turbo") || l.includes("flux") || l.includes("sd")) return "IMAGE_GEN";
  return "LLM";
}
