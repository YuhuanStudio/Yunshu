import { requestJson, type Connection } from "./api.ts";

/** What `POST /v1/ocr` reports. `confidence` is deliberately not read: the engine echoes a placeholder. */
export interface OcrResult {
  text: string;
  model: string | null;
  language: string | null;
  promptTokens: number | null;
  completionTokens: number | null;
  imageTokens: number | null;
}

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const rec = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};

/** Tolerant parse: a missing field is unknown (null), never 0; `text` may legitimately be empty. */
export function parseOcr(payload: unknown): OcrResult {
  const r = rec(payload),
    usage = rec(r.usage);
  return {
    text: typeof r.text === "string" ? r.text : "",
    model: typeof r.model === "string" ? r.model : null,
    language: typeof r.language === "string" ? r.language : null,
    promptTokens: num(usage.prompt_tokens ?? r.prompt_tokens),
    completionTokens: num(usage.completion_tokens ?? r.completion_tokens),
    imageTokens: num(usage.image_tokens),
  };
}

/** Image to text through the engine's OCR route (an OCR model, or any loaded VLM as the fallback). */
export async function runOcr(
  connection: Connection,
  file: File,
  model: string,
  signal?: AbortSignal,
): Promise<{ result: OcrResult; ms: number }> {
  const form = new FormData();
  form.append("file", file);
  if (model) form.append("model", model);
  const started = performance.now();
  const payload = await requestJson<unknown>(connection, "/ocr", {
    method: "POST",
    form,
    signal,
    timeoutMs: 180_000,
  });
  return { result: parseOcr(payload), ms: performance.now() - started };
}

/** Loaded models that can answer `/v1/ocr`: OCR engines and VLMs (the route's fallback). */
export const ocrCapable = (m: { type: string; loaded: boolean }): boolean =>
  m.loaded && /ocr|vlm/i.test(m.type);
