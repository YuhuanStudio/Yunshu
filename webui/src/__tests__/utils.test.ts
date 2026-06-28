import { describe, it, expect } from "vitest";
import { fmtBytes, guessModelType } from "../lib/utils";

/**
 * Wave 442: Unit tests for shared WebUI utilities — closes the
 * VALIDATION_REPORT §4520 "Component-level UI suite still deferred"
 * gap by testing the pure-logic layer that doesn't need DOM rendering.
 */

describe("fmtBytes", () => {
  it("formats 0 bytes", () => {
    expect(fmtBytes(0)).toBe("0 B");
  });

  it("formats sub-KB as bytes", () => {
    expect(fmtBytes(512)).toBe("512.0 B");
  });

  it("formats KB", () => {
    expect(fmtBytes(1024)).toBe("1.0 KB");
    expect(fmtBytes(2048)).toBe("2.0 KB");
  });

  it("formats MB", () => {
    expect(fmtBytes(1024 * 1024)).toBe("1.0 MB");
    expect(fmtBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });

  it("formats GB", () => {
    expect(fmtBytes(1024 ** 3)).toBe("1.0 GB");
  });

  it("formats TB", () => {
    expect(fmtBytes(1024 ** 4)).toBe("1.0 TB");
  });

  it("caps at TB for petabyte+ values", () => {
    const oneEB = 1024 ** 6;
    const formatted = fmtBytes(oneEB);
    expect(formatted).toMatch(/^[\d.]+ TB$/);
  });

  it("handles negative values gracefully", () => {
    expect(fmtBytes(-1)).toBe("0 B");
  });

  it("handles NaN gracefully", () => {
    expect(fmtBytes(NaN)).toBe("0 B");
  });

  it("handles Infinity gracefully", () => {
    expect(fmtBytes(Infinity)).toBe("0 B");
  });
});

describe("guessModelType", () => {
  it("detects TTS by tts/voice/cosyvoice keyword", () => {
    expect(guessModelType("Qwen3-TTS-12Hz-1.7B")).toBe("TTS");
    expect(guessModelType("cosyvoice-300m")).toBe("TTS");
    expect(guessModelType("voice-design")).toBe("TTS");
  });

  it("detects ASR by asr/whisper keyword", () => {
    expect(guessModelType("Qwen3-ASR-1.7B")).toBe("ASR");
    expect(guessModelType("whisper-large-v3")).toBe("ASR");
  });

  it("detects VLM by vlm/omni/vision/qwen2-vl keyword", () => {
    expect(guessModelType("Qwen3-Omni-30B-A3B")).toBe("VLM");
    expect(guessModelType("Qwen2-VL-7B")).toBe("VLM");
    expect(guessModelType("llava-vision-base")).toBe("VLM");
  });

  it("detects IMAGE_GEN by image/turbo/flux/sd keyword", () => {
    expect(guessModelType("Z-Image-Turbo-MLX-4bit")).toBe("IMAGE_GEN");
    expect(guessModelType("FLUX.1-dev")).toBe("IMAGE_GEN");
    expect(guessModelType("stable-diffusion-3-medium")).toBe("IMAGE_GEN");
  });

  it("defaults unknown patterns to LLM", () => {
    expect(guessModelType("Qwen3.5-9B-MLX-bf16")).toBe("LLM");
    expect(guessModelType("Llama-3-70B")).toBe("LLM");
    expect(guessModelType("random-string")).toBe("LLM");
  });

  it("is case-insensitive", () => {
    expect(guessModelType("Q3-TTS-LARGE")).toBe("TTS");
    expect(guessModelType("WHISPER-ZH")).toBe("ASR");
    expect(guessModelType("FluX-Schnell")).toBe("IMAGE_GEN");
  });

  it("handles empty string", () => {
    expect(guessModelType("")).toBe("LLM");
  });
});
