import { describe, it, expect } from "vitest";
import { fmtBytes, fmtNumber, fmtPct, fmtDuration, fmtClock } from "../lib/format";
import { guessModelType, MODEL_TYPES } from "../lib/model-type";

describe("fmtBytes", () => {
  it("handles zero and negatives", () => {
    expect(fmtBytes(0)).toBe("0 B");
    expect(fmtBytes(-5)).toBe("0 B");
  });
  it("scales units", () => {
    expect(fmtBytes(1536)).toBe("1.5 KB");
    expect(fmtBytes(5 * 1024 ** 3)).toBe("5.0 GB");
  });
});

describe("fmtNumber / fmtPct", () => {
  it("adds separators and rounds", () => {
    expect(fmtNumber(48213)).toBe("48,213");
  });
  it("clamps percentages", () => {
    expect(fmtPct(72.345)).toBe("72.3%");
    expect(fmtPct(120)).toBe("100.0%");
    expect(fmtPct(-4)).toBe("0.0%");
  });
});

describe("fmtDuration / fmtClock", () => {
  it("formats durations", () => {
    expect(fmtDuration(45)).toBe("45s");
    expect(fmtDuration(200)).toBe("3m 20s");
    expect(fmtDuration(3720)).toBe("1h 02m");
  });
  it("formats a media clock", () => {
    expect(fmtClock(0)).toBe("0:00");
    expect(fmtClock(75)).toBe("1:15");
  });
});

describe("guessModelType", () => {
  it("classifies by id heuristics", () => {
    expect(guessModelType("cosyvoice-tts")).toBe("TTS");
    expect(guessModelType("whisper-large")).toBe("ASR");
    expect(guessModelType("qwen2-vl-7b")).toBe("VLM");
    expect(guessModelType("flux-schnell")).toBe("IMAGE_GEN");
    expect(guessModelType("qwen2.5-3b-instruct")).toBe("LLM");
  });
  it("every returned type has a config entry", () => {
    for (const id of ["a-tts", "whisper", "vlm-x", "flux", "plain"]) {
      expect(MODEL_TYPES[guessModelType(id)]).toBeTruthy();
    }
  });
});
