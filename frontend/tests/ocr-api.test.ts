import assert from "node:assert/strict";
import test from "node:test";
import { ocrCapable, parseOcr } from "../src/ocr-api.ts";

test("parseOcr: text and usage read, confidence ignored, unknown stays null", () => {
  const r = parseOcr({
    text: "INVOICE 42",
    model: "glm-ocr",
    confidence: 0.0,
    usage: { prompt_tokens: 120, completion_tokens: 8, image_tokens: 100 },
  });
  assert.equal(r.text, "INVOICE 42");
  assert.equal(r.promptTokens, 120);
  assert.equal(r.imageTokens, 100);
  assert.ok(!("confidence" in r));
  const empty = parseOcr({ text: "" });
  assert.equal(empty.text, "");
  assert.equal(empty.promptTokens, null);
  assert.equal(parseOcr(null).text, "");
});

test("ocrCapable: only loaded OCR engines and VLMs", () => {
  assert.equal(ocrCapable({ type: "OCREngine", loaded: true }), true);
  assert.equal(ocrCapable({ type: "VLMEngine", loaded: true }), true);
  assert.equal(ocrCapable({ type: "VLMEngine", loaded: false }), false);
  assert.equal(ocrCapable({ type: "LLM", loaded: true }), false);
});
