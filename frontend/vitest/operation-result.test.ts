import { expect, test } from "vitest";
import { operationResult } from "../src/operation-result";

test("partial failures remain visible even when warmup skipped generation", () => {
  const result = operationResult("warmup:model", { generated: false, warning: "load failed" });
  expect(result.error).toBe(true);
  expect(result.text).toContain("load failed");
});
