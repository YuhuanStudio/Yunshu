/**
 * Wave 458: Vitest setup file.
 *
 * - Imports `@testing-library/jest-dom` so all suites get the extended
 *   matchers (`toBeInTheDocument`, `toHaveTextContent`, etc.) without
 *   per-file boilerplate.
 * - Stubs `window.confirm` to a vi.fn so destructive-action tests can
 *   assert the prompt fired and choose accept/reject deterministically.
 */
import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
