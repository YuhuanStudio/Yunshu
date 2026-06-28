/**
 * Wave 463: Component tests for the /models page.
 *
 * Mounts the real ModelsPage (src/app/models/page.tsx) against jsdom with
 * mocked fetch. The page uses relative URLs (/v1/models, /v1/models/load,
 * /v1/models/unload/<id>, /api/v1/admin/models/discover). Asserts list
 * render and that the Load/Unload buttons hit the right endpoints.
 * Production source is READ ONLY.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ModelsPage from "../app/models/page";

type FetchInit = { method?: string; headers?: Record<string, string>; body?: string };

interface FetchCall {
  url: string;
  init?: FetchInit;
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => "",
  } as unknown as Response;
}

function installFetchMock(
  handler: (url: string, init?: FetchInit) => Response,
): { calls: FetchCall[] } {
  const calls: FetchCall[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: FetchInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({ url, init });
    return handler(url, init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls };
}

describe("ModelsPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders model cards from the /v1/models list", async () => {
    installFetchMock((url) => {
      if (url.includes("/v1/models")) {
        return jsonResponse({
          data: [
            { id: "qwen-loaded", loaded: true, size_gb: 1.2 },
            { id: "qwen-available", loaded: false, size_gb: 0.5 },
          ],
        });
      }
      return jsonResponse({});
    });
    render(<ModelsPage />);

    await waitFor(() => expect(screen.getByText("qwen-loaded")).toBeInTheDocument());
    expect(screen.getByText("qwen-available")).toBeInTheDocument();
    expect(screen.getByText("Loaded")).toBeInTheDocument();
    expect(screen.getByText("Available")).toBeInTheDocument();
  });

  it("shows empty-state guidance when no models are registered", async () => {
    installFetchMock((url) => {
      if (url.includes("/v1/models")) return jsonResponse({ data: [] });
      return jsonResponse({});
    });
    render(<ModelsPage />);
    await waitFor(() => expect(screen.getByText(/No models found/)).toBeInTheDocument());
  });

  it("Load button POSTs to /v1/models/load with the model id", async () => {
    const { calls } = installFetchMock((url, init) => {
      if (url.includes("/v1/models/load") && init?.method === "POST") {
        return jsonResponse({ ok: true });
      }
      if (url.includes("/v1/models")) {
        return jsonResponse({ data: [{ id: "qwen-available", loaded: false }] });
      }
      return jsonResponse({});
    });
    render(<ModelsPage />);
    const user = userEvent.setup();

    await waitFor(() => expect(screen.getByText("qwen-available")).toBeInTheDocument());
    // The per-card Load button (the toolbar Load is disabled with empty input).
    const loadButtons = screen.getAllByRole("button", { name: /^Load$/i });
    await user.click(loadButtons[loadButtons.length - 1]);

    await waitFor(() => {
      const post = calls.find(
        (c) => c.url.includes("/v1/models/load") && c.init?.method === "POST",
      );
      expect(post).toBeDefined();
      const body = JSON.parse(post!.init!.body!);
      expect(body.model).toBe("qwen-available");
    });
  });

  it("Unload button POSTs to /v1/models/unload/<id>", async () => {
    const { calls } = installFetchMock((url, init) => {
      if (url.includes("/v1/models/unload/") && init?.method === "POST") {
        return jsonResponse({ ok: true });
      }
      if (url.includes("/v1/models")) {
        return jsonResponse({ data: [{ id: "qwen-loaded", loaded: true }] });
      }
      return jsonResponse({});
    });
    render(<ModelsPage />);
    const user = userEvent.setup();

    await waitFor(() => expect(screen.getByText("qwen-loaded")).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: /Unload/i }));

    await waitFor(() => {
      const post = calls.find(
        (c) => c.url.includes("/v1/models/unload/") && c.init?.method === "POST",
      );
      expect(post).toBeDefined();
      expect(post!.url).toContain("/v1/models/unload/qwen-loaded");
    });
  });

  it("Discover button GETs /api/v1/admin/models/discover and lists results", async () => {
    const { calls } = installFetchMock((url) => {
      if (url.includes("/api/v1/admin/models/discover")) {
        return jsonResponse({
          models: {
            "found-model": { model_type: "LLM", engine_type: "mlx", estimated_size_gb: 2.5 },
          },
        });
      }
      if (url.includes("/v1/models")) return jsonResponse({ data: [] });
      return jsonResponse({});
    });
    render(<ModelsPage />);
    const user = userEvent.setup();

    await user.click(screen.getByTitle("Auto-discover models"));

    await waitFor(() => expect(screen.getByText("found-model")).toBeInTheDocument());
    expect(screen.getByText(/Discovered 1 model/)).toBeInTheDocument();
    expect(
      calls.find((c) => c.url.includes("/api/v1/admin/models/discover")),
    ).toBeDefined();
  });
});
