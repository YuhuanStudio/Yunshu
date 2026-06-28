/**
 * Wave 463: Component tests for the /embeddings page.
 *
 * Mounts the real EmbeddingsPage (src/app/embeddings/page.tsx) against
 * jsdom with a mocked fetch. The page uses relative URLs (/v1/models,
 * /v1/embeddings). Per strict rules production source is READ ONLY.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import EmbeddingsPage from "../app/embeddings/page";

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

describe("EmbeddingsPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders heading and populates the model select from /v1/models", async () => {
    installFetchMock((url) => {
      if (url.includes("/v1/models")) return jsonResponse({ data: [{ id: "embed-model" }] });
      return jsonResponse({});
    });
    render(<EmbeddingsPage />);
    expect(screen.getByText("Embeddings")).toBeInTheDocument();
    expect(screen.getByText("Default Model")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("embed-model")).toBeInTheDocument());
  });

  it("POSTs to /v1/embeddings and displays the vector dimensions", async () => {
    const embedding = Array.from({ length: 32 }, (_, i) => i / 100);
    const { calls } = installFetchMock((url, init) => {
      if (url.includes("/v1/embeddings") && init?.method === "POST") {
        return jsonResponse({
          data: [{ index: 0, embedding }],
          usage: { total_tokens: 7 },
        });
      }
      if (url.includes("/v1/models")) return jsonResponse({ data: [] });
      return jsonResponse({});
    });
    render(<EmbeddingsPage />);
    const user = userEvent.setup();

    await user.type(
      screen.getByPlaceholderText("Enter text to embed (one per line for batch)..."),
      "hello",
    );
    await user.click(screen.getByRole("button", { name: /Generate Embeddings/i }));

    await waitFor(() => expect(screen.getByText("32 dimensions")).toBeInTheDocument());
    // Header reports count + token usage.
    expect(screen.getByText(/1 embedding, 7 tokens/)).toBeInTheDocument();

    const post = calls.find((c) => c.init?.method === "POST");
    expect(post).toBeDefined();
    expect(post!.url).toContain("/v1/embeddings");
    const body = JSON.parse(post!.init!.body!);
    expect(body.input).toBe("hello");
  });

  it("renders an error box when the API returns an error payload", async () => {
    installFetchMock((url, init) => {
      if (url.includes("/v1/embeddings") && init?.method === "POST") {
        return jsonResponse({ error: { message: "model not loaded" } }, 400);
      }
      return jsonResponse({ data: [] });
    });
    render(<EmbeddingsPage />);
    const user = userEvent.setup();

    await user.type(
      screen.getByPlaceholderText("Enter text to embed (one per line for batch)..."),
      "x",
    );
    await user.click(screen.getByRole("button", { name: /Generate Embeddings/i }));

    await waitFor(() => expect(screen.getByText("model not loaded")).toBeInTheDocument());
  });
});
