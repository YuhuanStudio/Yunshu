/**
 * Wave 463: Component tests for the /completions page.
 *
 * Mounts the real CompletionsPage (src/app/completions/page.tsx) against
 * jsdom and exercises the user flow with a mocked global fetch. Per Wave
 * 458/463 strict rules the production source is READ ONLY — we drive it
 * exactly as a user would (type prompt, click Generate) and assert the
 * POST URL/body plus the SSE-streamed result render.
 *
 * The page derives its API base from window.location.origin, so the
 * POST URL is the full `http://localhost/v1/completions` in jsdom.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CompletionsPage from "../app/completions/page";

type FetchInit = { method?: string; headers?: Record<string, string>; body?: string };

interface FetchCall {
  url: string;
  init?: FetchInit;
}

/**
 * Builds a Response-like object whose `body.getReader()` replays the
 * provided SSE chunks, matching what the page's stream loop consumes.
 */
function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  let i = 0;
  const reader = {
    read: async () => {
      if (i >= chunks.length) return { done: true, value: undefined };
      return { done: false, value: encoder.encode(chunks[i++]) };
    },
  };
  return {
    ok: true,
    status: 200,
    body: { getReader: () => reader },
    json: async () => ({}),
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

function modelsResponse(): Response {
  return {
    ok: true,
    status: 200,
    json: async () => ({ data: [{ id: "qwen-0.5b" }] }),
    text: async () => "",
  } as unknown as Response;
}

describe("CompletionsPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the heading and prompt input", async () => {
    installFetchMock((url) => {
      if (url.includes("/v1/models")) return modelsResponse();
      return modelsResponse();
    });
    render(<CompletionsPage />);
    expect(screen.getByText("Completions")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Enter your prompt here...")).toBeInTheDocument();
    // Models load asynchronously into the <select>.
    await waitFor(() => expect(screen.getByText("qwen-0.5b")).toBeInTheDocument());
  });

  it("POSTs to /v1/completions and renders streamed text", async () => {
    const chunks = [
      `data: ${JSON.stringify({ id: "cmpl-1", choices: [{ text: "Hello " }] })}\n`,
      `data: ${JSON.stringify({
        id: "cmpl-1",
        choices: [{ text: "world" }],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
      })}\n`,
      "data: [DONE]\n",
    ];
    const { calls } = installFetchMock((url) => {
      if (url.includes("/v1/completions")) return sseResponse(chunks);
      return modelsResponse();
    });
    render(<CompletionsPage />);
    const user = userEvent.setup();

    await user.type(screen.getByPlaceholderText("Enter your prompt here..."), "Hi");
    await user.click(screen.getByRole("button", { name: /Generate/i }));

    // Streamed text becomes a finalized result card with the full text.
    await waitFor(() => expect(screen.getByText("Hello world")).toBeInTheDocument());

    const post = calls.find((c) => c.init?.method === "POST");
    expect(post).toBeDefined();
    expect(post!.url).toContain("/v1/completions");
    const body = JSON.parse(post!.init!.body!);
    expect(body.prompt).toBe("Hi");
    expect(body.stream).toBe(true);
    expect(body.model).toBe("qwen-0.5b");
  });

  it("shows an Error: prefix when the POST returns a non-ok response", async () => {
    installFetchMock((url) => {
      if (url.includes("/v1/completions")) {
        return {
          ok: false,
          status: 500,
          text: async () => "boom",
          json: async () => ({}),
        } as unknown as Response;
      }
      return modelsResponse();
    });
    render(<CompletionsPage />);
    const user = userEvent.setup();

    await user.type(screen.getByPlaceholderText("Enter your prompt here..."), "trigger error");
    await user.click(screen.getByRole("button", { name: /Generate/i }));

    await waitFor(() => expect(screen.getByText(/Error: boom/)).toBeInTheDocument());
  });

  it("sends stream:false and renders a non-streaming JSON result", async () => {
    const { calls } = installFetchMock((url) => {
      if (url.includes("/v1/completions")) {
        return {
          ok: true,
          status: 200,
          body: null,
          json: async () => ({
            id: "cmpl-ns",
            choices: [{ text: "non-stream answer", index: 0, finish_reason: "stop", logprobs: null }],
            usage: { prompt_tokens: 1, completion_tokens: 3, total_tokens: 4 },
          }),
          text: async () => "",
        } as unknown as Response;
      }
      return modelsResponse();
    });
    render(<CompletionsPage />);
    const user = userEvent.setup();

    // Toggle streaming off. The Toggle's onClick lives on the sliding pill
    // div (sibling of the "Streaming" label span), so click the label's
    // last element child rather than the text span.
    const streamingLabel = screen.getByText("Streaming").closest("label")!;
    await user.click(streamingLabel.querySelector("div")!);
    await user.type(screen.getByPlaceholderText("Enter your prompt here..."), "ask");
    await user.click(screen.getByRole("button", { name: /Generate/i }));

    await waitFor(() => expect(screen.getByText("non-stream answer")).toBeInTheDocument());
    const post = calls.find((c) => c.init?.method === "POST");
    const body = JSON.parse(post!.init!.body!);
    expect(body.stream).toBe(false);
  });
});
