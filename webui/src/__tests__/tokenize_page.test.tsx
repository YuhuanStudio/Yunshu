/**
 * Wave 463: Component tests for the /tokenize page.
 *
 * Mounts the real TokenizePage (src/app/tokenize/page.tsx) against jsdom
 * with mocked fetch. Exercises all three tabs (Tokenize / Detokenize /
 * Count) and the model select. The page derives its base URL from
 * window.location.origin. Production source is READ ONLY.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TokenizePage from "../app/tokenize/page";

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

describe("TokenizePage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the three tabs and loads models into the select", async () => {
    installFetchMock((url) => {
      if (url.includes("/v1/models")) return jsonResponse({ data: [{ id: "tok-model" }] });
      return jsonResponse({});
    });
    render(<TokenizePage />);
    // Tab labels ("Tokenize"/"Detokenize"/"Count") collide with the action
    // buttons, so each appears >=1 time — assert presence by count.
    expect(screen.getAllByRole("button", { name: /Tokenize/i }).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByRole("button", { name: /Detokenize/i }).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByRole("button", { name: /Count/i }).length).toBeGreaterThanOrEqual(1);
    await waitFor(() => expect(screen.getByText("tok-model")).toBeInTheDocument());
  });

  it("Tokenize tab POSTs to /v1/tokenize and renders the token ids", async () => {
    const { calls } = installFetchMock((url, init) => {
      if (url.includes("/v1/tokenize") && init?.method === "POST") {
        return jsonResponse({ tokens: [101, 7592, 102], model: "tok-model" });
      }
      if (url.includes("/v1/models")) return jsonResponse({ data: [] });
      return jsonResponse({});
    });
    render(<TokenizePage />);
    const user = userEvent.setup();

    await user.type(screen.getByPlaceholderText("Enter text to tokenize..."), "hello");
    // The action button (not the tab) is the second "Tokenize" button.
    const tokenizeButtons = screen.getAllByRole("button", { name: /Tokenize/i });
    await user.click(tokenizeButtons[tokenizeButtons.length - 1]);

    await waitFor(() => expect(screen.getByText("Tokens (3)")).toBeInTheDocument());
    expect(screen.getByText("7592")).toBeInTheDocument();

    const post = calls.find((c) => c.url.includes("/v1/tokenize") && c.init?.method === "POST");
    expect(post).toBeDefined();
    const body = JSON.parse(post!.init!.body!);
    expect(body.text).toBe("hello");
    expect(body.add_special_tokens).toBe(true);
  });

  it("Detokenize tab POSTs parsed ids to /v1/detokenize and shows decoded text", async () => {
    const { calls } = installFetchMock((url, init) => {
      if (url.includes("/v1/detokenize") && init?.method === "POST") {
        return jsonResponse({ text: "hello world", model: "tok-model" });
      }
      if (url.includes("/v1/models")) return jsonResponse({ data: [] });
      return jsonResponse({});
    });
    render(<TokenizePage />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Detokenize/i }));
    await user.type(screen.getByPlaceholderText("e.g. 1234, 5678, 9012"), "101, 7592, 102");
    const detokButtons = screen.getAllByRole("button", { name: /Detokenize/i });
    await user.click(detokButtons[detokButtons.length - 1]);

    await waitFor(() => expect(screen.getByText("hello world")).toBeInTheDocument());

    const post = calls.find((c) => c.url.includes("/v1/detokenize") && c.init?.method === "POST");
    expect(post).toBeDefined();
    const body = JSON.parse(post!.init!.body!);
    expect(body.tokens).toEqual([101, 7592, 102]);
  });

  it("Detokenize tab rejects non-numeric ids without calling fetch", async () => {
    const { calls } = installFetchMock((url) => {
      if (url.includes("/v1/models")) return jsonResponse({ data: [] });
      return jsonResponse({});
    });
    render(<TokenizePage />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Detokenize/i }));
    await user.type(screen.getByPlaceholderText("e.g. 1234, 5678, 9012"), "abc, def");
    const detokButtons = screen.getAllByRole("button", { name: /Detokenize/i });
    await user.click(detokButtons[detokButtons.length - 1]);

    await waitFor(() =>
      expect(screen.getByText(/Invalid token IDs — must be numbers/)).toBeInTheDocument(),
    );
    expect(calls.find((c) => c.url.includes("/v1/detokenize"))).toBeUndefined();
  });

  it("Count tab POSTs to /v1/token_count and shows count + over-limit badge", async () => {
    const { calls } = installFetchMock((url, init) => {
      if (url.includes("/v1/token_count") && init?.method === "POST") {
        return jsonResponse({ token_count: 42, over_context_limit: true });
      }
      if (url.includes("/v1/models")) return jsonResponse({ data: [] });
      return jsonResponse({});
    });
    render(<TokenizePage />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /Count/i }));
    await user.type(screen.getByPlaceholderText("Enter prompt text..."), "some prompt");
    const countButtons = screen.getAllByRole("button", { name: /Count/i });
    await user.click(countButtons[countButtons.length - 1]);

    await waitFor(() => expect(screen.getByText("42")).toBeInTheDocument());
    expect(screen.getByText(/Over limit/)).toBeInTheDocument();

    const post = calls.find((c) => c.url.includes("/v1/token_count") && c.init?.method === "POST");
    expect(post).toBeDefined();
    const body = JSON.parse(post!.init!.body!);
    expect(body.prompt).toBe("some prompt");
  });
});
