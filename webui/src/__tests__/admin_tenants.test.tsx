/**
 * Wave 458: Component tests for the TenantManager surface on /admin.
 *
 * The Admin page renders TenantManager when the "Tenants" tab is active.
 * Rather than refactor production source to export each sub-component,
 * we mount the full page and click into the Tenants tab — this matches
 * how users actually reach the screen and keeps src/app/admin/page.tsx
 * untouched (per Wave 458 strict rules: READ ONLY on production WebUI).
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AdminPage from "../app/admin/page";

type FetchInit = { method?: string; headers?: Record<string, string>; body?: string };

interface FetchCall {
  url: string;
  init?: FetchInit;
}

function installFetchMock(
  routes: Record<string, (init: FetchInit | undefined) => { status?: number; body: unknown }>,
): { calls: FetchCall[] } {
  const calls: FetchCall[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: FetchInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({ url, init });
    // Find longest matching prefix so /tenants matches before /tenants/<id>.
    const sortedKeys = Object.keys(routes).sort((a, b) => b.length - a.length);
    const matched = sortedKeys.find((k) => url.startsWith(k));
    if (!matched) {
      return {
        ok: true,
        status: 200,
        json: async () => ({}),
      } as unknown as Response;
    }
    const { status = 200, body } = routes[matched](init);
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    } as unknown as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls };
}

async function switchToTenantsTab() {
  const user = userEvent.setup();
  // Two "Tenants" texts — the tab button and the table header in API Keys —
  // but only the tab is a <button>. getByRole disambiguates.
  await user.click(screen.getByRole("button", { name: /Tenants/i }));
  return user;
}

describe("TenantManager (admin /tenants tab)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders empty state when API returns no tenants", async () => {
    installFetchMock({
      "/api/v1/admin/tenants": () => ({ body: { tenants: [] } }),
    });
    render(<AdminPage />);
    await switchToTenantsTab();
    await waitFor(() => {
      expect(screen.getByText("No tenants")).toBeInTheDocument();
    });
  });

  it("renders table rows from the mocked tenant list", async () => {
    installFetchMock({
      "/api/v1/admin/tenants": () => ({
        body: {
          tenants: [
            { id: "t-001", name: "AcmeCo", tier: "PRO", quota_rpm: 600, active: true },
            { id: "t-002", name: "BetaCorp", tier: "FREE", quota_rpm: 60, active: false },
          ],
        },
      }),
    });
    render(<AdminPage />);
    await switchToTenantsTab();

    await waitFor(() => {
      expect(screen.getByText("AcmeCo")).toBeInTheDocument();
    });
    expect(screen.getByText("t-001")).toBeInTheDocument();
    // "PRO" / "FREE" appear in both the table cell AND the <option> list in
    // the create form. Two matches is correct — assert count, not uniqueness.
    expect(screen.getAllByText("PRO").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("600")).toBeInTheDocument();
    expect(screen.getByText("BetaCorp")).toBeInTheDocument();
    expect(screen.getByText("60")).toBeInTheDocument();
    // Active column renders ✓ / ✗
    expect(screen.getByText("✓")).toBeInTheDocument();
    expect(screen.getByText("✗")).toBeInTheDocument();
  });

  it("POSTs to /tenants with the entered name+tier+rpm and shows the new api key", async () => {
    const { calls } = installFetchMock({
      "/api/v1/admin/tenants": (init) => {
        if (init?.method === "POST") {
          return { body: { api_key: "sk-newly-minted-XYZ", tenant_id: "t-new" } };
        }
        return { body: { tenants: [] } };
      },
    });
    render(<AdminPage />);
    const user = await switchToTenantsTab();

    await waitFor(() => screen.getByPlaceholderText("Tenant name"));
    await user.type(screen.getByPlaceholderText("Tenant name"), "NewCo");
    await user.type(screen.getByPlaceholderText("Custom RPM (optional)"), "300");
    await user.click(screen.getByRole("button", { name: /^Create$/i }));

    await waitFor(() => {
      expect(screen.getByText("sk-newly-minted-XYZ")).toBeInTheDocument();
    });

    const post = calls.find((c) => c.init?.method === "POST");
    expect(post).toBeDefined();
    const body = JSON.parse(post!.init!.body!);
    expect(body).toEqual({ name: "NewCo", tier: "FREE", requests_per_minute: 300 });
    // Yellow warning box
    expect(screen.getByText(/save it — cannot recover/i)).toBeInTheDocument();
  });

  it("shows graceful 503 message when YUNSHU_TENANT_PATH not configured", async () => {
    installFetchMock({
      "/api/v1/admin/tenants": () => ({
        status: 503,
        body: { detail: "Tenant management not enabled (set YUNSHU_TENANT_PATH)" },
      }),
    });
    render(<AdminPage />);
    await switchToTenantsTab();
    await waitFor(() => {
      expect(
        screen.getByText(/Tenant management not enabled \(set YUNSHU_TENANT_PATH\)/i),
      ).toBeInTheDocument();
    });
  });

  it("prompts via confirm() before DELETE and skips fetch when user cancels", async () => {
    const { calls } = installFetchMock({
      "/api/v1/admin/tenants": () => ({
        body: {
          tenants: [{ id: "t-009", name: "DoomedCo", tier: "FREE", quota_rpm: 60, active: true }],
        },
      }),
    });
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<AdminPage />);
    const user = await switchToTenantsTab();

    await waitFor(() => screen.getByText("DoomedCo"));
    await user.click(screen.getByRole("button", { name: /^delete$/i }));

    expect(confirmSpy).toHaveBeenCalledWith(expect.stringMatching(/Delete tenant t-009/));
    // No DELETE issued because user cancelled
    expect(calls.find((c) => c.init?.method === "DELETE")).toBeUndefined();
  });

  it("issues DELETE /tenants/<id> when user confirms", async () => {
    const { calls } = installFetchMock({
      "/api/v1/admin/tenants/t-009": (init) => {
        if (init?.method === "DELETE") return { body: { ok: true } };
        return { body: {} };
      },
      "/api/v1/admin/tenants": () => ({
        body: {
          tenants: [{ id: "t-009", name: "DoomedCo", tier: "FREE", quota_rpm: 60, active: true }],
        },
      }),
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<AdminPage />);
    const user = await switchToTenantsTab();

    await waitFor(() => screen.getByText("DoomedCo"));
    await user.click(screen.getByRole("button", { name: /^delete$/i }));

    await waitFor(() => {
      const del = calls.find(
        (c) => c.init?.method === "DELETE" && c.url === "/api/v1/admin/tenants/t-009",
      );
      expect(del).toBeDefined();
    });
  });
});
