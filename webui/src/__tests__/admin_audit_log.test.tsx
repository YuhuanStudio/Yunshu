/**
 * Wave 458: Component tests for the AuditLogViewer surface on /admin.
 *
 * Same approach as admin_tenants — mount the full Admin page and click
 * into the "Audit Log" tab. We assert (a) the empty-state row, (b) row
 * rendering for mocked events, and (c) that filter+limit inputs cause
 * the component to refetch with the matching query string.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AdminPage from "../app/admin/page";

interface FetchCall {
  url: string;
  init?: { method?: string };
}

function installFetchMock(handler: (url: string) => unknown): { calls: FetchCall[] } {
  const calls: FetchCall[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: { method?: string }) => {
      const url = typeof input === "string" ? input : input.toString();
      calls.push({ url, init });
      const body = handler(url);
      return {
        ok: true,
        status: 200,
        json: async () => body ?? {},
      } as unknown as Response;
    }),
  );
  return { calls };
}

async function switchToAuditTab() {
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: /Audit Log/i }));
  return user;
}

describe("AuditLogViewer (admin /audit tab)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders empty state when no audit events are returned", async () => {
    installFetchMock(() => ({ events: [] }));
    render(<AdminPage />);
    await switchToAuditTab();
    await waitFor(() => {
      expect(screen.getByText("No audit events")).toBeInTheDocument();
    });
  });

  it("renders table rows from mocked audit events", async () => {
    installFetchMock(() => ({
      events: [
        {
          ts: "2026-05-28T12:34:56.789Z",
          op: "tenant_create",
          actor: "admin@example.com",
          resource: "tenant:t-042",
          result: "success",
        },
        {
          ts: "2026-05-28T12:35:00.000Z",
          op: "tenant_delete",
          actor: "ops@example.com",
          resource: "tenant:t-007",
          result: "failure",
        },
      ],
    }));
    render(<AdminPage />);
    await switchToAuditTab();

    await waitFor(() => {
      expect(screen.getByText("tenant_create")).toBeInTheDocument();
    });
    expect(screen.getByText("admin@example.com")).toBeInTheDocument();
    expect(screen.getByText("tenant:t-042")).toBeInTheDocument();
    expect(screen.getByText("success")).toBeInTheDocument();
    expect(screen.getByText("failure")).toBeInTheDocument();
    // Timestamp truncated to first 19 chars
    expect(screen.getByText("2026-05-28T12:34:56")).toBeInTheDocument();
  });

  it("changing the op filter triggers a refetch with ?op=<value>", async () => {
    const { calls } = installFetchMock(() => ({ events: [] }));
    render(<AdminPage />);
    const user = await switchToAuditTab();

    // Initial fetch (tab switch)
    await waitFor(() => {
      expect(
        calls.some((c) => c.url.startsWith("/api/v1/admin/audit-log") && !c.url.includes("op=")),
      ).toBe(true);
    });

    const filterInput = screen.getByPlaceholderText(/Filter by op/i);
    await user.type(filterInput, "tenant_create");

    await waitFor(() => {
      expect(
        calls.some((c) => c.url.includes("op=tenant_create")),
      ).toBe(true);
    });
  });

  it("changing the limit triggers a refetch with the new limit", async () => {
    const { calls } = installFetchMock(() => ({ events: [] }));
    render(<AdminPage />);
    await switchToAuditTab();

    await waitFor(() => {
      expect(calls.some((c) => c.url.includes("limit=50"))).toBe(true);
    });

    // Use fireEvent.change for a direct value swap — user.type on a number
    // input goes through clear→keystroke states (50→""→"2"→"20"→"200") and
    // the component's `parseInt(value)||50` collapses the empty-string state
    // back to 50 on every effect run, racing with the final 200.
    const limitInput = screen.getByDisplayValue("50") as HTMLInputElement;
    fireEvent.change(limitInput, { target: { value: "200" } });

    await waitFor(() => {
      expect(calls.some((c) => c.url.includes("limit=200"))).toBe(true);
    });
  });
});
