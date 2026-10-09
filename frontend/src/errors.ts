import { t } from "./i18n/index.ts";

/** Localised copy for API failures. Raw backend text is kept apart as the details. */
export type FailureKind = "network" | "timeout" | "json" | "empty" | "field";

export function statusMessage(status: number): string {
  if (status === 401) return t("errors.status.401");
  if (status === 403) return t("errors.status.403");
  if (status === 404) return t("errors.status.404");
  if (status === 409) return t("errors.status.409");
  if (status === 429) return t("errors.status.429");
  if (status === 400 || status === 422) return t("errors.status.rejected");
  if (status === 502 || status === 504) return t("errors.failure.network");
  if (status >= 500) return t("errors.status.server", { status });
  return t("errors.status.other", { status });
}

export function failureMessage(kind: FailureKind): string {
  switch (kind) {
    case "network":
      return t("errors.failure.network");
    case "timeout":
      return t("errors.failure.timeout");
    case "json":
      return t("errors.failure.json");
    case "empty":
      return t("errors.failure.empty");
    case "field":
      return t("errors.failure.field");
  }
}

/** Raw text for the expandable details, or undefined when there is nothing extra. */
export function detailText(
  error: unknown,
  statusText?: string,
): string | undefined {
  const parts: string[] = [];
  const raw = (error as { detail?: unknown } | null)?.detail;
  if (typeof raw === "string" && raw.trim()) parts.push(raw.trim());
  else if (raw != null && typeof raw === "object")
    parts.push(JSON.stringify(raw));
  const status = (error as { status?: unknown } | null)?.status;
  if (typeof status === "number")
    parts.unshift(`HTTP ${status}${statusText ? ` ${statusText}` : ""}`);
  const cause = (error as { cause?: unknown } | null)?.cause;
  if (cause instanceof Error && cause.message) parts.push(cause.message);
  return parts.length ? parts.join("\n") : undefined;
}

/** Why the console shows the engine as not online: banner title plus the two-word pill value. */
export function offlineCause(
  phase: "connecting" | "online" | "offline" | "unauthorized",
  status: number | null | undefined,
): { title: string; short: string; hint: string } {
  if (phase === "connecting")
    return {
      title: t("errors.offline.connecting.title"),
      short: t("errors.offline.connecting.short"),
      hint: t("errors.offline.connecting.hint"),
    };
  if (phase === "unauthorized" || status === 401 || status === 403)
    return {
      title: t("errors.offline.unauthorized.title"),
      short: t("errors.offline.unauthorized.short"),
      hint: t("errors.offline.unauthorized.hint"),
    };
  // A proxy that cannot reach the engine answers 502 (refused) or 504 (timeout)
  // itself; that is "can't reach the engine", not an engine fault.
  const proxyDown = status === 502 || status === 504;
  if (typeof status === "number" && status >= 500 && !proxyDown)
    return {
      title: t("errors.offline.server.title"),
      short: t("errors.offline.server.short"),
      hint: t("errors.offline.server.hint", { status }),
    };
  if (typeof status === "number" && !proxyDown)
    return {
      title: t("errors.offline.http.title", { status }),
      short: t("errors.offline.http.short", { status }),
      hint: t("errors.offline.http.hint", { status }),
    };
  return {
    title: t("errors.offline.unreachable.title"),
    short: t("errors.offline.unreachable.short"),
    hint: t("errors.offline.unreachable.hint"),
  };
}
