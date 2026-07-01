"use client";

/**
 * Auth for Yunshu is a single static Bearer token (`YUNSHU_AUTH_TOKEN` on the
 * server) — there is no user system. When the server has a token configured,
 * every request (and the realtime WS) must send `Authorization: Bearer <token>`
 * to reach anything beyond open inference (model details, monitoring,
 * load/unload). When the server runs with auth disabled, no token is needed.
 *
 * We persist the operator's token in localStorage and attach it everywhere.
 */

const TOKEN_KEY = "yunshu_token";
const EVENT = "yunshu-token-change";

export function getToken(): string {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setToken(token: string): void {
  if (typeof window === "undefined") return;
  try {
    const t = token.trim();
    if (t) window.localStorage.setItem(TOKEN_KEY, t);
    else window.localStorage.removeItem(TOKEN_KEY);
    window.dispatchEvent(new CustomEvent(EVENT));
  } catch {
    /* storage unavailable — token simply won't persist */
  }
}

/** Bearer headers for the current token (empty object when none is set). */
export function authHeaders(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

/** Append the token as a `?token=` param (for the WS, which also accepts it). */
export function withTokenParam(url: string): string {
  const t = getToken();
  if (!t) return url;
  return url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(t);
}

/** Subscribe to token changes (returns an unsubscribe fn). */
export function onTokenChange(fn: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  window.addEventListener(EVENT, fn);
  window.addEventListener("storage", fn);
  return () => {
    window.removeEventListener(EVENT, fn);
    window.removeEventListener("storage", fn);
  };
}
