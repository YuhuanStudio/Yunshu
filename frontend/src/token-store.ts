/**
 * Opt-in memory for the engine access token. Off by default: the token lives in page memory only.
 * The Settings connection card turns it on ("remember on this device"); App reads it at startup.
 * Every storage call is guarded, because storage may be blocked or full.
 */
const KEY = "yunshu.console.rememberedToken";

export interface RememberedToken {
  baseUrl: string;
  token: string;
}

function store(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

const normalise = (baseUrl: string) => baseUrl.trim().replace(/\/+$/, "");

/** The remembered token for this server address, or "" (a token is bound to the address it was saved for). */
export function rememberedToken(baseUrl: string): string {
  try {
    const raw = store()?.getItem(KEY);
    if (!raw) return "";
    const parsed = JSON.parse(raw) as Partial<RememberedToken>;
    return typeof parsed.token === "string" &&
      typeof parsed.baseUrl === "string" &&
      normalise(parsed.baseUrl) === normalise(baseUrl)
      ? parsed.token
      : "";
  } catch {
    return "";
  }
}

export function isTokenRemembered(): boolean {
  try {
    return Boolean(store()?.getItem(KEY));
  } catch {
    return false;
  }
}

/** Remember a token (non-empty) for an address, or forget it (empty / null). Returns false if storage failed. */
export function setRememberedToken(
  baseUrl: string,
  token: string | null,
): boolean {
  const s = store();
  if (!s) return false;
  try {
    if (token) {
      s.setItem(KEY, JSON.stringify({ baseUrl: normalise(baseUrl), token }));
    } else s.removeItem(KEY);
    return true;
  } catch {
    return false;
  }
}

export const forgetRememberedToken = () => setRememberedToken("", null);
