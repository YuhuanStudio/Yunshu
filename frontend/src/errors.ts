/** zh-TW copy for API failures. Raw backend text is kept apart as the 詳細資訊. */
export type FailureKind = "network" | "timeout" | "json" | "empty" | "field";

export function statusMessage(status: number): string {
  if (status === 401) return "驗證失敗，請檢查存取權杖。";
  if (status === 403) return "沒有執行此操作的權限。";
  if (status === 404) return "找不到資源，此引擎版本可能尚未提供這個介面。";
  if (status === 409) return "操作與引擎目前的狀態衝突，請確認後再試。";
  if (status === 429) return "請求過於頻繁或引擎忙碌，請稍後再試。";
  if (status === 400 || status === 422)
    return "引擎拒絕了這個請求，請檢查輸入內容。";
  if (status >= 500) return `引擎內部錯誤（HTTP ${status}），請查看引擎日誌。`;
  return `引擎回傳 HTTP ${status}。`;
}

export function failureMessage(kind: FailureKind): string {
  switch (kind) {
    case "network":
      return "無法連線到引擎，請確認服務位址與引擎是否在執行。";
    case "timeout":
      return "引擎回應逾時，請稍後再試。";
    case "json":
      return "引擎回傳了無法解析的內容（不是有效的 JSON）。";
    case "empty":
      return "引擎回傳了空的內容。";
    case "field":
      return "引擎回傳的資料格式不符合預期。";
  }
}

/** Raw text for the expandable 詳細資訊, or undefined when there is nothing extra. */
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
      title: "正在連接引擎",
      short: "連線中",
      hint: "正在連線到本機引擎。",
    };
  if (phase === "unauthorized" || status === 401 || status === 403)
    return {
      title: "需要有效的存取權杖",
      short: "未授權",
      hint: "引擎拒絕了這組存取金鑰，請到設定更新。",
    };
  if (typeof status === "number" && status >= 500)
    return {
      title: "引擎內部錯誤",
      short: "內部錯誤",
      hint: `引擎回應 HTTP ${status}，請查看引擎日誌。`,
    };
  if (typeof status === "number")
    return {
      title: `引擎回傳 HTTP ${status}`,
      short: `HTTP ${status}`,
      hint: `引擎對狀態查詢回應了 HTTP ${status}。`,
    };
  return {
    title: "無法連接引擎",
    short: "無法連線",
    hint: "沒有收到引擎的回應，請確認 Yunshu 服務正在執行。",
  };
}
