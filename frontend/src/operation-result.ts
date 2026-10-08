export function operationResult(
  key: string,
  result: unknown,
): { error: boolean; text: string } {
  const body =
    result && typeof result === "object"
      ? (result as Record<string, unknown>)
      : null;
  if (typeof body?.warning === "string" && body.warning)
    return {
      error: true,
      text: `操作部分完成：${body.warning}。請檢查最新引擎狀態。`,
    };
  if (key.startsWith("warmup:") && body?.generated === false)
    return { error: false, text: "模型已載入；此類型未執行文字預熱。" };
  return {
    error: false,
    text: key.startsWith("cancel:") ? "已送出取消請求。" : "操作已完成。",
  };
}
