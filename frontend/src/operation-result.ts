import { t } from "./i18n/index.ts";

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
      text: t("errors.operation.partial", { warning: body.warning }),
    };
  if (key.startsWith("warmup:") && body?.generated === false)
    return { error: false, text: t("errors.operation.warmupSkipped") };
  return {
    error: false,
    text: key.startsWith("cancel:")
      ? t("errors.operation.cancelSent")
      : t("errors.operation.done"),
  };
}
