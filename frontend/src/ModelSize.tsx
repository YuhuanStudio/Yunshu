import { t } from "./i18n/index.ts";
import { number } from "./ui";

export const sizeUnknownReason = () => t("models.size.unknownReason");

/** Model size in GB, or an honest dash that explains itself. */
export function ModelSize({ gb }: { gb: number | null | undefined }) {
  if (gb != null && Number.isFinite(gb) && gb > 0)
    return <>{`${number(gb)} GB`}</>;
  return (
    <span
      title={sizeUnknownReason()}
      aria-label={t("models.size.unknownAria", { reason: sizeUnknownReason() })}
      tabIndex={0}
      className="cursor-help"
      data-testid="size-unknown"
    >
      —
    </span>
  );
}
