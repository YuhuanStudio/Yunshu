import { number } from "./ui";

export const SIZE_UNKNOWN_REASON =
  "這個引擎版本沒有回報模型大小（size_gb）；升級後會顯示。這裡不猜測數字。";

/** Model size in GB, or an honest dash that explains itself. */
export function ModelSize({ gb }: { gb: number | null | undefined }) {
  if (gb != null && Number.isFinite(gb) && gb > 0)
    return <>{`${number(gb)} GB`}</>;
  return (
    <span
      title={SIZE_UNKNOWN_REASON}
      aria-label={`大小未知。${SIZE_UNKNOWN_REASON}`}
      tabIndex={0}
      className="cursor-help"
      data-testid="size-unknown"
    >
      —
    </span>
  );
}
