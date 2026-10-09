import { splitBytes } from "./byte-format.ts";

/** A byte count with its unit small and muted right after the number (no gap). */
export function ByteValue({
  bytes,
  className = "",
}: {
  bytes: number | null | undefined;
  className?: string;
}) {
  const { value, unit } = splitBytes(bytes);
  return (
    <span className={`tabular-nums ${className}`}>
      {value}
      {unit && (
        <span className="text-xs font-normal text-muted-foreground">
          {unit}
        </span>
      )}
    </span>
  );
}
