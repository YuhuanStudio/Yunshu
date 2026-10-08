import { number } from "./i18n/format.ts";
import { useTween } from "./tween";

/** A live reading that glides between samples; tabular digits, so the width does not change as it moves. */
export function LiveNumber({
  value,
  digits = 0,
  ms = 220,
}: {
  value: number | null;
  digits?: number;
  ms?: number;
}) {
  const shown = useTween(value, ms);
  return (
    <span className="tabular-nums">
      {shown == null ? "—" : number(shown, digits)}
    </span>
  );
}
