import { useEffect, useRef, useState, type ReactNode } from "react";
import { reducedMotion } from "./ticker.ts";

/**
 * Keeps `children` mounted for the exit animation: it fades in on mount (data-state="open", the soft
 * entrance from the motion tokens) and, when `show` turns false, fades out (120 ms unless `ms`) before it unmounts (data-state="closing"). The slot
 * keeps its height while closing, so what is below does not jump before the fade is done. Under
 * reduced motion it mounts and unmounts at once.
 */
export function Presence({
  show,
  ms,
  className,
  children,
}: {
  show: boolean;
  ms?: number;
  className?: string;
  children: ReactNode;
}) {
  const [mounted, setMounted] = useState(show);
  // while closing, the last thing shown stays (the data behind it may already be gone)
  const last = useRef(children);
  if (show) last.current = children;
  useEffect(() => {
    if (show) {
      setMounted(true);
      return;
    }
    if (reducedMotion()) {
      setMounted(false);
      return;
    }
    const id = setTimeout(() => setMounted(false), ms ?? 120);
    return () => clearTimeout(id);
  }, [show, ms]);
  if (!show && !mounted) return null;
  return (
    <div
      className={`live-presence ${className ?? ""}`}
      data-state={show ? "open" : "closing"}
      style={ms ? { ["--live-ms" as string]: `${ms}ms` } : undefined}
    >
      {last.current}
    </div>
  );
}
