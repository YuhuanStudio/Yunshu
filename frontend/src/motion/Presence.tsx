import { useEffect, useRef, useState, type ReactNode } from "react";
import { mayAnimate } from "./burst-core.ts";
import { reducedMotion } from "./ticker.ts";

/**
 * Keeps `children` mounted for the exit animation: it fades in on mount (data-state="open", the soft
 * entrance from the motion tokens) and, when `show` turns false, fades out (120 ms unless `ms`) before it unmounts (data-state="closing"). The slot
 * keeps its height while closing, so what is below does not jump before the fade is done. Under
 * reduced motion it mounts and unmounts at once. With `adaptive` (data decides when it comes and
 * goes, not the user) it animates only while the screen is calm (BurstGovernor); a block that flips
 * with every update is drawn and dropped in place.
 */
export function Presence({
  show,
  ms,
  adaptive = false,
  className,
  children,
}: {
  show: boolean;
  ms?: number;
  adaptive?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const [mounted, setMounted] = useState(show);
  // one decision per change of `show`, taken while rendering so the first frame already knows
  const [prev, setPrev] = useState(show);
  const [still, setStill] = useState(false);
  if (prev !== show) {
    setPrev(show);
    setStill(adaptive && !mayAnimate());
  }
  // while closing, the last thing shown stays (the data behind it may already be gone)
  const last = useRef(children);
  if (show) last.current = children;
  useEffect(() => {
    if (show) {
      setMounted(true);
      return;
    }
    if (reducedMotion() || still) {
      setMounted(false);
      return;
    }
    const id = setTimeout(() => setMounted(false), ms ?? 120);
    return () => clearTimeout(id);
  }, [show, ms, still]);
  if (!show && (!mounted || still)) return null;
  return (
    <div
      className={`live-presence ${className ?? ""}`}
      data-state={show ? "open" : "closing"}
      data-still={still ? "" : undefined}
      style={ms ? { ["--live-ms" as string]: `${ms}ms` } : undefined}
    >
      {last.current}
    </div>
  );
}
