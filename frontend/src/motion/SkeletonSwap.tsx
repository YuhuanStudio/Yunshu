import { useRef, useState, type ReactNode } from "react";

/**
 * Skeleton -> content without a pop and without a layout shift. While `ready` is false the
 * skeleton shows. When it turns true, the content takes the same grid cell and arrives with the
 * soft entrance while the skeleton fades out under it; the cell is as tall as the taller of the two,
 * so nothing around it moves. Content that is ready on first paint (a revisit) renders alone, with
 * no entrance, and later polls never replay anything: the entrance is a one-shot CSS animation on
 * an element that stays mounted.
 */
export function SkeletonSwap({
  ready,
  skeleton,
  children,
  className,
}: {
  ready: boolean;
  skeleton: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  const hadSkeleton = useRef(!ready);
  const [skeletonGone, setSkeletonGone] = useState(ready);
  if (!ready) {
    hadSkeleton.current = true;
    if (skeletonGone) setSkeletonGone(false);
  }
  if (!hadSkeleton.current) return <>{children}</>;
  return (
    <div
      className={`grid ${className ?? ""}`}
      data-swap={ready ? "ready" : "loading"}
    >
      {!skeletonGone && (
        <div
          aria-hidden="true"
          style={{ gridArea: "1 / 1" }}
          className={ready ? "motion-exit pointer-events-none" : undefined}
          onAnimationEnd={() => ready && setSkeletonGone(true)}
        >
          {skeleton}
        </div>
      )}
      {ready && (
        <div style={{ gridArea: "1 / 1" }} className="motion-enter min-w-0">
          {children}
        </div>
      )}
    </div>
  );
}
