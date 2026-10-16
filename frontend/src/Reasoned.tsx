import { useRef, useState, type ReactNode } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@yuhuanowo/yunui";

/**
 * A disabled control that can still be hovered or focused to read why. A disabled
 * button receives no pointer events, so the tooltip hangs on a wrapper.
 */
export function Reasoned({
  reason,
  children,
}: {
  reason?: string | null;
  children: ReactNode;
}) {
  const trigger = useRef<HTMLSpanElement>(null);
  const dismissed = useRef(false);
  const [open, setOpen] = useState(false);
  // Same wrapper box with or without a reason, so a reason appearing or going never changes the layout.
  if (!reason) return <span className="inline-flex">{children}</span>;
  // The tooltip closes on any scroll around its trigger, and focusing an off-screen trigger scrolls it
  // into view: a keyboard user saw no reason at all. While the trigger keeps focus only an explicit
  // dismissal (Escape, a press) closes it.
  const onOpenChange = (next: boolean) => {
    if (
      !next &&
      !dismissed.current &&
      trigger.current != null &&
      document.activeElement === trigger.current
    )
      return;
    dismissed.current = false;
    setOpen(next);
  };
  return (
    <TooltipProvider>
      <Tooltip open={open} onOpenChange={onOpenChange}>
        <TooltipTrigger asChild>
          <span
            ref={trigger}
            tabIndex={0}
            className="inline-flex"
            data-testid="reasoned"
            onPointerDown={() => (dismissed.current = true)}
          >
            {children}
          </span>
        </TooltipTrigger>
        <TooltipContent onEscapeKeyDown={() => (dismissed.current = true)}>
          {reason}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
