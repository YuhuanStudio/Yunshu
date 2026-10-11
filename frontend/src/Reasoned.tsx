import type { ReactNode } from "react";
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
  // Same wrapper box with or without a reason, so a reason appearing or going never changes the layout.
  if (!reason) return <span className="inline-flex">{children}</span>;
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span tabIndex={0} className="inline-flex" data-testid="reasoned">
            {children}
          </span>
        </TooltipTrigger>
        <TooltipContent>{reason}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
