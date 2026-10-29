import type { ReactNode } from "react";
import { Button, cn } from "@yuhuanowo/yunui";
import { BrandMark } from "../BrandMark";
import {
  ChatComposer,
  ChatMessage,
  ChatMessageList,
} from "@yuhuanowo/yunui/chat";
import type {
  ChatComposerProps,
  ChatMessageProps,
} from "@yuhuanowo/yunui/chat";

/** One conversation layout for the Playground and Ask AI: header, a scrolling transcript in a single
 *  centred column, and the composer pinned under it in the same column. The transcript sticks to the
 *  bottom until the reader scrolls up (YunUI ChatMessageList), and the dock keeps the phone's safe area. */
export function ChatSurface({
  header,
  empty,
  jumpToLatestLabel,
  children,
  dock,
  wide = false,
  liftEmpty = true,
  className,
  "data-testid": testId,
}: {
  header?: ReactNode;
  /** Shown centred while there is nothing in the transcript. */
  empty?: ReactNode;
  jumpToLatestLabel: string;
  /** The turns; no children shows `empty`. */
  children?: ReactNode;
  /** Notices, composer and footnote, pinned under the transcript. */
  dock: ReactNode;
  /** A wider column for side-by-side replies. */
  wide?: boolean;
  /** Sit the empty state a little above the middle (a full page); off inside a dialog. */
  liftEmpty?: boolean;
  className?: string;
  "data-testid"?: string;
}) {
  const hasTurns = Boolean(children);
  return (
    <div
      className={cn("flex min-h-0 flex-1 flex-col", className)}
      data-testid={testId}
      data-empty={hasTurns ? undefined : ""}
    >
      {header}
      <div
        className={cn(
          "flex min-h-0 flex-1 flex-col",
          !hasTurns && "justify-center",
          !hasTurns && liftEmpty && "pb-[6vh]",
        )}
      >
        <div
          className={cn(
            "flex min-h-0 flex-col",
            hasTurns ? "flex-1" : "flex-none",
          )}
        >
          <ChatMessageList
            jumpToLatestLabel={jumpToLatestLabel}
            className="px-3 sm:px-6"
            empty={<div className="w-full max-w-xl px-1 py-6">{empty}</div>}
          >
            {children ? (
              <div
                className={cn(
                  "mx-auto w-full pb-6 pt-2",
                  wide ? "max-w-5xl" : "max-w-[44rem]",
                )}
                data-chat-column=""
              >
                {children}
              </div>
            ) : null}
          </ChatMessageList>
        </div>
        <div
          className="shrink-0 px-3 pb-[max(0.5rem,env(safe-area-inset-bottom))] pt-2 sm:px-6"
          data-chat-dock=""
        >
          <div className="mx-auto w-full max-w-[44rem]">{dock}</div>
        </div>
      </div>
    </div>
  );
}

/** The composer as one line: attach buttons on the left, the text in the middle, send on the right. The
 *  box grows with the text up to `maxRows`; YunUI's own layout (a 68 px textarea over a button row) is
 *  flattened with CSS because the component has no compact density. */
export function CompactComposer({ className, ...props }: ChatComposerProps) {
  return (
    <ChatComposer
      maxRows={6}
      className={cn(
        "flex-row flex-wrap items-end gap-x-1.5 rounded-[20px] p-1.5 sm:p-1.5",
        "[&_textarea]:min-h-0 [&_textarea]:py-[7px]",
        "[&>div:first-child:not(:has(textarea))]:basis-full",
        "[&>div:has(textarea)]:order-2 [&>div:has(textarea)]:min-w-0 [&>div:has(textarea)]:flex-1 [&>div:has(textarea)]:pt-0",
        "[&>div:last-child:not(:has(textarea))]:contents",
        "[&>div:last-child>div:first-child]:order-1 [&>div:last-child>div:first-child]:pb-px",
        "[&>div:last-child>div:last-child]:order-3 [&>div:last-child>div:last-child]:pb-px",
        className,
      )}
      {...props}
    />
  );
}

/** A turn in the transcript: the user's message is a right-aligned bubble, the assistant's is plain text
 *  across the column. Its actions show on hover or focus (always on touch). The avatar gutter is dropped on phones, where 38 px of the width is worth more as text. */
export function ChatTurn({ className, actions, ...props }: ChatMessageProps) {
  return (
    <ChatMessage
      actions={
        actions ? (
          <div
            className="flex items-center gap-1 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100"
            data-chat-actions=""
          >
            {actions}
          </div>
        ) : undefined
      }
      density="compact"
      avatar={props.role === "assistant" ? <BrandMark size={20} /> : undefined}
      className={cn(
        "max-w-none max-sm:[&>div:first-child:not(:only-child)]:hidden max-sm:gap-0",
        props.role === "user" && "justify-end",
        className,
      )}
      {...props}
    />
  );
}

/** What an empty conversation says: one line, one hint, and starter prompts that fill the composer. */
export function ChatEmpty({
  icon,
  title,
  description,
  suggestions,
  onPick,
  disabled,
  groupLabel,
  "data-testid": testId,
}: {
  icon?: ReactNode;
  title: string;
  description?: string;
  suggestions: string[];
  onPick: (text: string) => void;
  disabled?: boolean;
  groupLabel: string;
  "data-testid"?: string;
}) {
  return (
    <div className="flex flex-col items-center gap-3 text-center">
      {icon && <div className="text-muted-foreground">{icon}</div>}
      <div className="space-y-1">
        <p className="text-base font-semibold">{title}</p>
        {description && (
          <p className="text-[13px] text-muted-foreground">{description}</p>
        )}
      </div>
      <div
        className="flex w-full flex-col items-stretch gap-1.5 sm:items-center"
        role="group"
        aria-label={groupLabel}
        data-testid={testId}
      >
        {suggestions.map((text) => (
          <Button
            key={text}
            size="sm"
            variant="secondary"
            className="h-auto max-w-full justify-start whitespace-normal py-1.5 text-left sm:justify-center sm:text-center"
            disabled={disabled}
            onClick={() => onPick(text)}
          >
            {text}
          </Button>
        ))}
      </div>
    </div>
  );
}
