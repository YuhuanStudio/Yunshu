import type { ReactNode } from "react";
import { Button, cn } from "@yuhuanowo/yunui";
import { ChatMessage, ChatMessageList } from "@yuhuanowo/yunui/chat";
import type { ChatMessageProps } from "@yuhuanowo/yunui/chat";

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
  className?: string;
  "data-testid"?: string;
}) {
  return (
    <div
      className={cn("flex min-h-0 flex-1 flex-col", className)}
      data-testid={testId}
    >
      {header}
      <ChatMessageList
        jumpToLatestLabel={jumpToLatestLabel}
        className="px-3 sm:px-6"
        empty={<div className="w-full max-w-xl px-1 py-8">{empty}</div>}
      >
        {children ? (
          <div
            className={cn(
              "mx-auto w-full pb-6 pt-2",
              wide ? "max-w-5xl" : "max-w-3xl",
            )}
            data-chat-column=""
          >
            {children}
          </div>
        ) : null}
      </ChatMessageList>
      <div
        className="shrink-0 px-3 pb-[max(0.5rem,env(safe-area-inset-bottom))] pt-2 sm:px-6"
        data-chat-dock=""
      >
        <div className="mx-auto w-full max-w-3xl">{dock}</div>
      </div>
    </div>
  );
}

/** A turn in the transcript: the user's message is a right-aligned bubble, the assistant's is plain text
 *  across the column. The avatar gutter is dropped on phones, where 38 px of the width is worth more as text. */
export function ChatTurn({ className, ...props }: ChatMessageProps) {
  return (
    <ChatMessage
      density="compact"
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
