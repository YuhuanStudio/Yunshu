type KeyLike = {
  key: string;
  defaultPrevented: boolean;
  target: unknown;
};

/** True when the key is typed into a text field, where Escape clears or blurs the field first. */
function typing(target: unknown) {
  const el = target as { tagName?: string; isContentEditable?: boolean } | null;
  const tag = el?.tagName?.toUpperCase();
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    el?.isContentEditable === true
  );
}

/**
 * Whether a document-level Escape belongs to the page (closing its inspector) rather than to an open
 * dialog, a handler that already used the key, or a text field being typed in.
 */
export function pageOwnsEscape(e: KeyLike, dialogOpen: boolean): boolean {
  return (
    e.key === "Escape" &&
    !e.defaultPrevented &&
    !dialogOpen &&
    !typing(e.target)
  );
}
