/**
 * Whether the reasoning block is open. While reasoning streams it opens by
 * itself and folds once the answer text starts; a click by the user wins.
 */
export function thinkingOpen(
  streaming: boolean,
  hasAnswer: boolean,
  userChoice: boolean | null,
): boolean {
  if (userChoice !== null) return userChoice;
  return streaming && !hasAnswer;
}

/** Seconds the undo strip stays after clearing a conversation. */
export const UNDO_WINDOW_MS = 8000;
