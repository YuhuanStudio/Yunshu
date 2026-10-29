import { HugWidth } from "./HugWidth";

const NUMBER = /(\d[\d.,]*\d|\d)/;

/**
 * Text with numbers in it ("11.2 / 59.6 GB", "上次 254 tok/s · 5s") for a place where the box shows
 * (a status pill). Each number holds its own width (HugWidth: grows at once, gives the room back
 * after 8 s of being narrower), so a count that crosses a digit boundary moves the text by one
 * digit instead of resizing the whole pill, and no room is held beyond one digit. Words around the
 * numbers take their natural width. When the text takes another shape (another phase, another
 * sentence) the numbers start over, so a new state hugs at once.
 */
export function HeldText({ text }: { text: string }) {
  const parts = text.split(NUMBER);
  if (parts.length === 1) return <>{text}</>;
  const shape = text.replace(new RegExp(NUMBER, "g"), "#");
  let n = 0;
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 0 ? (
          part
        ) : (
          <HugWidth
            key={`${shape}|${n++}`}
            align="right"
            holdMs={8000}
            className="tabular-nums"
          >
            {part}
          </HugWidth>
        ),
      )}
    </>
  );
}
