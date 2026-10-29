/** The Yunshu tile (public/favicon.svg, drawn from the same source as the favicon and the touch icons). */
export const BRAND_MARK_SRC = `${import.meta.env.BASE_URL}favicon.svg`;

/** The mark as an image: sidebar, docs title, Ask AI and the assistant's avatar all show this one tile. */
export function BrandMark({
  size = 20,
  className = "",
}: {
  size?: number;
  className?: string;
}) {
  return (
    <img
      src={BRAND_MARK_SRC}
      alt=""
      width={size}
      height={size}
      draggable={false}
      className={`shrink-0 select-none ${className}`}
    />
  );
}
