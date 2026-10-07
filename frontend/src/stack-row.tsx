import type { ReactNode } from "react";

/**
 * A settings row that stacks on narrow screens (label and description above, control full width
 * below) and sits side by side from `sm` up. YunUI's SettingRow keeps two columns at every width,
 * which squeezes the text to a character per line on a phone.
 */
export function StackRow({
  title,
  description,
  control,
}: {
  title: ReactNode;
  description?: ReactNode;
  control?: ReactNode;
}) {
  return (
    <div
      className="flex flex-col gap-2 border-b border-border py-3 last:border-0 sm:flex-row sm:items-center sm:justify-between sm:gap-4"
      data-testid="stack-row"
    >
      <div className="min-w-0 sm:flex-1">
        <div className="text-sm font-medium">{title}</div>
        {description && (
          <div className="mt-0.5 whitespace-normal wrap-break-word text-xs leading-relaxed text-(--text-tertiary)">
            {description}
          </div>
        )}
      </div>
      {control != null && (
        <div className="min-w-0 sm:max-w-[60%] sm:shrink-0 [&>*]:max-w-full">
          {control}
        </div>
      )}
    </div>
  );
}
