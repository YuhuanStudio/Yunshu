import { Tabs, TabsList, TabsTrigger, cn } from "@yuhuanowo/yunui";
import type { LucideIcon } from "lucide-react";

export type TrayOption<T extends string> = {
  value: T;
  label: string;
  icon?: LucideIcon;
};

/**
 * One container with an inner highlight on the selected segment (tray style), instead of a row of
 * separately bordered buttons. Composed from YunUI `Tabs` parts; no tab panels are rendered.
 */
export function SegmentedTray<T extends string = string>({
  options,
  value,
  onChange,
  className,
  "aria-label": ariaLabel,
  fillOnPhone,
}: {
  options: TrayOption<T>[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
  "aria-label"?: string;
  /** Accepted for drop-in compatibility; the tray always wraps if it must. */
  wrap?: boolean;
  /** Phone: the tray spans its row and the segments share it equally, so stacked trays align. */
  fillOnPhone?: boolean;
}) {
  return (
    <Tabs
      value={value}
      onValueChange={(v) => onChange(v as T)}
      className={cn(fillOnPhone && "max-sm:w-full", className)}
    >
      <TabsList
        aria-label={ariaLabel}
        className={cn(
          "h-8 flex-wrap gap-0.5 rounded-lg bg-(--tray-track) p-0.5",
          fillOnPhone && "max-sm:w-full",
        )}
      >
        {options.map(({ value: v, label, icon: Icon }) => (
          <TabsTrigger
            key={v}
            value={v}
            className={cn(
              "h-7 gap-1.5 rounded-md px-2.5 py-0 text-xs",
              fillOnPhone && "max-sm:flex-1",
              "data-[state=active]:bg-(--tray-selected) data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:ring-1 data-[state=active]:ring-(--tray-selected-ring)",
              "data-[state=inactive]:text-muted-foreground",
            )}
          >
            {Icon && <Icon size={13} />}
            {label}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
