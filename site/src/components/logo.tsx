import { asset } from "@/lib/site";

/** YuhuanStudio mark + product wordmark. */
export function Logo({ size = 22, suffix }: { size?: number; suffix?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-fd-foreground">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={asset("/yuhuanstudio-logo.png")}
        alt=""
        width={size}
        height={size}
        className="shrink-0"
        style={{ width: size, height: size }}
      />
      <span className="text-[0.975rem] leading-none font-semibold tracking-tight">Yunshu</span>
      {suffix ? <span className="text-fd-muted-foreground text-[0.975rem] leading-none font-normal">{suffix}</span> : null}
    </span>
  );
}
