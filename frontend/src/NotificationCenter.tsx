import { useRef, useState } from "react";
import {
  Button,
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "@yuhuanowo/yunui";
import {
  NotificationBell,
  NotificationItem,
  NotificationPanel,
} from "@yuhuanowo/yunui/patterns";
import {
  AlertTriangle,
  CheckCircle2,
  CircleAlert,
  Info,
  type LucideIcon,
} from "lucide-react";
import { FRESH_ROW, useFreshIds } from "./fresh-rows";
import { NOTIF_HREF, type NotifTone } from "./notifications";
import { describeNotification } from "./notification-text";
import { relative } from "./i18n/format.ts";
import { t, useLocale } from "./i18n/index.ts";
import { useSignals } from "./signals";

const TONES: Record<NotifTone, { icon: LucideIcon; tint: string }> = {
  success: { icon: CheckCircle2, tint: "bg-muted text-success" },
  info: { icon: Info, tint: "bg-muted text-info" },
  warn: { icon: AlertTriangle, tint: "bg-muted text-warning" },
  error: { icon: CircleAlert, tint: "bg-muted text-error" },
};

/** The bell in the top bar: the session's events, newest first, read when the panel closes. */
export function NotificationCenter() {
  useLocale();
  const { notifications, unread, markRead, clearAll } = useSignals();
  const fresh = useFreshIds(notifications.map((n) => n.id));
  const [open, setOpen] = useState(false);
  const anchor = useRef<HTMLSpanElement>(null);
  const change = (next: boolean) => {
    setOpen(next);
    if (!next) markRead();
  };
  return (
    <Popover open={open} onOpenChange={change}>
      <PopoverAnchor asChild>
        <span ref={anchor} className="relative inline-flex">
          {/* The badge is always in the tree (hidden at zero), so one arriving changes no box. */}
          <span
            aria-hidden="true"
            data-testid="bell-badge"
            className={`pointer-events-none absolute -right-0.5 -top-0.5 z-10 flex h-4 min-w-4 items-center justify-center rounded-full bg-(--error) px-1 text-[10px] font-bold text-pure-white ${unread ? "" : "invisible"}`}
          >
            {unread > 9 ? "9+" : unread || 0}
          </span>
          <NotificationBell
            className="card size-8 rounded-full p-0"
            count={0}
            label={
              unread
                ? t("shell.notif.bellUnread", { n: unread })
                : t("shell.notif.bell")
            }
            onClick={() => change(!open)}
          />
        </span>
      </PopoverAnchor>
      <PopoverContent
        align="end"
        sideOffset={8}
        className="w-auto min-w-0 border-0 bg-transparent p-0 shadow-none backdrop-blur-none"
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          anchor.current?.querySelector("button")?.focus();
        }}
      >
        <NotificationPanel
          title={t("shell.notif.title")}
          unreadCount={unread}
          empty={!notifications.length}
          labels={{
            unread: t("shell.notif.unread"),
            empty: t("shell.notif.empty"),
          }}
          footer={
            notifications.length ? (
              <Button
                size="sm"
                variant="ghost"
                className="my-1"
                onClick={clearAll}
              >
                {t("shell.notif.clear")}
              </Button>
            ) : undefined
          }
        >
          {notifications.map((n) => {
            const { icon: Icon, tint } = TONES[n.tone];
            const { title, body } = describeNotification(n);
            return (
              <NotificationItem
                key={n.id}
                className={fresh(n.id) ? FRESH_ROW : undefined}
                icon={<Icon size={14} />}
                iconClassName={tint}
                title={title}
                body={body}
                time={relative((Date.now() - n.at) / 1000)}
                unread={!n.read}
                href={NOTIF_HREF[n.kind]}
                onSelect={() => change(false)}
              />
            );
          })}
        </NotificationPanel>
      </PopoverContent>
    </Popover>
  );
}
