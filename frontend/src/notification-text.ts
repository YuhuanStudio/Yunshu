import { clockShort } from "./i18n/format.ts";
import { tr } from "./i18n/index.ts";
import type { NotifEvent } from "./notifications.ts";

/** Title and body of one event, built when shown so a language switch applies to old rows too. */
export function describeNotification(e: NotifEvent): {
  title: string;
  body: string;
} {
  const vars = { ...e.vars, time: clockShort(e.at) };
  return {
    // i18n-keys: shell.notif.kind.
    title: tr(`shell.notif.kind.${e.kind}.title`, vars),
    // i18n-keys: shell.notif.kind.
    body: tr(`shell.notif.kind.${e.kind}.body`, vars),
  };
}
