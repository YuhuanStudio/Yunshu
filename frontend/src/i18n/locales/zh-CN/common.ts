import type zh from "../zh-TW/common.ts";
import type { Shape } from "../../types.ts";

const common: Shape<typeof zh> = {
  retry: "重试",
  configure: "连接设置",
  lastOk: "最后一次成功连接于 {time}，下方保留上次的数据。",
  details: "详细信息",
  copy: "复制",
  copied: "已复制",
  copyFailed: "无法写入剪贴板",
  copyLabel: "复制{label}",
  "language.label": "语言",
  sentenceGap: "",
  offlineSince: "引擎自 {time} 离线，已 {t}。",
  retryIn: "{s} 秒后重试。",
  retryNow: "正在重试…",
  engineMessage: "引擎返回：{message}",
  retrying: "暂时无法更新，数字为上一次读取，将自动重试",
  staleAt: "数据停在 {time}",
  "unlock.placeholder": "粘贴访问令牌",
  "unlock.label": "解锁密钥",
  "unlock.remember": "在此设备记住",
  "unlock.submit": "解锁",
};
export default common;
