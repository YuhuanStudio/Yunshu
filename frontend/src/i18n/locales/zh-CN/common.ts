import type zh from "../zh-TW/common.ts";
import type { Shape } from "../../types.ts";

const common: Shape<typeof zh> = {
  retry: "重试",
  configure: "连接设置",
  autoRetry: "每 3 秒自动重试。",
  lastOk: "最后一次成功连接于 {time}，下方保留上次的数据。",
  details: "详细信息",
  copy: "复制",
  copied: "已复制",
  copyFailed: "无法写入剪贴板",
  copyLabel: "复制{label}",
  "language.label": "语言",
  sentenceGap: "",
  offlineFor: "已离线 {t}。",
  staleAt: "数据停在 {time}",
  "unlock.placeholder": "粘贴访问令牌",
  "unlock.label": "解锁密钥",
  "unlock.remember": "在此设备记住",
  "unlock.submit": "解锁",
};
export default common;
