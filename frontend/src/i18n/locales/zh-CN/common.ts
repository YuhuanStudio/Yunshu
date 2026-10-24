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
  "outage.restarting": "引擎重新启动中…",
  "outage.switching": "切换到 {engine}",
  "outage.following": "改追最新运行中的引擎，历史会保留。",
  "outage.wasUp": "此前已运行 {up}。",
  "outage.backIn": "通常约 {s} 后恢复。",
  "outage.longer": "比平常的 {usual} 久。",
  "outage.waiting": "已等待 {s}。",
  "outage.downTitle": "已 {s} 没有任何引擎响应",
  "outage.downEngines":
    "控制台已知的引擎：{engines}。请启动引擎，或检查连接设置。",
  "outage.downNone":
    "找不到任何引擎。请启动 Yunshu 引擎，控制台会自动重新连上。",
  "outage.unknownState": "状态未知",
};
export default common;
