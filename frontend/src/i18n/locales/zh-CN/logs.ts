import type zh from "../zh-TW/logs.ts";
import type { Shape } from "../../types.ts";

const logs: Shape<typeof zh> = {
  "page.title": "日志",
  "page.description": "引擎最近的运行日志；可实时跟踪、搜索和下载。",
  "action.copyVisible": "复制可见行",
  "action.download": "下载可见行",
  "action.pause": "暂停",
  "action.resume": "继续",
  "action.copyLine": "复制这一行",
  "action.jump": "跳到最新",
  "action.jumpNew": "跳到最新（{count} 条新记录）",
  copiedAll: "已复制 {count} 行",
  copiedLine: "已复制这一行",
  copyFailed: "无法写入剪贴板",
  "search.aria": "搜索日志",
  "search.placeholder": "搜索消息或模块",
  "level.aria": "日志级别",
  "level.all": "全部",
  "level.DEBUG": "调试",
  "level.INFO": "信息",
  "level.WARNING": "警告",
  "level.ERROR": "错误",
  "level.CRITICAL": "严重",
  "span.all": "全部（缓冲内）",
  "span.5m": "最近 5 分钟",
  "span.1h": "最近 1 小时",
  "span.window": "{from}–{to}",
  "live.connecting": "连接中…",
  "live.live": "实时跟踪中",
  "live.polling": "实时跟踪（轮询）",
  "live.reconnecting": "正在重新连接…",
  "live.paused": "已暂停",
  "live.window": "固定时间范围",
  "pause.windowWhy":
    "正在查看固定时间范围，不会跟踪新记录；改选时间范围即可恢复。",
  "list.aria": "日志记录",
  "state.loading": "正在读取日志…",
  "state.missingTitle": "这个引擎还没有日志接口",
  "state.missingBody":
    "需要较新的引擎版本；在此之前请从终端或服务的日志文件查看。",
  "state.deniedTitle": "需要管理权限",
  "state.deniedBody":
    "日志只对具有管理权限的密钥开放；请到设置确认连接所用的密钥。",
  "state.errorTitle": "暂时无法读取日志",
  "state.errorBody": "引擎没有响应；连接恢复后会自动重试，或调整筛选再试一次。",
  "state.emptyTitle": "没有日志",
  "state.emptyBody": "引擎启动后产生的日志会显示在这里。",
  "state.emptyFiltered":
    "没有符合当前筛选的记录；放宽级别、搜索词或时间范围再看看。",
  "footer.counts": "显示 {count} 行（服务器最多保留 {capacity} 行）",
  "footer.dropped": "已有 {count} 行较旧记录被挤出缓冲",
  "footer.span": "最早 {from}",
  "note.redaction":
    "日志在服务器端写入时就已屏蔽凭据和请求内容，这里看到的就是屏蔽后的结果；缓冲只存在内存中（最多 {capacity} 行），引擎重启后清空。",
};
export default logs;
