import type zh from "../zh-TW/service.ts";
import type { Shape } from "../../types.ts";

const service: Shape<typeof zh> = {
  title: "服务",
  description: "引擎作为后台服务（launchd）的状态，以及重新启动。",
  refresh: "刷新",
  loading: "读取服务状态…",
  "unavailable.unsupported.title": "此引擎版本没有服务管理",
  "unavailable.unsupported.description":
    "升级 Yunshu 后即可在此查看和重新启动。",
  "unavailable.denied.title": "需要管理员令牌",
  "unavailable.denied.description":
    "请在“引擎连接”输入服务的 YUNSHU_AUTH_TOKEN 或具有管理权限的密钥。",
  "unavailable.error.title": "无法读取",
  "unavailable.error.description": "请确认服务仍在运行，稍后再试。",
  "state.managed": "由 launchd 管理",
  "state.other": "launchd 已加载，但当前这个进程不是它启动的",
  "state.stopped": "已安装，未运行",
  "state.notInstalled": "尚未安装为服务",
  "row.status": "状态",
  "row.pid": "进程 ID",
  "row.uptime": "已运行",
  "row.uptimeHelp": "自此进程启动起算。",
  "row.version": "版本",
  "row.plist": "plist 文件",
  "row.log": "日志文件",
  "restart.button": "重新启动",
  "restart.confirmTitle": "重新启动引擎",
  "restart.confirmBody":
    "会先等待进行中的请求完成，最久 {seconds} 秒，之后强制重启。重启期间服务暂时无法连接，已加载的模型需要重新加载。",
  "restart.confirmBodyUnknown":
    "会先等待进行中的请求完成，之后强制重启。重启期间服务暂时无法连接，已加载的模型需要重新加载。",
  "restart.confirm": "重新启动",
  "restart.accepted":
    "已安排重新启动：正在等待 {count} 个进行中的请求（最久 {seconds} 秒）。完成后控制台会自动重新连接。",
  "restart.failed": "重新启动没有成功发出，请稍后再试。",
  "restart.notLaunchd":
    "这个引擎不是由 launchd 服务启动的，无法自行重启。请手动重新启动：",
  "restart.manual": "手动重新启动的命令",
  "restart.cli": "命令",
  "restart.help": "一键重新启动只在由 launchd 服务管理时可用。",
  "network.title": "网络",
  "network.description":
    "服务监听的地址（只读）。要改变监听地址，请重新安装服务。",
  "network.address": "控制台连接的地址",
  "network.addressHelp": "来自当前的服务地址设置。",
  "network.exposure": "可访问范围",
  "network.exposureHelp":
    "仅本机表示只有这台电脑能连；局域网表示同网络的其他设备也能连。",
  "network.loopback": "仅本机",
  "network.lan": "局域网可访问",
  "network.cmdLocal": "改为仅本机",
  "network.cmdLan": "改为局域网可访问",
  "network.lanWarning":
    "开放局域网前，请先设置 YUNSHU_AUTH_TOKEN 或创建 API 密钥，并在 CORS 只允许需要的来源。",
  "cors.title": "CORS 来源",
  "cors.description": "允许哪些网站从浏览器调用这个服务。保存后立即生效。",
  "cors.invalid":
    "请输入以 http:// 或 https:// 开头的来源，不含路径，例如 https://app.example.com；或单独的 *。",
  "cors.mixed": "* 不能和其他来源同时使用。",
  "cors.duplicate": "这个来源已在列表中。",
  "cors.rejected": "服务器拒绝了这些来源：{list}",
  "cors.forced":
    "当前由{source}设置，优先于配置文件；这里的修改会保存，但不会改变运行中的值。",
  "cors.source.env": "环境变量 YUNSHU_CORS_ORIGINS",
  "cors.source.cli": "命令行参数",
  "cors.wildcardTitle": "任何网站都能调用这个服务",
  "cors.wildcardBody":
    "* 让任何网站的浏览器页面都能向此服务发请求，且不会携带凭据。除非服务只在本机可连，否则请改用明确的来源。",
  "cors.thisAllowed": "当前的控制台来源 {origin} 已被允许。",
  "cors.thisBlocked":
    "当前的控制台来源 {origin} 不在列表中，其他浏览器页面会被拦下。",
  "cors.listAria": "允许的来源",
  "cors.remove": "移除 {origin}",
  "cors.none": "列表是空的",
  "cors.add": "添加来源",
  "cors.addButton": "添加",
  "cors.confirmAny": "我了解 * 会让任何网站调用此服务",
  "cors.confirmAnyNeeded": "使用 * 前需要先确认",
  "cors.save": "保存",
  "cors.reset": "恢复默认",
  "cors.alreadyDefault": "当前已是默认值",
  "cors.saved": "已保存",
  "cors.help": "默认：{default}。",
};
export default service;
