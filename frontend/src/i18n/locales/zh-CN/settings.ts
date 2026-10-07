import type zh from "../zh-TW/settings.ts";
import type { Shape } from "../../types.ts";

const settings: Shape<typeof zh> = {
  title: "设置",
  description: "连接到本机服务，并调整控制台偏好。",
  "nav.label": "设置分类",
  "nav.connection": "引擎连接",
  "nav.appearance": "外观",
  "nav.models": "模型保留",
  "nav.config": "生效设置",
  "nav.shortcuts": "键盘快捷键",
  "connection.title": "引擎连接",
  "connection.description": "控制台要连接到哪一个 Yunshu 服务。",
  "connection.url": "服务地址",
  "connection.urlHelp":
    "默认使用同一个服务来源。开发模式下由 Vite 转发到本机 8000 端口。",
  "connection.token": "访问令牌",
  "connection.tokenHelp":
    "只保留在此页内存中；刷新后需要重新输入。更改服务地址会清除令牌。",
  "connection.tokenPlaceholder": "服务未启用验证时可留空",
  "connection.showToken": "显示令牌",
  "connection.hideToken": "隐藏令牌",
  "connection.save": "保存并连接",
  "connection.invalid":
    "请使用不含账号密码、查询参数或锚点的 HTTP(S) 服务地址。",
  "connection.invalidShort": "服务地址无效",
  "appearance.title": "外观",
  "appearance.dark": "深色界面",
  "appearance.darkHelp": "保存在此浏览器中。",
  "appearance.language": "语言",
  "appearance.languageHelp": "保存在此浏览器中；切换后立即生效。",
  "permissions.title": "模型操作权限",
  "permissions.description":
    "加载与卸载需要服务允许的权限；若出现 401，请使用服务配置的 YUNSHU_AUTH_TOKEN。此页不会修改引擎启动参数或关闭验证。",
  "shortcuts.title": "键盘快捷键",
  "shortcuts.palette": "打开命令面板，快速切换页面与模型",
  "shortcuts.send": "在测试台发送消息",
  "shortcuts.close": "关闭对话框与菜单",
};
export default settings;
