import type zh from "../zh-TW/api.ts";
import type { Shape } from "../../types.ts";

const api: Shape<typeof zh> = {
  "address.anthropic": "Anthropic Base URL",
  "address.description":
    "粘贴到 SDK 或编程代理的 Base URL；示例命令使用下方选定的模型。",
  "address.modelAria": "接入使用的模型",
  "address.modelLabel": "命令中使用的模型",
  "address.openai": "OpenAI API Base URL",
  "address.title": "服务地址",
  "catalog.copy": "复制 {method} {path} 的完整网址",
  "catalog.description": "从当前引擎的 OpenAPI 定义读取，共 {count} 个操作。",
  "catalog.docs": "API 文档",
  "catalog.empty": "没有匹配的 API",
  "catalog.emptyNote": "此目录反映服务实际提供的路由，不会推测未提供的功能。",
  "catalog.error": "无法获取 API 定义",
  "catalog.loading": "正在读取 API 定义…",
  "catalog.search.aria": "搜索 API",
  "catalog.search.placeholder": "搜索路径、方法或功能",
  "catalog.table.aria": "服务 API 目录",
  "catalog.table.function": "功能",
  "catalog.table.method": "方法",
  "catalog.table.path": "路径",
  "catalog.tag.all": "全部功能",
  "catalog.tag.aria": "API 类别",
  "catalog.tag.other": "其他",
  "catalog.title": "此服务的完整 API",
  "catalog.unavailable": "尚未获取 API 定义",
  "clients.note":
    "命令只引用环境变量 {env}（未设置时为 {fallback}），不会写入真实令牌。",
  "clients.title": "客户端设置",
  "integration.anthropic.description":
    "base_url 不含 /v1，SDK 会自行添加路径。",
  "integration.claudeCode.description":
    "使用 Anthropic Messages 接口；所有模型别名都指向当前的模型。",
  "integration.codex.description":
    "加入 ~/.codex/config.toml，使用 Responses 接口，并从环境变量 YUNSHU_AUTH_TOKEN 读取令牌。",
  "integration.curl.description": "直接调用 chat completions。",
  "integration.openai.description": "标准 OpenAI 接口，只需替换 base_url。",
  "integration.opencode.description":
    "加入 opencode.json；limit 请按模型实际的上下文长度调整。",
  "clients.tokenWhere":
    "服务需要验证时，把 {env} 设为服务的令牌，或使用在“API 密钥”页创建的密钥。",
  "clients.keysLink": "API 密钥",
  "launch.label": "一行命令（自动配置）",
  "page.description": "使用熟悉的 SDK 或编程代理，让你的应用连接本机模型。",
  "page.title": "API 接入",
};
export default api;
