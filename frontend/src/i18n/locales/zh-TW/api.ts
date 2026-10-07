const api = {
  "address.anthropic": "Anthropic Base URL",
  "address.description":
    "貼進 SDK 或程式代理的 Base URL；範例命令使用下方選定的模型。",
  "address.modelAria": "接入使用的模型",
  "address.modelLabel": "命令中使用的模型",
  "address.openai": "OpenAI API Base URL",
  "address.title": "服務位址",
  "catalog.copy": "複製 {method} {path} 的完整網址",
  "catalog.description": "從目前引擎的 OpenAPI 定義讀取，共 {count} 個操作。",
  "catalog.docs": "API 文件",
  "catalog.empty": "沒有符合的 API",
  "catalog.emptyNote": "此目錄反映服務實際提供的路由，不會推測未提供的功能。",
  "catalog.error": "無法取得 API 定義",
  "catalog.loading": "讀取 API 定義…",
  "catalog.search.aria": "搜尋 API",
  "catalog.search.placeholder": "搜尋路徑、方法或功能",
  "catalog.table.aria": "服務 API 目錄",
  "catalog.table.function": "功能",
  "catalog.table.method": "方法",
  "catalog.table.path": "路徑",
  "catalog.tag.all": "全部功能",
  "catalog.tag.aria": "API 類別",
  "catalog.tag.other": "其他",
  "catalog.title": "此服務的完整 API",
  "catalog.unavailable": "API 定義尚未取得",
  "clients.note":
    "命令只引用環境變數 {env}（未設定時為 {fallback}），不會寫入真實權杖。",
  "clients.title": "用戶端設定",
  "integration.anthropic.description":
    "base_url 不含 /v1，SDK 會自行加上路徑。",
  "integration.claudeCode.description":
    "走 Anthropic Messages 介面；所有模型別名都指向目前的模型。也可執行 yunshu launch claude 自動設定。",
  "integration.codex.description":
    "加入 ~/.codex/config.toml，使用 Responses 介面，並設定環境變數 YUNSHU_API_KEY。也可執行 yunshu launch codex。",
  "integration.curl.description": "直接呼叫 chat completions。",
  "integration.openai.description": "標準 OpenAI 介面，只需換掉 base_url。",
  "integration.opencode.description":
    "加入 opencode.json；limit 請依模型實際的上下文長度調整。也可執行 yunshu launch opencode。",
  "page.description": "使用熟悉的 SDK 或程式代理，讓你的應用連接本機模型。",
  "page.title": "API 接入",
};
export default api;
