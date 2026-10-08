import type { Lang } from "./i18n";

export interface Messages {
  siteTitle: string;
  docs: string;
  api: string;
  guides: string;
  developers: string;
  github: string;
  search: string;
  searchNoResult: string;
  toc: string;
  next: string;
  previous: string;
  theme: string;
  language: string;
  menu: string;
  heroBadge: string;
  heroTitle: string;
  heroAccent: string;
  heroSubtitle: string;
  start: string;
  apiRef: string;
  facts: string[];
  sections: { title: string; desc: string; href: string }[];
  codeTitle: string;
  footer: string;
  copyMd: string;
  copiedMd: string;
  openGithub: string;
  feedbackQ: string;
  good: string;
  bad: string;
  feedbackThanks: string;
  feedbackPlaceholder: string;
  feedbackOpen: string;
  notFound: string;
  backHome: string;
}

export const MESSAGES: Record<Lang, Messages> = {
  en: {
    siteTitle: "Yunshu",
    docs: "Docs",
    api: "API",
    guides: "Guides",
    developers: "Developers",
    github: "GitHub",
    search: "Search documentation...",
    searchNoResult: "No results found",
    toc: "On this page",
    next: "Next",
    previous: "Previous",
    theme: "Theme",
    language: "Language",
    menu: "Menu",
    heroBadge: "Local inference for Apple Silicon",
    heroTitle: "A local LLM engine that speaks",
    heroAccent: "every API you already use",
    heroSubtitle:
      "Yunshu serves MLX models behind OpenAI, Anthropic and Ollama compatible endpoints, with prefix caching, speculative decoding and constrained output. These docs cover the HTTP API, the guides and the engine itself.",
    start: "Get started",
    apiRef: "API reference",
    facts: ["OpenAI compatible", "Anthropic compatible", "Ollama compatible", "Runs on your Mac"],
    sections: [
      { title: "Getting started", desc: "Install Yunshu, pull a model, run the server and send the first request.", href: "/docs/getting-started/install" },
      { title: "API reference", desc: "One page per endpoint group: parameters, examples, streaming events, errors and what is not supported.", href: "/docs/api/overview" },
      { title: "Guides", desc: "Coding agents, web search, prompt caching, structured output, configuration and troubleshooting.", href: "/docs/guides/agents" },
      { title: "Developers", desc: "How the engine is built, how to contribute, how changes are verified and which upstream code is tracked.", href: "/docs/developers/architecture" },
    ],
    codeTitle: "Three commands to a running endpoint",
    footer: "Documentation for the Yunshu inference engine.",
    copyMd: "Copy Markdown",
    copiedMd: "Copied",
    openGithub: "Open in GitHub",
    feedbackQ: "Was this page helpful?",
    good: "Yes",
    bad: "No",
    feedbackThanks: "Thanks for the feedback.",
    feedbackPlaceholder: "What is missing or wrong?",
    feedbackOpen: "Open a GitHub issue",
    notFound: "This page does not exist.",
    backHome: "Back to the docs",
  },
  "zh-TW": {
    siteTitle: "Yunshu",
    docs: "文件",
    api: "API",
    guides: "指南",
    developers: "開發者",
    github: "GitHub",
    search: "搜尋文件...",
    searchNoResult: "找不到結果",
    toc: "本頁目錄",
    next: "下一頁",
    previous: "上一頁",
    theme: "主題",
    language: "語言",
    menu: "選單",
    heroBadge: "為 Apple Silicon 打造的本機推論",
    heroTitle: "一個本機 LLM 引擎，說得懂",
    heroAccent: "你已經在用的每一種 API",
    heroSubtitle:
      "Yunshu在 OpenAI、Anthropic 與 Ollama 相容的端點之後提供 MLX 模型，並具備前綴快取、推測式解碼與受限輸出。這份文件涵蓋 HTTP API、使用指南與引擎本身。",
    start: "開始使用",
    apiRef: "API 參考",
    facts: ["相容 OpenAI", "相容 Anthropic", "相容 Ollama", "在你的 Mac 上執行"],
    sections: [
      { title: "快速開始", desc: "安裝 Yunshu、下載模型、啟動伺服器並送出第一個請求。", href: "/docs/getting-started/install" },
      { title: "API 參考", desc: "每個端點群組一頁：參數、範例、串流事件、錯誤，以及尚未支援的部分。", href: "/docs/api/overview" },
      { title: "指南", desc: "程式碼代理、網路搜尋、提示快取、結構化輸出、設定與疑難排解。", href: "/docs/guides/agents" },
      { title: "開發者", desc: "引擎如何組成、如何貢獻、變更如何驗證，以及追蹤的上游程式碼。", href: "/docs/developers/architecture" },
    ],
    codeTitle: "三個指令，得到一個運作中的端點",
    footer: "Yunshu 推論引擎文件。",
    copyMd: "複製 Markdown",
    copiedMd: "已複製",
    openGithub: "在 GitHub 開啟",
    feedbackQ: "這一頁有幫助嗎？",
    good: "有",
    bad: "沒有",
    feedbackThanks: "感謝你的回饋。",
    feedbackPlaceholder: "缺少或有誤的地方是什麼？",
    feedbackOpen: "在 GitHub 開立 issue",
    notFound: "找不到這一頁。",
    backHome: "回到文件",
  },
  "zh-CN": {
    siteTitle: "Yunshu",
    docs: "文档",
    api: "API",
    guides: "指南",
    developers: "开发者",
    github: "GitHub",
    search: "搜索文档...",
    searchNoResult: "未找到结果",
    toc: "本页目录",
    next: "下一页",
    previous: "上一页",
    theme: "主题",
    language: "语言",
    menu: "菜单",
    heroBadge: "为 Apple Silicon 打造的本机推理",
    heroTitle: "一个本机 LLM 引擎，听得懂",
    heroAccent: "你已经在用的每一种 API",
    heroSubtitle:
      "Yunshu在 OpenAI、Anthropic 与 Ollama 兼容的端点之后提供 MLX 模型，并具备前缀缓存、推测式解码与受限输出。这份文档涵盖 HTTP API、使用指南与引擎本身。",
    start: "开始使用",
    apiRef: "API 参考",
    facts: ["兼容 OpenAI", "兼容 Anthropic", "兼容 Ollama", "在你的 Mac 上运行"],
    sections: [
      { title: "快速开始", desc: "安装 Yunshu、下载模型、启动服务器并发出第一个请求。", href: "/docs/getting-started/install" },
      { title: "API 参考", desc: "每个端点群组一页：参数、示例、流式事件、错误，以及尚未支持的部分。", href: "/docs/api/overview" },
      { title: "指南", desc: "编程代理、网络搜索、提示缓存、结构化输出、配置与故障排查。", href: "/docs/guides/agents" },
      { title: "开发者", desc: "引擎如何组成、如何贡献、变更如何验证，以及追踪的上游代码。", href: "/docs/developers/architecture" },
    ],
    codeTitle: "三个命令，得到一个运行中的端点",
    footer: "Yunshu 推理引擎文档。",
    copyMd: "复制 Markdown",
    copiedMd: "已复制",
    openGithub: "在 GitHub 打开",
    feedbackQ: "这一页有帮助吗？",
    good: "有",
    bad: "没有",
    feedbackThanks: "感谢你的反馈。",
    feedbackPlaceholder: "缺少或有误的地方是什么？",
    feedbackOpen: "在 GitHub 提交 issue",
    notFound: "找不到这一页。",
    backHome: "回到文档",
  },
};
