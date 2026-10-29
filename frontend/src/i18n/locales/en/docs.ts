import type zh from "../zh-TW/docs.ts";
import type { Shape } from "../../types.ts";

const docs: Shape<typeof zh> = {
  "ask.retry": "Check again",
  "ask.title": "Ask AI",
  "ask.subtitle": "Answers from your local Yunshu engine",
  "ask.answers": "Documentation answers",
  "ask.welcome": "Ask anything about Yunshu",
  "ask.local": "Retrieved from these docs. Generated on your attached engine.",
  "ask.waiting": "Reading the docs…",
  "ask.sources": "Sources",
  "ask.copy": "Copy answer",
  "ask.good": "Helpful answer",
  "ask.bad": "Unhelpful answer",
  "ask.placeholder": "Ask a question or follow up…",
  "ask.stop": "Stop answer",
  "ask.send": "Send question",
  "ask.disclaimer": "AI can make mistakes. Check the sources.",
  "ask.clear": "Clear",
  "ask.suggestion.start": "How do I get started?",
  "ask.suggestion.models": "Which models are supported?",
  "ask.suggestion.api": "How do I use the chat API?",
  "ask.error.measurement_paused":
    "Ask AI is paused during measurement. Try again when measurement finishes.",
  "ask.error.no_engine": "Start an engine to ask the docs.",
  "ask.error.no_model": "Start an engine with a text model to ask the docs.",
  "ask.error.question_too_long":
    "The question or a history message is too long.",
  "ask.error.history_too_long":
    "Too much chat history. Clear the conversation and try again.",
  "ask.error.ask_in_flight": "An answer is already running in this tab.",
  "ask.error.context_too_small":
    "The model context is too small. Shorten the question.",
  "ask.error.docs_index_missing":
    "Rebuild the console to create the docs index.",
  "ask.error.history_disabled": "Enable console history to save feedback.",
  "ask.error.stream_incomplete": "The answer stream ended early. Please retry.",
  "ask.error.auth_required":
    "Enter your engine API key in the connection settings.",
  "ask.error.engine_error":
    "The engine could not answer. Check its status and try again.",
  "ask.error.copy_failed": "Could not copy the answer.",

  "reference.openaiSdk": "OpenAI SDK",
  "reference.anthropicSdk": "Anthropic SDK",
  "reference.proxy": "Console same-origin proxy",
  "actions.chatgpt": "ChatGPT",
  "actions.claude": "Claude",
  "reference.link": "OpenAPI ↗",
  sse: "Server-Sent Events",
  "toc.title": "On this page",
  prev: "Previous",
  next: "Next",
  copyLink: "Copy link to this section",
  loading: "Loading the page…",
  "error.title": "Could not load this page",
  "error.body": "The docs page failed to load. Reload and try again.",
  "notFound.title": "No such docs page",
  "notFound.body": "This address has no docs page.",
  "notFound.back": "Back to the docs home",
  "link.api": "Full API reference",
  "link.settings": "What every setting does",
  "link.keys": "Authentication and keys",
  "link.lead": "Docs",
  "actions.copy": "Copy Markdown",
  "actions.copied": "Copied",
  "actions.failed": "Could not copy this page",
  updated: "Last updated",
  reference: "Generated API reference",
  "reference.description":
    "Interactive reference generated from the engine OpenAPI schema. Requests use this console’s same-origin proxy.",
  "reference.guides": "Endpoint guides",
  "reference.operation": "API operation",
  "fd.openSidebar": "Open page navigation",
  "fd.closeSidebar": "Close page navigation",
  "fd.noHeadings": "No headings on this page",
  "fd.copyText": "Copy code",
  "fd.copiedText": "Code copied",
  "fd.collapse": "Collapse page navigation",
};

export default docs;
