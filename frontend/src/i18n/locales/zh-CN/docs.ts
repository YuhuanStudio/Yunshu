import type zh from "../zh-TW/docs.ts";
import type { Shape } from "../../types.ts";

const docs: Shape<typeof zh> = {
  "nav.label": "文档目录",
  "nav.jump": "跳到文档页面",
  "toc.title": "本页内容",
  prev: "上一页",
  next: "下一页",
  copyLink: "复制此段落的链接",
  loading: "载入文档中…",
  "error.title": "无法载入这份文档",
  "error.body": "文档页面载入失败，请刷新后再试。",
  "notFound.title": "找不到这份文档",
  "notFound.body": "这个网址没有对应的文档页面。",
  "notFound.back": "回到文档首页",
  "link.api": "完整 API 参考",
  "link.settings": "每个设置的说明",
  "link.keys": "验证与密钥说明",
  "link.lead": "文档",
};

export default docs;
