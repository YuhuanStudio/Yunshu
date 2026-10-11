import type zh from "../zh-TW/docs.ts";
import type { Shape } from "../../types.ts";

const docs: Shape<typeof zh> = {
  "nav.label": "Documentation",
  "nav.jump": "Jump to a docs page",
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
};

export default docs;
