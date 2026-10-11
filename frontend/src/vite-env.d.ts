/// <reference types="vite/client" />

declare module "*.mdx" {
  import type { ComponentType } from "react";
  const Component: ComponentType<{ components?: Record<string, unknown> }>;
  export default Component;
}
declare module "virtual:docs-toc-en" {
  const toc: import("./docs/types").DocsToc;
  export default toc;
}
declare module "virtual:docs-toc-zh-TW" {
  const toc: import("./docs/types").DocsToc;
  export default toc;
}
declare module "virtual:docs-toc-zh-CN" {
  const toc: import("./docs/types").DocsToc;
  export default toc;
}
declare module "virtual:docs-search-en" {
  const entries: import("./docs/types").SearchEntry[];
  export default entries;
}
declare module "virtual:docs-search-zh-TW" {
  const entries: import("./docs/types").SearchEntry[];
  export default entries;
}
declare module "virtual:docs-search-zh-CN" {
  const entries: import("./docs/types").SearchEntry[];
  export default entries;
}
