import { source } from "@/lib/source";
import { createFromSource } from "fumadocs-core/search/server";
import { createTokenizer } from "@orama/tokenizers/mandarin";

export const dynamic = "force-static";

const chinese = {
  components: { tokenizer: createTokenizer() },
  search: { threshold: 0, tolerance: 0 },
};

export const { staticGET: GET } = createFromSource(source, {
  localeMap: {
    en: { language: "english" },
    "zh-CN": chinese,
    "zh-TW": chinese,
  },
});
