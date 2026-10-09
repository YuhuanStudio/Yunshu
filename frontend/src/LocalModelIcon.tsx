import { useEffect, useState } from "react";
import {
  ModelIcon,
  getDeveloperIconPath,
  getModelDeveloperId,
} from "@yuhuanowo/yunui/ai";

// YunUI's ModelIcon defaults to jsDelivr; the console is local-first, so
// resolve the icon files that ship in the package to bundled asset URLs. Each
// icon loads on demand, so the files are not inlined into one chunk.
const bundledIcons = import.meta.glob<string>(
  "../node_modules/@yuhuanowo/yunui/icons/models/*.{webp,png,jpeg}",
  { query: "?url", import: "default" },
);
const iconByFile = new Map(
  Object.entries(bundledIcons).map(([path, load]) => [
    path.split("/").at(-1) ?? path,
    load,
  ]),
);

export default function LocalModelIcon({
  id,
  size = 28,
}: {
  id: string;
  size?: number;
}) {
  const developer = getModelDeveloperId(id);
  const file = getDeveloperIconPath(developer)?.split("/").at(-1);
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    const load = file ? iconByFile.get(file) : undefined;
    setUrl(null);
    void load?.().then(
      (value) => live && setUrl(value),
      () => undefined,
    );
    return () => {
      live = false;
    };
  }, [file]);
  // The slot has its final size from the first paint; the icon fills it when it loads,
  // so nothing around it moves.
  return (
    <span
      data-icon-slot=""
      aria-hidden={url ? undefined : "true"}
      className="inline-flex shrink-0 items-center justify-center"
      style={{ width: size, height: size }}
    >
      {url && (
        <ModelIcon
          iconUrl={url}
          developer={developer}
          provider={developer}
          size={size}
          rounded
        />
      )}
    </span>
  );
}
