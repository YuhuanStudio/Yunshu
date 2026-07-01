"use client";

import { useState, useEffect } from "react";
import { ModelSelect, type ModelSelectOption } from "yunui/ai";
import { StatusIndicator } from "yunui";
import { ModelTypeGlyph, guessModelType, MODEL_TYPES, type ModelType } from "@/lib/model-type";
import type { Model } from "@/lib/types";

/**
 * Yunshu's model picker — the thin adapter over YunUI's `ModelSelect` (the same
 * component Yunxin uses), mapping our minimal `/v1/models` rows into grouped,
 * searchable options with a type glyph and a loaded indicator. Pinned models are
 * remembered in localStorage. Mirrors Yunxin's model-selector wrapper so both
 * apps present models identically.
 */

const PIN_KEY = "yunshu_pinned_models";

function toOptions(models: Model[]): ModelSelectOption[] {
  return models.map((m) => {
    const type = (m.type ?? guessModelType(m.id)) as ModelType;
    return {
      id: m.id,
      label: m.id,
      group: type,
      groupLabel: MODEL_TYPES[type]?.label ?? String(type),
      searchText: m.id,
      icon: <ModelTypeGlyph type={type} size={16} />,
      badges: m.loaded ? <StatusIndicator status="online" /> : undefined,
    };
  });
}

export function ModelPicker({
  models,
  value,
  onChange,
  className,
}: {
  models: Model[];
  value: string;
  onChange: (id: string) => void;
  className?: string;
}) {
  const [pinned, setPinned] = useState<string[]>([]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(PIN_KEY);
      if (raw) setPinned(JSON.parse(raw));
    } catch {
      /* ignore */
    }
  }, []);

  const togglePin = (id: string) =>
    setPinned((prev) => {
      const next = prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id];
      localStorage.setItem(PIN_KEY, JSON.stringify(next));
      return next;
    });

  return (
    <ModelSelect
      options={toOptions(models)}
      value={value}
      onChange={onChange}
      pinned={pinned}
      onTogglePin={togglePin}
      className={className}
    />
  );
}
