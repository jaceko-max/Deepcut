"use client";

import { useRef } from "react";
import { formatPercent } from "@/lib/format";
import type { Vibe } from "@/lib/types";
import type { VibeSelection } from "@/lib/vibe";

/** Radiogroup of chips: "All", then each vibe with its share. Arrow keys move the selection (roving tabindex). */
export function VibeFilter({
  vibes,
  value,
  counts,
  onChange,
}: {
  vibes: Vibe[];
  value: VibeSelection;
  counts: Map<number, number>;
  onChange: (v: VibeSelection) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const options: { id: VibeSelection; label: string; share?: number; count?: number }[] = [
    { id: null, label: "All" },
    ...vibes.map((v) => ({ id: v.id, label: v.label, share: v.share, count: counts.get(v.id) ?? 0 })),
  ];

  function onKey(e: React.KeyboardEvent, i: number) {
    const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    const next = (i + step + options.length) % options.length;
    onChange(options[next].id);
    ref.current?.querySelectorAll<HTMLButtonElement>('[role="radio"]')[next]?.focus();
  }

  return (
    <div ref={ref} role="radiogroup" aria-label="Filter picks by vibe" className="flex flex-wrap gap-2">
      {options.map((o, i) => {
        const on = value === o.id;
        return (
          <button
            key={o.id ?? "all"}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            onClick={() => onChange(o.id)}
            onKeyDown={(e) => onKey(e, i)}
            className={`min-h-11 rounded-full border px-4 py-1.5 text-left text-sm ${
              on ? "border-accent bg-accent font-semibold text-accent-ink" : "border-line hover:border-accent"
            }`}
          >
            {o.label}
            {o.share !== undefined && (
              <span className={`num ml-2 ${on ? "" : "text-muted"}`}>
                {formatPercent(o.share)}
                <span className="sr-only"> of your playlist, {o.count} picks</span>
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
