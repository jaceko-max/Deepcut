"use client";

import { useState } from "react";
import { formatBpm, formatCompact } from "@/lib/format";
import type { Result, Vibe } from "@/lib/types";
import { FeatureBars } from "./FeatureBars";
import { PlayButton } from "./PreviewPlayer";

export function ResultCard({ result, vibe, showVibe }: { result: Result; vibe?: Vibe; showVibe: boolean }) {
  const [open, setOpen] = useState(false);
  const panelId = `cmp-${result.deezer_id}`;
  return (
    <li className="rounded-xl border border-line bg-surface p-4 sm:p-5">
      <div className="flex gap-3 sm:gap-4">
        <div className="relative shrink-0">
          {result.artwork_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={result.artwork_url} alt="" width={80} height={80} loading="lazy" className="size-16 rounded-md bg-surface-2 object-cover sm:size-20" />
          ) : (
            <div aria-hidden className="size-16 rounded-md bg-surface-2 sm:size-20" />
          )}
          <span className="display num absolute -left-2 -top-2 grid min-w-6 place-items-center rounded-md bg-fg px-1 text-xs font-semibold text-bg">
            {result.rank}
          </span>
        </div>

        <div className="min-w-0 flex-1">
          <h3 className="display truncate text-lg font-semibold leading-tight">{result.title}</h3>
          <p className="truncate text-muted">
            {result.artist}
            {result.album ? ` · ${result.album}` : ""}
          </p>
          <p className="num mt-1 text-sm">
            <span className="text-accent-text">{formatCompact(result.listeners)} listeners</span>
            <span className="text-muted"> · {formatBpm(result.bpm)} BPM</span>
          </p>
        </div>

        <div
          className="display num grid size-14 shrink-0 -rotate-6 place-items-center self-start rounded-full bg-stamp text-stamp-ink sm:size-16"
          role="img"
          aria-label={`Match score ${result.match_score} out of 100`}
        >
          <span className="text-xl font-bold leading-none sm:text-2xl">{result.match_score}</span>
        </div>
      </div>

      <p className="mt-3 text-sm">{result.why}</p>
      {showVibe && vibe && <p className="mt-1 text-xs text-muted">Matches: {vibe.label}</p>}

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
        <PlayButton result={result} />
        <a className="inline-flex min-h-11 items-center text-sm underline decoration-line underline-offset-4 hover:decoration-accent" href={result.deezer_url} target="_blank" rel="noreferrer">
          Open on Deezer
        </a>
        <a className="inline-flex min-h-11 items-center text-sm underline decoration-line underline-offset-4 hover:decoration-accent" href={result.spotify_search_url} target="_blank" rel="noreferrer">
          Find on Spotify
        </a>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-controls={panelId}
          className="ml-auto inline-flex min-h-11 items-center text-sm text-muted hover:text-fg"
        >
          {open ? "Hide beat flow" : "Show beat flow"}
        </button>
      </div>

      {open && (
        <div id={panelId} className="mt-3 border-t border-line pt-4">
          <FeatureBars features={result.features} compare={vibe?.features} />
          {vibe && <p className="mt-2 text-xs text-muted">The white tick marks your playlist&apos;s vibe.</p>}
        </div>
      )}
    </li>
  );
}
