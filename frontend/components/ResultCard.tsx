"use client";

import { useState } from "react";
import { formatBpm, formatCompact, tierBadge } from "@/lib/format";
import type { Result, Vibe } from "@/lib/types";
import { FeatureBars } from "./FeatureBars";
import { PlayButton } from "./PreviewPlayer";

function Thumb({ down, filled }: { down?: boolean; filled: boolean }) {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden className={down ? "rotate-180" : undefined}
      fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round">
      <path d="M7 10v11H3V10h4Zm0 0 4.2-7.2c.4-.6 1.1-.9 1.8-.7 1 .3 1.6 1.3 1.4 2.3L14 9h5.6c1.4 0 2.4 1.3 2.1 2.6l-1.6 7.2c-.2 1.2-1.3 2.2-2.6 2.2H7" />
    </svg>
  );
}

export function ResultCard({
  result,
  vibe,
  showVibe,
  feedback,
  onVote,
  fresh = false,
}: {
  result: Result;
  vibe?: Vibe;
  showVibe: boolean;
  /** Current thumb for this track. Leaving `onVote` undefined hides the buttons (read-only share view). */
  feedback?: 1 | -1;
  onVote?: (v: 1 | -1) => void;
  /** Pulled in from the reserve because of the user's feedback. */
  fresh?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const panelId = `cmp-${result.deezer_id}`;
  const deepCut = result.source === "deep_cut";
  const tier = deepCut ? null : tierBadge(result.tier);
  return (
    <li className={`rounded-xl border bg-surface p-4 sm:p-5 ${feedback === 1 ? "border-accent" : "border-line"}`}>
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
          {fresh && (
            <p className="mb-1">
              <span className="inline-flex items-center rounded-full bg-accent px-2 py-0.5 text-xs font-medium text-accent-ink">
                New pick from your feedback
              </span>
            </p>
          )}
          {deepCut && (
            <p className="mb-1">
              <span
                className="inline-flex items-center rounded-full border border-accent/60 px-2 py-0.5 text-xs font-medium text-accent-text"
                title="A lesser-known track by an artist already in your playlist"
              >
                Deep cut<span aria-hidden> · </span>
                <span className="sr-only">: a lesser-known track </span>from your playlist&apos;s artist
              </span>
            </p>
          )}
          <h3 className="display truncate text-lg font-semibold leading-tight">{result.title}</h3>
          <p className="truncate text-muted">
            {result.artist}
            {result.album ? ` · ${result.album}` : ""}
          </p>
          <p className="num mt-1 text-sm">
            <span className="text-accent-text">
              {formatCompact(result.listeners)} {deepCut ? "artist listeners" : "listeners"}
            </span>
            <span className="text-muted"> · {formatBpm(result.bpm)} BPM</span>
            {tier && (
              <span className="ml-2 inline-flex items-center rounded-full border border-line px-1.5 py-px align-middle text-[11px] font-medium text-muted" title={tier.title}>
                {tier.label}
              </span>
            )}
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
        {onVote && (
          <div className="flex items-center gap-1">
            {([1, -1] as const).map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => onVote(v)}
                aria-pressed={feedback === v}
                aria-label={v === 1 ? `More like ${result.title}` : `Less like ${result.title}`}
                title={v === 1 ? "More like this" : "Less like this (removes it)"}
                className={`grid size-11 place-items-center rounded-lg border ${
                  feedback === v ? "border-accent bg-accent text-accent-ink" : "border-line text-muted hover:border-accent hover:text-fg"
                }`}
              >
                <Thumb down={v === -1} filled={feedback === v} />
              </button>
            ))}
          </div>
        )}
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
