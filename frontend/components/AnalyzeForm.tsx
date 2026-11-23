"use client";

import { useId, useMemo, useState } from "react";
import { isPlaylistLink } from "@/lib/format";
import { MAX_PASTED_TRACKS, MIN_PASTED_TRACKS, parseTracks } from "@/lib/paste";
import { SLIDER_DEFAULT, sliderToListeners } from "@/lib/slider";
import type { AnalyzeRequest } from "@/lib/types";
import { ListenerSlider } from "./ListenerSlider";

const LIMITS = [10, 20, 30] as const;
type Mode = "link" | "paste";

const PLACEHOLDER = `Low Hum Collective - Undertow
Undertow by Low Hum Collective
Or paste a CSV exported from Exportify`;

export function AnalyzeForm({
  onSubmit,
  busy,
  initialUrl = "",
  initialMode = "link",
}: {
  onSubmit: (req: AnalyzeRequest) => void;
  busy: boolean;
  initialUrl?: string;
  initialMode?: Mode;
}) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [url, setUrl] = useState(initialUrl);
  const [text, setText] = useState("");
  const [t, setT] = useState(SLIDER_DEFAULT);
  const [limit, setLimit] = useState<number>(20);
  const [error, setError] = useState<string | null>(null);
  const errId = useId();
  const countId = useId();
  const parsed = useMemo(() => parseTracks(text), [text]);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const base = { max_listeners: sliderToListeners(t), limit };
    if (mode === "link") {
      if (!isPlaylistLink(url)) {
        setError(
          "That doesn't look like a Spotify or Apple Music playlist link. Paste a link like open.spotify.com/playlist/… or music.apple.com/…/playlist/….",
        );
        return;
      }
      setError(null);
      onSubmit({ playlist_url: url.trim(), ...base });
      return;
    }
    if (parsed.tracks.length < MIN_PASTED_TRACKS) {
      setError(
        parsed.tracks.length === 0
          ? "No tracks found. Put one track per line, like “Artist - Title”."
          : `Add at least ${MIN_PASTED_TRACKS} tracks so Deepcut has enough audio to learn from (found ${parsed.tracks.length}).`,
      );
      return;
    }
    setError(null);
    onSubmit({ tracks: parsed.tracks, ...base });
  }

  const clear = () => error && setError(null);

  return (
    <form onSubmit={submit} noValidate className="space-y-8 rounded-2xl border border-line bg-surface p-5 sm:p-7">
      <fieldset>
        <legend className="sr-only">Where are your tracks?</legend>
        <div className="inline-flex rounded-lg border border-line p-1" role="radiogroup" aria-label="Playlist source">
          {(
            [
              ["link", "Spotify / Apple Music link"],
              ["paste", "Paste tracks"],
            ] as const
          ).map(([m, label]) => (
            <label key={m} className="relative">
              <input
                type="radio"
                name="mode"
                value={m}
                checked={mode === m}
                onChange={() => {
                  setMode(m);
                  setError(null);
                }}
                className="peer sr-only"
              />
              <span className="flex min-h-11 cursor-pointer items-center justify-center rounded-md px-4 text-sm peer-checked:bg-accent peer-checked:font-semibold peer-checked:text-accent-ink peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent-text">
                {label}
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      {mode === "link" ? (
        <div>
          <label htmlFor="playlist" className="font-medium">
            Spotify or Apple Music playlist link
          </label>
          <input
            id="playlist"
            type="text"
            inputMode="url"
            autoComplete="off"
            spellCheck={false}
            placeholder="https://open.spotify.com/playlist/… or https://music.apple.com/…/playlist/…"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              clear();
            }}
            aria-invalid={error ? true : undefined}
            aria-describedby={error ? errId : undefined}
            className="mt-2 min-h-12 w-full rounded-lg border border-line bg-bg px-4 text-base placeholder:text-muted/70"
          />
          {error ? (
            <p id={errId} role="alert" className="mt-2 text-sm text-danger">
              {error}
            </p>
          ) : (
            <p className="mt-2 text-sm text-muted">The playlist must be public. Apple Music: Share → Copy Link.</p>
          )}
        </div>
      ) : (
        <div>
          <label htmlFor="tracks" className="font-medium">
            Your tracks
          </label>
          <textarea
            id="tracks"
            rows={8}
            autoComplete="off"
            spellCheck={false}
            placeholder={PLACEHOLDER}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              clear();
            }}
            aria-invalid={error ? true : undefined}
            aria-describedby={`${countId}${error ? ` ${errId}` : ""}`}
            className="mt-2 min-h-44 w-full rounded-lg border border-line bg-bg px-4 py-3 text-base placeholder:text-muted/70"
          />
          <p id={countId} className="num mt-2 text-sm text-muted" aria-live="polite">
            {parsed.tracks.length} of {MAX_PASTED_TRACKS} tracks
            {parsed.format === "csv" ? " (CSV)" : ""}
            {parsed.skipped > 0 ? ` · ${parsed.skipped} ${parsed.skipped === 1 ? "line" : "lines"} not understood` : ""}
            {parsed.overflow > 0 ? ` · ${parsed.overflow} over the limit will be left out` : ""}
          </p>
          {error ? (
            <p id={errId} role="alert" className="mt-1 text-sm text-danger">
              {error}
            </p>
          ) : (
            <p className="mt-1 text-sm text-muted">
              One per line: “Artist - Title” or “Title by Artist”. A CSV from Exportify works too. Pasting also skips Spotify&apos;s
              100-track limit.
            </p>
          )}
        </div>
      )}

      <ListenerSlider value={t} onChange={setT} />

      <fieldset>
        <legend className="font-medium">How many tracks</legend>
        <div className="mt-2 inline-flex rounded-lg border border-line p-1" role="radiogroup">
          {LIMITS.map((n) => (
            <label key={n} className="relative">
              <input
                type="radio"
                name="limit"
                value={n}
                checked={limit === n}
                onChange={() => setLimit(n)}
                className="peer sr-only"
              />
              <span className="num flex min-h-11 min-w-16 cursor-pointer items-center justify-center rounded-md px-4 text-sm peer-checked:bg-accent peer-checked:font-semibold peer-checked:text-accent-ink peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent-text">
                {n}
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      <button
        type="submit"
        disabled={busy}
        className="display min-h-13 w-full rounded-lg bg-accent px-6 text-lg font-semibold text-accent-ink transition hover:brightness-110 disabled:opacity-60 sm:w-auto"
      >
        {busy ? "Starting…" : "Start digging"}
      </button>
    </form>
  );
}
