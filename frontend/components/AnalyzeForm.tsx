"use client";

import { useId, useState } from "react";
import { parsePlaylistId } from "@/lib/format";
import { SLIDER_DEFAULT, sliderToListeners } from "@/lib/slider";
import type { AnalyzeRequest } from "@/lib/types";
import { ListenerSlider } from "./ListenerSlider";

const LIMITS = [10, 20, 30] as const;

export function AnalyzeForm({
  onSubmit,
  busy,
  initialUrl = "",
}: {
  onSubmit: (req: AnalyzeRequest) => void;
  busy: boolean;
  initialUrl?: string;
}) {
  const [url, setUrl] = useState(initialUrl);
  const [t, setT] = useState(SLIDER_DEFAULT);
  const [limit, setLimit] = useState<number>(20);
  const [error, setError] = useState<string | null>(null);
  const errId = useId();

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const id = parsePlaylistId(url);
    if (!id) {
      setError("That doesn't look like a Spotify playlist link. Paste a link like open.spotify.com/playlist/… or a playlist ID.");
      return;
    }
    setError(null);
    onSubmit({ playlist_url: url.trim(), max_listeners: sliderToListeners(t), limit });
  }

  return (
    <form onSubmit={submit} noValidate className="space-y-8 rounded-2xl border border-line bg-surface p-5 sm:p-7">
      <div>
        <label htmlFor="playlist" className="font-medium">
          Spotify playlist link
        </label>
        <input
          id="playlist"
          type="text"
          inputMode="url"
          autoComplete="off"
          spellCheck={false}
          placeholder="https://open.spotify.com/playlist/…"
          value={url}
          onChange={(e) => {
            setUrl(e.target.value);
            if (error) setError(null);
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
          <p className="mt-2 text-sm text-muted">The playlist must be public.</p>
        )}
      </div>

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
