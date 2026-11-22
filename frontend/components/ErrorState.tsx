const COPY: Record<string, { title: string; body: string }> = {
  invalid_playlist_url: { title: "That link isn't a playlist", body: "Paste a link like open.spotify.com/playlist/… or a 22-character playlist ID." },
  invalid_request: { title: "Something was off in the request", body: "Check the playlist link and settings, then try again." },
  too_many_jobs: { title: "Slow down a moment", body: "You've started a few searches already. Try again in a few minutes." },
  server_busy: { title: "The crates are crowded", body: "Lots of searches are queued right now. Try again in a minute." },
  job_not_found: {
    title: "This search expired",
    body: "This search is no longer on the server (results are kept up to an hour, and the free server sometimes restarts). Run it again — it's faster the second time.",
  },
  playlist_not_found: { title: "Playlist not found", body: "Check that the link is right and the playlist is public." },
  playlist_unavailable: { title: "Can't read that playlist", body: "It may be private or region-locked. Make it public and try again." },
  playlist_empty: { title: "That playlist is empty", body: "Add some tracks to it, or try a different playlist." },
  no_previews: { title: "No audio to listen to", body: "Deepcut needs 30-second previews to hear the beat, and almost none were available for this playlist. Try another one." },
  no_candidates: { title: "No underground matches", body: "Nothing under the listener cap fit this playlist. Raise the cap and try again." },
  upstream_error: { title: "A music service is struggling", body: "Spotify, Deezer or Last.fm didn't respond properly. Try again shortly." },
  internal_error: { title: "Deepcut tripped", body: "Something broke on our side. Try again; if it keeps happening, try a different playlist." },
  network_error: { title: "Can't reach the server", body: "Check your connection and try again. If the server was asleep, it may need another minute." },
};

export function ErrorState({
  code,
  message,
  retryAfter,
  onRetry,
  retryLabel = "Try again",
}: {
  code: string;
  message?: string | null;
  retryAfter?: number | null;
  onRetry: () => void;
  retryLabel?: string;
}) {
  const c = COPY[code] ?? { title: "Something went wrong", body: "Try again in a moment." };
  const detail = message?.trim();
  return (
    <section role="alert" className="rounded-2xl border border-danger/60 bg-surface p-5 sm:p-7">
      <h2 className="display text-2xl font-semibold">{c.title}</h2>
      <p className="mt-2 max-w-prose text-muted">
        {c.body}
        {retryAfter ? ` Wait about ${Math.ceil(retryAfter / 60)} min before retrying.` : ""}
      </p>
      {detail && detail !== c.body && <p className="mt-2 max-w-prose text-sm text-muted">{detail}</p>}
      <button
        type="button"
        onClick={onRetry}
        className="display mt-5 min-h-12 rounded-lg bg-accent px-6 font-semibold text-accent-ink hover:brightness-110"
      >
        {retryLabel}
      </button>
    </section>
  );
}
