/** 12000 -> "12K", 1234567 -> "1.2M", 950 -> "950". */
export function formatCompact(n: number): string {
  if (!Number.isFinite(n)) return "–";
  const abs = Math.abs(n);
  if (abs < 1000) return String(Math.round(n));
  if (abs < 999_500) {
    const k = n / 1000;
    return `${trim(k < 10 ? k.toFixed(1) : String(Math.round(k)))}K`;
  }
  const m = n / 1_000_000;
  return `${trim(m < 10 ? m.toFixed(1) : String(Math.round(m)))}M`;
}

function trim(s: string): string {
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
}

/** BPM with at most one decimal, e.g. 92.3 -> "92.3", 92 -> "92". */
export function formatBpm(bpm: number): string {
  return trim(bpm.toFixed(1));
}

/** Badge for the popularity tier of a new-artist pick; deep cuts have their own badge, older payloads none. */
export function tierBadge(tier: string | null | undefined): { label: string; title: string } | null {
  if (tier === "rising") return { label: "Rising", title: "Rising artist: 100K+ Last.fm listeners, under your cap" };
  if (tier === "underground") return { label: "Underground", title: "Underground artist: under 100K Last.fm listeners" };
  return null;
}

export function formatPercent(share: number): string {
  return `${Math.round(share * 100)}%`;
}

/** Public Apple Music playlist share link: music.apple.com/<cc>/playlist/<name>/pl.<id>. */
export const APPLE_PLAYLIST_REGEX = /^(?:https?:\/\/)?music\.apple\.com\/[a-z]{2}\/playlist\/(?:[^/?#]+\/)?pl\.[A-Za-z0-9-]{8,64}(?:[/?#].*)?$/i;

export function isApplePlaylistUrl(input: string): boolean {
  return APPLE_PLAYLIST_REGEX.test(input.trim());
}

/** True for anything the backend can read: a Spotify link/URI/ID or an Apple Music playlist link. */
export function isPlaylistLink(input: string): boolean {
  return isApplePlaylistUrl(input) || parsePlaylistId(input) !== null;
}

export const PLAYLIST_REGEX = /(?:playlist[/:])?([A-Za-z0-9]{22})/;

/** Returns the 22-char playlist id or null. Rejects obviously non-playlist input. */
export function parsePlaylistId(input: string): string | null {
  const s = input.trim();
  if (!s) return null;
  if (/spotify/i.test(s) && !/playlist[/:]/i.test(s)) return null;
  const m = PLAYLIST_REGEX.exec(s);
  return m ? m[1] : null;
}
