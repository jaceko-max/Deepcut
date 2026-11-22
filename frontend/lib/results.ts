import type { Result } from "./types";

/**
 * The backend lists one track per artist first, then extra tracks from those same artists.
 * Returns the index where that trailing run of repeats starts, or the list length if there is none.
 */
export function moreFromStart(results: Result[]): number {
  const seen = new Set<string>();
  let start = results.length;
  results.forEach((r, i) => {
    const key = r.artist.toLowerCase();
    if (seen.has(key)) {
      if (start === results.length) start = i;
    } else {
      seen.add(key);
      start = results.length;
    }
  });
  return start;
}
