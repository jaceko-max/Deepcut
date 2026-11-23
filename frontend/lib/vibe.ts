import type { Result, Vibe } from "./types";

/** `null` = all vibes. */
export type VibeSelection = number | null;

export function filterByVibe(results: Result[], vibe: VibeSelection): Result[] {
  return vibe === null ? results : results.filter((r) => r.vibe_id === vibe);
}

/** Parses `?vibe=<id>`; unknown or malformed ids mean "All". */
export function parseVibeParam(search: string, vibes: Vibe[]): VibeSelection {
  const raw = new URLSearchParams(search).get("vibe");
  if (raw === null || !/^\d+$/.test(raw)) return null;
  const id = Number(raw);
  return vibes.some((v) => v.id === id) ? id : null;
}

/** Returns the URL with `vibe` set or removed, leaving `job` and any `#fragment` untouched. */
export function withVibeParam(href: string, vibe: VibeSelection): string {
  const url = new URL(href);
  if (vibe === null) url.searchParams.delete("vibe");
  else url.searchParams.set("vibe", String(vibe));
  return url.toString();
}

export function countByVibe(results: Result[]): Map<number, number> {
  const m = new Map<number, number>();
  for (const r of results) m.set(r.vibe_id, (m.get(r.vibe_id) ?? 0) + 1);
  return m;
}
