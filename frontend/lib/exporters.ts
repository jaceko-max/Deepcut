import type { Result } from "./types";

/** "1. Artist – Title", one per line. */
export function toPlainText(results: Result[]): string {
  return results.map((r, i) => `${i + 1}. ${r.artist} – ${r.title}`).join("\n");
}

function cell(v: string | number | null): string {
  const s = v === null ? "" : String(v);
  // quote fields that need it; neutralize spreadsheet formula injection
  const safe = /^[=+\-@\t\r]/.test(s) ? `'${s}` : s;
  return /[",\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
}

export function toCsv(results: Result[]): string {
  const head = ["Rank", "Artist", "Title", "Album", "Match", "Listeners", "BPM", "Deezer", "Spotify search"];
  const rows = results.map((r, i) =>
    [i + 1, r.artist, r.title, r.album, r.match_score, r.listeners, r.bpm, r.deezer_url, r.spotify_search_url].map(cell).join(","),
  );
  return [head.join(","), ...rows].join("\r\n") + "\r\n";
}

export function downloadFile(name: string, text: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: `${mime};charset=utf-8` }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}
