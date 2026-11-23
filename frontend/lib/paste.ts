import type { PastedTrack } from "./types";

export const MAX_PASTED_TRACKS = 200;
export const MIN_PASTED_TRACKS = 5;
const MAX_ARTIST = 200;
const MAX_TITLE = 300;

export interface ParsedTracks {
  tracks: PastedTrack[];
  /** Non-empty lines that could not be read as "Artist - Title", "Title by Artist" or a CSV row. */
  skipped: number;
  /** Tracks beyond the first 200 that were left out. */
  overflow: number;
  format: "csv" | "lines" | "empty";
}

/** Minimal RFC 4180 parser: quoted fields, doubled quotes, commas and newlines inside quotes. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else quoted = false;
      } else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      field = "";
      if (row.some((f) => f.trim() !== "")) rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.some((f) => f.trim() !== "")) rows.push(row);
  return rows;
}

const norm = (s: string) => s.toLowerCase().replace(/[^a-z]/g, "");

/** Header names used by Exportify ("Track Name", "Artist Name(s)") and similar exporters. */
function csvColumns(header: string[]): { title: number; artist: number } | null {
  const h = header.map(norm);
  const title = h.findIndex((x) => x === "trackname" || x === "title" || x === "track" || x === "songname" || x === "name");
  const artist = h.findIndex((x) => x === "artistnames" || x === "artistname" || x === "artist" || x === "artists");
  return title >= 0 && artist >= 0 ? { title, artist } : null;
}

const clean = (s: string) => s.replace(/\s+/g, " ").trim();

/** "Artist - Title" (hyphen, en dash or em dash, spaced) first, then "Title by Artist". */
export function parseLine(raw: string): PastedTrack | null {
  const line = clean(raw.replace(/^\s*(?:\d{1,3}[.)]|[-*•])\s+/, ""));
  if (!line) return null;
  const dash = /\s[-–—]\s/.exec(line);
  if (dash) {
    const artist = clean(line.slice(0, dash.index));
    const title = clean(line.slice(dash.index + dash[0].length));
    return artist && title ? { artist, title } : null;
  }
  const idx = line.toLowerCase().lastIndexOf(" by ");
  if (idx > 0) {
    const title = clean(line.slice(0, idx));
    const artist = clean(line.slice(idx + 4));
    return artist && title ? { artist, title } : null;
  }
  return null;
}

export function parseTracks(text: string): ParsedTracks {
  const input = text.replace(/^﻿/, "");
  if (!input.trim()) return { tracks: [], skipped: 0, overflow: 0, format: "empty" };

  const found: (PastedTrack | null)[] = [];
  let format: ParsedTracks["format"] = "lines";
  const firstLine = input.split(/\r?\n/).find((l) => l.trim()) ?? "";
  const cols = firstLine.includes(",") ? csvColumns(parseCsv(firstLine)[0] ?? []) : null;
  if (cols) {
    format = "csv";
    for (const r of parseCsv(input).slice(1)) {
      const title = clean(r[cols.title] ?? "");
      const artist = clean(r[cols.artist] ?? "");
      found.push(title && artist ? { artist, title } : null);
    }
  } else {
    for (const l of input.split(/\r?\n/)) if (l.trim()) found.push(parseLine(l));
  }

  const seen = new Set<string>();
  const tracks: PastedTrack[] = [];
  let skipped = 0;
  let overflow = 0;
  for (const t of found) {
    if (!t || t.artist.length > MAX_ARTIST || t.title.length > MAX_TITLE) {
      skipped++;
      continue;
    }
    const key = `${t.artist}\u0000${t.title}`.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    if (tracks.length >= MAX_PASTED_TRACKS) overflow++;
    else tracks.push(t);
  }
  return { tracks, skipped, overflow, format };
}
