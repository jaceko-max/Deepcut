import type { FeatureSummary, JobResponse, PlaylistInfo, Result, Vibe } from "./types";

/**
 * Permanent share links: the finished results are packed into the URL fragment (`#r=<token>`), so the link renders
 * without the server. Fragments are never sent to a server. Token = "z." + base64url(deflate-raw(JSON)), or
 * "j." + base64url(JSON) when CompressionStream is unavailable.
 */

export const SHARE_PREFIX = "#r=";
const SHARE_VERSION = 1;

export interface ShareInput {
  playlist: PlaylistInfo | null;
  vibes: Vibe[];
  results: Result[];
  maxListeners: number;
}

const SOURCES: Result["source"][] = ["similar_artist", "similar_track", "tag", "deep_cut"];
const FEATURE_KEYS = [
  "pulse_clarity",
  "onset_density",
  "regularity",
  "syncopation",
  "brightness",
  "low_end",
  "energy",
  "dynamics",
] as const satisfies readonly (keyof FeatureSummary)[];

const ART_RE = /^https:\/\/e-cdns-images\.dzcdn\.net\/images\/cover\/([0-9a-f]{32})\/250x250-000000-80-0-0\.jpg$/;
const art = (h: string) => `https://e-cdns-images.dzcdn.net/images/cover/${h}/250x250-000000-80-0-0.jpg`;
const SIG_RE = /^\/api\/preview\/(\d+)\?sig=([0-9a-f]+)$/;

const q = (x: number) => Math.round(Math.min(1, Math.max(0, x)) * 100);
const packFeatures = (f: FeatureSummary) => FEATURE_KEYS.map((k) => q(f[k]));
const unpackFeatures = (bpm: number, a: number[]): FeatureSummary => {
  const f = { bpm } as FeatureSummary;
  FEATURE_KEYS.forEach((k, i) => (f[k] = (a[i] ?? 0) / 100));
  return f;
};

type Row = (string | number | null | number[])[];

interface Packed {
  v: number;
  p: [string, string | null, number, number, string, string | null] | null; // name, owner, analyzed, total, source, image
  m: number; // max listeners
  vb: [number, string, number, number, number, number[]][]; // id,label,share,bpm,track_count,features
  r: Row[];
}

export function pack(input: ShareInput): Packed {
  const p = input.playlist;
  return {
    v: SHARE_VERSION,
    p: p ? [p.name, p.owner, p.analyzed_tracks, p.total_tracks, p.source, p.image_url] : null,
    m: input.maxListeners,
    vb: input.vibes.map((v) => [v.id, v.label, Math.round(v.share * 1000) / 1000, v.bpm, v.track_count, packFeatures(v.features)]),
    r: input.results.map((r) => {
      const am = ART_RE.exec(r.artwork_url ?? "");
      const sm = SIG_RE.exec(r.preview_proxy_url);
      const row: Row = [
        r.deezer_id,
        r.title,
        r.artist,
        r.album,
        am ? am[1] : r.artwork_url,
        sm && Number(sm[1]) === r.deezer_id ? sm[2] : r.preview_proxy_url,
        r.listeners,
        r.match_score,
        r.bpm,
        r.vibe_id,
        r.why,
        SOURCES.indexOf(r.source),
        packFeatures(r.features),
      ];
      if (r.deezer_url !== `https://www.deezer.com/track/${r.deezer_id}`) row.push(r.deezer_url);
      return row;
    }),
  };
}

export function spotifySearchUrl(artist: string, title: string): string {
  return "https://open.spotify.com/search/" + encodeURIComponent(`${artist} ${title}`);
}

export function unpack(d: Packed): JobResponse {
  if (!d || d.v !== SHARE_VERSION || !Array.isArray(d.r) || !Array.isArray(d.vb)) throw new Error("Unsupported share link");
  const results: Result[] = d.r.map((row, i) => {
    const id = row[0] as number;
    const artwork = row[4] as string | null;
    const sig = row[5] as string;
    const bpm = row[8] as number;
    const artist = row[2] as string;
    const title = row[1] as string;
    return {
      rank: i + 1,
      deezer_id: id,
      title,
      artist,
      album: (row[3] as string | null) ?? null,
      artwork_url: artwork && /^[0-9a-f]{32}$/.test(artwork) ? art(artwork) : artwork,
      preview_url: "",
      preview_proxy_url: sig.startsWith("/") ? sig : `/api/preview/${id}?sig=${sig}`,
      deezer_url: (row[13] as string | undefined) ?? `https://www.deezer.com/track/${id}`,
      spotify_search_url: spotifySearchUrl(artist, title),
      listeners: row[6] as number,
      match_score: row[7] as number,
      bpm,
      vibe_id: row[9] as number,
      why: row[10] as string,
      source: SOURCES[row[11] as number] ?? "similar_artist",
      features: unpackFeatures(bpm, row[12] as number[]),
    };
  });
  const vibes: Vibe[] = d.vb.map(([id, label, share, bpm, track_count, f]) => ({
    id, label, share, bpm, track_count, features: unpackFeatures(bpm, f), example_tracks: [],
  }));
  const [name, owner, analyzed, total, source, image] = d.p ?? ["Shared playlist", null, 0, 0, "spotify_embed", null];
  const now = new Date(0).toISOString();
  return {
    job_id: "shared",
    status: "done",
    stage: "done",
    progress: 1,
    message: null,
    created_at: now,
    updated_at: now,
    params: { max_listeners: d.m, limit: results.length },
    playlist: { id: "shared", name, owner, image_url: image, total_tracks: total, analyzed_tracks: analyzed, source: source as PlaylistInfo["source"] },
    playlist_profile: {
      bpm_median: vibes[0]?.bpm ?? 0,
      bpm_range: [Math.min(...vibes.map((v) => v.bpm)), Math.max(...vibes.map((v) => v.bpm))],
      features: vibes[0]?.features ?? unpackFeatures(0, []),
      vibes,
    },
    results,
    reserve: [],
    stats: null,
    warnings: [],
    error: null,
  };
}

// ---------------------------------------------------------------- byte plumbing
export function toBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function fromBase64Url(s: string): Uint8Array {
  const b = atob(s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4));
  return Uint8Array.from(b, (c) => c.charCodeAt(0));
}

export function hasCompression(): boolean {
  return typeof CompressionStream !== "undefined" && typeof DecompressionStream !== "undefined";
}

async function pipe(bytes: Uint8Array, t: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const src = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(bytes);
      c.close();
    },
  });
  const reader = src.pipeThrough(t as unknown as ReadableWritablePair<Uint8Array, Uint8Array>).getReader();
  const chunks: Uint8Array[] = [];
  let len = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    len += value.length;
  }
  const out = new Uint8Array(len);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

/** Token for the `#r=` fragment. */
export async function encodeShare(input: ShareInput, opts: { compress?: boolean } = {}): Promise<string> {
  const json = new TextEncoder().encode(JSON.stringify(pack(input)));
  if (opts.compress !== false && hasCompression()) return "z." + toBase64Url(await pipe(json, new CompressionStream("deflate-raw")));
  return "j." + toBase64Url(json);
}

export async function decodeShare(token: string): Promise<JobResponse> {
  const kind = token.slice(0, 2);
  const body = token.slice(2);
  let json: Uint8Array;
  if (kind === "j.") json = fromBase64Url(body);
  else if (kind === "z.") {
    if (typeof DecompressionStream === "undefined") throw new Error("This browser can't open compressed share links.");
    json = await pipe(fromBase64Url(body), new DecompressionStream("deflate-raw"));
  } else throw new Error("Unrecognized share link");
  return unpack(JSON.parse(new TextDecoder().decode(json)) as Packed);
}

/** Returns the token if the hash is a share fragment (`#r=...`). */
export function shareTokenFromHash(hash: string): string | null {
  return hash.startsWith(SHARE_PREFIX) && hash.length > SHARE_PREFIX.length + 2 ? hash.slice(SHARE_PREFIX.length) : null;
}

export async function buildShareUrl(input: ShareInput, base: string): Promise<string> {
  return `${base.split("#")[0]}${SHARE_PREFIX}${await encodeShare(input)}`;
}
