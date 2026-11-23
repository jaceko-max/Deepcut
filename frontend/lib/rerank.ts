import type { FeatureSummary, Result } from "./types";

/** +1 = thumbs up, -1 = thumbs down, keyed by deezer_id. `order` is the order the votes were made (for undo). */
export interface Feedback {
  votes: Record<number, 1 | -1>;
  order: number[];
}

export const EMPTY_FEEDBACK: Feedback = { votes: {}, order: [] };

/** Weight of the liked / disliked similarity in the score (score is 0..100). */
export const ALPHA = 60;
export const BETA = 60;
const MAX_PER_ARTIST = 2;

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

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

/** BPM mapped from 60..180 to 0..1, then the eight 0..1 features. */
export function featureVector(f: FeatureSummary): number[] {
  return [clamp01((f.bpm - 60) / 120), ...FEATURE_KEYS.map((k) => clamp01(f[k]))];
}

function centroid(vs: number[][]): number[] {
  const out = new Array<number>(vs[0].length).fill(0);
  for (const v of vs) v.forEach((x, i) => (out[i] += x / vs.length));
  return out;
}

/** 1 for identical, 0 once the RMS difference per dimension reaches 0.3. */
function featureSim(a: number[], b: number[]): number {
  const d = Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0) / a.length);
  return Math.max(0, 1 - d / 0.3);
}

interface Profile {
  center: number[];
  artists: Set<string>;
  vibes: Map<number, number>;
  sources: Map<string, number>;
  n: number;
}

function profile(items: Result[]): Profile | null {
  if (items.length === 0) return null;
  const vibes = new Map<number, number>();
  const sources = new Map<string, number>();
  for (const r of items) {
    vibes.set(r.vibe_id, (vibes.get(r.vibe_id) ?? 0) + 1);
    sources.set(r.source, (sources.get(r.source) ?? 0) + 1);
  }
  return {
    center: centroid(items.map((r) => featureVector({ ...r.features, bpm: r.bpm }))),
    artists: new Set(items.map((r) => r.artist.toLowerCase())),
    vibes,
    sources,
    n: items.length,
  };
}

/** 0..1: 70% feature similarity to the centroid, 15% same artist, 10% same vibe share, 5% same source share. */
function affinity(r: Result, p: Profile | null): number {
  if (!p) return 0;
  return (
    0.7 * featureSim(featureVector({ ...r.features, bpm: r.bpm }), p.center) +
    0.15 * (p.artists.has(r.artist.toLowerCase()) ? 1 : 0) +
    0.1 * ((p.vibes.get(r.vibe_id) ?? 0) / p.n) +
    0.05 * ((p.sources.get(r.source) ?? 0) / p.n)
  );
}

export interface Refined {
  items: Result[];
  /** Reserve tracks that were pulled into the list. */
  pulledIn: Set<number>;
  /** Tracks removed because they were thumbed down (the disliked ones that were in the list). */
  removed: number;
  liked: number;
  disliked: number;
}

/**
 * Re-score list + reserve with `score + ALPHA*sim(liked) - BETA*sim(disliked)`, drop disliked tracks,
 * and keep the list at its original length by pulling in the best reserve tracks. Ranks are renumbered.
 * Without any feedback the list is returned unchanged.
 */
export function rerank(results: Result[], reserve: Result[], fb: Feedback): Refined {
  const votes = fb.votes;
  const pool = new Map<number, Result>();
  for (const r of [...results, ...reserve]) if (!pool.has(r.deezer_id)) pool.set(r.deezer_id, r);
  const liked = [...pool.values()].filter((r) => votes[r.deezer_id] === 1);
  const disliked = [...pool.values()].filter((r) => votes[r.deezer_id] === -1);
  const base = { pulledIn: new Set<number>(), removed: 0, liked: liked.length, disliked: disliked.length };
  if (liked.length === 0 && disliked.length === 0) return { ...base, items: results };

  const likeP = profile(liked);
  const dislikeP = profile(disliked);
  const inList = new Set(results.map((r) => r.deezer_id));
  const scored = [...pool.values()]
    .filter((r) => votes[r.deezer_id] !== -1)
    .map((r, i) => ({
      r,
      i,
      // a liked track keeps its place at the top of its own neighbourhood
      s: r.match_score + ALPHA * affinity(r, likeP) - BETA * affinity(r, dislikeP) + (votes[r.deezer_id] === 1 ? 5 : 0),
    }))
    .sort((a, b) => b.s - a.s || a.i - b.i);

  const perArtist = new Map<string, number>();
  const picked: Result[] = [];
  const rest: Result[] = [];
  for (const { r } of scored) {
    const a = r.artist.toLowerCase();
    if (picked.length < results.length && (perArtist.get(a) ?? 0) < MAX_PER_ARTIST) {
      perArtist.set(a, (perArtist.get(a) ?? 0) + 1);
      picked.push(r);
    } else rest.push(r);
  }
  for (const r of rest) if (picked.length < results.length) picked.push(r); // too few artists: allow extras

  const pulledIn = new Set(picked.filter((r) => !inList.has(r.deezer_id)).map((r) => r.deezer_id));
  const removed = results.filter((r) => votes[r.deezer_id] === -1).length;
  return { items: picked.map((r, i) => ({ ...r, rank: i + 1 })), pulledIn, removed, liked: liked.length, disliked: disliked.length };
}

export function vote(fb: Feedback, id: number, v: 1 | -1): Feedback {
  const votes = { ...fb.votes };
  const order = fb.order.filter((x) => x !== id);
  if (votes[id] === v) delete votes[id];
  else {
    votes[id] = v;
    order.push(id);
  }
  return { votes, order };
}

export function undoLast(fb: Feedback): Feedback {
  const id = fb.order[fb.order.length - 1];
  if (id === undefined) return fb;
  const votes = { ...fb.votes };
  delete votes[id];
  return { votes, order: fb.order.slice(0, -1) };
}

const key = (jobId: string) => `deepcut:feedback:${jobId}`;

export function loadFeedback(jobId: string): Feedback {
  try {
    const raw = window.localStorage.getItem(key(jobId));
    if (!raw) return EMPTY_FEEDBACK;
    const p = JSON.parse(raw) as Feedback;
    if (!p || typeof p.votes !== "object" || !Array.isArray(p.order)) return EMPTY_FEEDBACK;
    return p;
  } catch {
    return EMPTY_FEEDBACK;
  }
}

export function saveFeedback(jobId: string, fb: Feedback): void {
  try {
    if (fb.order.length === 0) window.localStorage.removeItem(key(jobId));
    else window.localStorage.setItem(key(jobId), JSON.stringify(fb));
  } catch {
    /* private mode or storage disabled: feedback just won't persist */
  }
}
