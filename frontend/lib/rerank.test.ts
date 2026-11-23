import { describe, expect, it } from "vitest";
import { EMPTY_FEEDBACK, loadFeedback, rerank, saveFeedback, undoLast, vote } from "./rerank";
import type { FeatureSummary, Result } from "./types";

const feat = (over: Partial<FeatureSummary> = {}): FeatureSummary => ({
  bpm: 92, pulse_clarity: 0.6, onset_density: 0.4, regularity: 0.8, syncopation: 0.3, brightness: 0.3, low_end: 0.7, energy: 0.6, dynamics: 0.4, ...over,
});
// "dark" = like the first mock vibe, "bright" = fast, bright, busy
const DARK = feat();
const BRIGHT = feat({ bpm: 126, onset_density: 0.8, brightness: 0.8, low_end: 0.2, syncopation: 0.6 });

let n = 0;
const mk = (artist: string, score: number, f: FeatureSummary, over: Partial<Result> = {}): Result => ({
  rank: 0, deezer_id: ++n, title: `T${n}`, artist, album: null, artwork_url: null, preview_url: "", preview_proxy_url: "",
  deezer_url: "", spotify_search_url: "", listeners: 1000, match_score: score, bpm: f.bpm, vibe_id: f.bpm > 100 ? 1 : 0,
  why: "", source: "similar_artist", features: f, ...over,
});

const list = [mk("A", 90, DARK), mk("B", 88, BRIGHT), mk("C", 86, DARK), mk("D", 84, BRIGHT)].map((r, i) => ({ ...r, rank: i + 1 }));
const reserve = [mk("E", 80, DARK, { rank: 5 }), mk("F", 79, BRIGHT, { rank: 6 }), mk("G", 70, DARK, { rank: 7 })];
const ids = (rs: Result[]) => rs.map((r) => r.artist);

describe("rerank", () => {
  it("returns the list unchanged without feedback", () => {
    const r = rerank(list, reserve, EMPTY_FEEDBACK);
    expect(r.items).toBe(list);
    expect(r.pulledIn.size).toBe(0);
  });

  it("removes disliked tracks and pulls in reserve tracks to keep the length", () => {
    const fb = vote(EMPTY_FEEDBACK, list[0].deezer_id, -1);
    const r = rerank(list, reserve, fb);
    expect(r.items).toHaveLength(4);
    expect(ids(r.items)).not.toContain("A");
    expect(r.removed).toBe(1);
    expect(r.pulledIn.size).toBe(1);
    expect(r.items.map((x) => x.rank)).toEqual([1, 2, 3, 4]);
  });

  it("a thumbs-up pulls similar tracks up and dissimilar ones down", () => {
    const fb = vote(EMPTY_FEEDBACK, list[3].deezer_id, 1); // like the bright track D
    const r = rerank(list, reserve, fb);
    expect(ids(r.items)[0]).toBe("D"); // liked track rises
    // bright B overtakes dark C even though C scored higher originally (C may even be pushed out by bright reserve F)
    const c = ids(r.items).indexOf("C");
    expect(ids(r.items).indexOf("B")).toBeLessThan(c === -1 ? 99 : c);
    expect(ids(r.items)).toContain("F");
  });

  it("dislike of a dark track demotes the other dark tracks relative to bright ones", () => {
    const fb = vote(EMPTY_FEEDBACK, list[0].deezer_id, -1);
    const r = rerank(list, reserve, fb);
    expect(ids(r.items).indexOf("B")).toBeLessThan(ids(r.items).indexOf("C"));
    expect(ids(r.items)).toContain("F"); // bright reserve track is pulled in ahead of dark E
    expect(ids(r.items)).not.toContain("E");
  });

  it("is stable: same feedback gives the same order, and undo restores the original", () => {
    const fb = vote(vote(EMPTY_FEEDBACK, list[3].deezer_id, 1), list[0].deezer_id, -1);
    expect(ids(rerank(list, reserve, fb).items)).toEqual(ids(rerank(list, reserve, fb).items));
    expect(rerank(list, reserve, undoLast(undoLast(fb))).items).toBe(list);
  });

  it("limits tracks per artist to two", () => {
    const many = [mk("X", 95, DARK), mk("X", 94, DARK), mk("X", 93, DARK), mk("Y", 60, BRIGHT)];
    const r = rerank(many.slice(0, 3), [many[3]], vote(EMPTY_FEEDBACK, many[0].deezer_id, 1));
    expect(ids(r.items).filter((a) => a === "X")).toHaveLength(2);
    expect(ids(r.items)).toContain("Y");
  });
});

describe("vote / undo / persistence", () => {
  it("toggles a vote off when clicked twice and switches direction", () => {
    let fb = vote(EMPTY_FEEDBACK, 1, 1);
    expect(fb.votes[1]).toBe(1);
    fb = vote(fb, 1, -1);
    expect(fb.votes[1]).toBe(-1);
    fb = vote(fb, 1, -1);
    expect(fb.votes[1]).toBeUndefined();
    expect(fb.order).toEqual([]);
  });

  it("saves per job id and survives broken storage", () => {
    const fb = vote(EMPTY_FEEDBACK, 7, 1);
    saveFeedback("job-a", fb);
    expect(loadFeedback("job-a")).toEqual(fb);
    expect(loadFeedback("job-b")).toEqual(EMPTY_FEEDBACK);
    window.localStorage.setItem("deepcut:feedback:bad", "{nope");
    expect(loadFeedback("bad")).toEqual(EMPTY_FEEDBACK);
    saveFeedback("job-a", EMPTY_FEEDBACK);
    expect(window.localStorage.getItem("deepcut:feedback:job-a")).toBeNull();
  });
});
