import { describe, expect, it } from "vitest";
import { mockDone, mockRunning } from "./mock";

describe("contract examples", () => {
  it("done example has expected shape", () => {
    expect(mockDone.status).toBe("done");
    expect(mockDone.results.length).toBeGreaterThan(0);
    expect(mockDone.stats?.filtered_mainstream).toBeTypeOf("number");
    const r = mockDone.results[0];
    for (const k of ["rank", "deezer_id", "title", "artist", "preview_url", "preview_proxy_url", "deezer_url", "spotify_search_url", "listeners", "match_score", "bpm", "vibe_id", "why", "source", "features"]) {
      expect(r).toHaveProperty(k);
    }
    expect(mockDone.playlist_profile?.vibes.length).toBeGreaterThanOrEqual(1);
  });
  it("done example includes a deep cut with a valid source", () => {
    const sources = new Set(["similar_artist", "similar_track", "tag", "deep_cut"]);
    for (const r of mockDone.results) expect(sources.has(r.source)).toBe(true);
    const deep = mockDone.results.filter((r) => r.source === "deep_cut");
    expect(deep.length).toBeGreaterThan(0);
    for (const r of deep) {
      expect(r.why.startsWith(`Deep cut from ${r.artist} · `)).toBe(true);
      expect(r.why.length).toBeLessThanOrEqual(90);
    }
    expect(mockDone.params.max_listeners).toBe(500_000);
  });
  it("running example has empty results", () => {
    expect(mockRunning.status).toBe("running");
    expect(mockRunning.results).toEqual([]);
  });
});
