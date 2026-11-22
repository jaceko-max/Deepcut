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
  it("running example has empty results", () => {
    expect(mockRunning.status).toBe("running");
    expect(mockRunning.results).toEqual([]);
  });
});
