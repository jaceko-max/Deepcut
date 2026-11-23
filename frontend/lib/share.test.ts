import { describe, expect, it } from "vitest";
import { toCsv, toPlainText } from "./exporters";
import { decodeShare, encodeShare, fromBase64Url, shareTokenFromHash, toBase64Url } from "./share";
import { mockDone } from "./mock";
import { filterByVibe, parseVibeParam, withVibeParam } from "./vibe";

const input = () => ({
  playlist: mockDone.playlist,
  vibes: mockDone.playlist_profile!.vibes,
  results: mockDone.results,
  maxListeners: mockDone.params.max_listeners,
});

function expectSameRender(got: ReturnType<typeof Object>, want = mockDone) {
  const j = got as typeof mockDone;
  expect(j.results).toHaveLength(want.results.length);
  j.results.forEach((r, i) => {
    const w = want.results[i];
    expect(r).toMatchObject({
      rank: i + 1, deezer_id: w.deezer_id, title: w.title, artist: w.artist, album: w.album, artwork_url: w.artwork_url,
      preview_proxy_url: w.preview_proxy_url, deezer_url: w.deezer_url, spotify_search_url: w.spotify_search_url,
      listeners: w.listeners, match_score: w.match_score, bpm: w.bpm, vibe_id: w.vibe_id, why: w.why, source: w.source,
    });
    for (const k of Object.keys(w.features) as (keyof typeof w.features)[]) expect(r.features[k]).toBeCloseTo(w.features[k], 2);
  });
  expect(j.playlist_profile!.vibes.map((v) => [v.id, v.label, v.share, v.bpm, v.track_count])).toEqual(
    want.playlist_profile!.vibes.map((v) => [v.id, v.label, v.share, v.bpm, v.track_count]),
  );
  expect(j.playlist!.name).toBe(want.playlist!.name);
  expect(j.params.max_listeners).toBe(want.params.max_listeners);
  expect(j.reserve).toEqual([]);
}

describe("share link", () => {
  it("round-trips compressed (deflate-raw + base64url)", async () => {
    const token = await encodeShare(input());
    expect(token.startsWith("z.")).toBe(true);
    expect(token).toMatch(/^z\.[A-Za-z0-9_-]+$/);
    expectSameRender(await decodeShare(token));
  });

  it("round-trips with the uncompressed fallback", async () => {
    const token = await encodeShare(input(), { compress: false });
    expect(token.startsWith("j.")).toBe(true);
    expectSameRender(await decodeShare(token));
  });

  it("does not carry the reserve or the expiring preview_url", async () => {
    const token = await encodeShare({ ...input() });
    const json = JSON.stringify(await decodeShare(token));
    expect(json).not.toContain("cdnt-preview");
    expect(token.length).toBeLessThan(2000);
  });

  it("handles unicode text", async () => {
    const results = [{ ...mockDone.results[0], title: "Ünderstrøm 夜 🎧", artist: "Ödeon & Sons" }];
    const out = await decodeShare(await encodeShare({ ...input(), results }));
    expect(out.results[0].title).toBe("Ünderstrøm 夜 🎧");
    expect(out.results[0].spotify_search_url).toBe("https://open.spotify.com/search/" + encodeURIComponent("Ödeon & Sons Ünderstrøm 夜 🎧"));
  });

  it("rejects garbage", async () => {
    await expect(decodeShare("x.abc")).rejects.toThrow();
    await expect(decodeShare("z.AAAA")).rejects.toThrow();
  });

  it("recognizes the hash and base64url is lossless", () => {
    expect(shareTokenFromHash("#r=z.abc")).toBe("z.abc");
    expect(shareTokenFromHash("#other")).toBeNull();
    const bytes = Uint8Array.from({ length: 300 }, (_, i) => (i * 37) % 256);
    expect(Array.from(fromBase64Url(toBase64Url(bytes)))).toEqual(Array.from(bytes));
  });
});

describe("export", () => {
  it("plain text is '1. Artist – Title'", () => {
    const lines = toPlainText(mockDone.results).split("\n");
    expect(lines[0]).toBe(`1. ${mockDone.results[0].artist} – ${mockDone.results[0].title}`);
    expect(lines).toHaveLength(mockDone.results.length);
  });

  it("csv quotes commas and neutralizes formulas", () => {
    const r = [{ ...mockDone.results[0], artist: "A, B", title: "=SUM(1)" }];
    const csv = toCsv(r).split("\r\n");
    expect(csv[0].startsWith("Rank,Artist,Title")).toBe(true);
    expect(csv[1].startsWith('1,"A, B",\'=SUM(1),')).toBe(true);
  });
});

describe("vibe filter", () => {
  const vibes = mockDone.playlist_profile!.vibes;
  it("filters by vibe id and treats null as all", () => {
    expect(filterByVibe(mockDone.results, null)).toBe(mockDone.results);
    const only = filterByVibe(mockDone.results, 1);
    expect(only.length).toBeGreaterThan(0);
    expect(only.every((r) => r.vibe_id === 1)).toBe(true);
  });
  it("reads and writes ?vibe= next to ?job= and keeps the hash", () => {
    expect(parseVibeParam("?job=abc&vibe=1", vibes)).toBe(1);
    expect(parseVibeParam("?vibe=9", vibes)).toBeNull();
    expect(parseVibeParam("?vibe=x", vibes)).toBeNull();
    expect(withVibeParam("http://h/?job=abc#r=z.q", 1)).toBe("http://h/?job=abc&vibe=1#r=z.q");
    expect(withVibeParam("http://h/?job=abc&vibe=1", null)).toBe("http://h/?job=abc");
  });
});
