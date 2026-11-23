import { describe, expect, it } from "vitest";
import { formatBpm, formatCompact, isApplePlaylistUrl, isPlaylistLink, parsePlaylistId, tierBadge } from "./format";

describe("tierBadge", () => {
  it("labels rising and underground picks, nothing otherwise", () => {
    expect(tierBadge("rising")?.label).toBe("Rising");
    expect(tierBadge("underground")?.label).toBe("Underground");
    expect(tierBadge("deep_cut")).toBeNull(); // deep cuts keep their own badge
    expect(tierBadge(undefined)).toBeNull(); // older payloads have no tier
    expect(tierBadge(null)).toBeNull();
  });
});

describe("formatCompact", () => {
  it("formats", () => {
    expect(formatCompact(950)).toBe("950");
    expect(formatCompact(1200)).toBe("1.2K");
    expect(formatCompact(12_000)).toBe("12K");
    expect(formatCompact(48_210)).toBe("48K");
    expect(formatCompact(199_526)).toBe("200K");
    expect(formatCompact(1_000_000)).toBe("1M");
    expect(formatCompact(1_234_567)).toBe("1.2M");
    expect(formatCompact(999_999)).toBe("1M");
  });
});

describe("formatBpm", () => {
  it("trims", () => {
    expect(formatBpm(92)).toBe("92");
    expect(formatBpm(92.3)).toBe("92.3");
  });
});

describe("parsePlaylistId", () => {
  const id = "37i9dQZF1DXcBWIGoYBM5M";
  it("accepts url, uri, bare id", () => {
    expect(parsePlaylistId(`https://open.spotify.com/playlist/${id}?si=abc`)).toBe(id);
    expect(parsePlaylistId(`https://open.spotify.com/intl-de/playlist/${id}`)).toBe(id);
    expect(parsePlaylistId(`spotify:playlist:${id}`)).toBe(id);
    expect(parsePlaylistId(id)).toBe(id);
  });
  it("rejects junk and non-playlist spotify links", () => {
    expect(parsePlaylistId("")).toBeNull();
    expect(parsePlaylistId("hello")).toBeNull();
    expect(parsePlaylistId(`https://open.spotify.com/album/${id}`)).toBeNull();
  });
});

describe("Apple Music links", () => {
  const apple = "https://music.apple.com/us/playlist/todays-hits/pl.f4d106fed2bd41149aaacabb233eb5eb";
  it("accepts public playlist share links", () => {
    expect(isApplePlaylistUrl(apple)).toBe(true);
    expect(isApplePlaylistUrl(`${apple}?l=en-US`)).toBe(true);
    expect(isApplePlaylistUrl("music.apple.com/gb/playlist/pl.u-AbCd1234-xYz")).toBe(true);
    expect(isPlaylistLink(apple)).toBe(true);
  });
  it("rejects other Apple Music links and look-alike hosts", () => {
    expect(isApplePlaylistUrl("https://music.apple.com/us/album/x/12345")).toBe(false);
    expect(isApplePlaylistUrl("https://music.apple.com.evil.example/us/playlist/x/pl.f4d106fed2bd41149aaacabb233eb5eb")).toBe(false);
    expect(isPlaylistLink("https://example.com/hello")).toBe(false);
  });
  it("still accepts Spotify links", () => {
    expect(isPlaylistLink("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M")).toBe(true);
  });
});
