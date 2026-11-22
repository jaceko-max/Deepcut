import { describe, expect, it } from "vitest";
import { formatBpm, formatCompact, parsePlaylistId } from "./format";

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
