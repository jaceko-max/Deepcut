import { describe, expect, it } from "vitest";
import { parseLine, parseTracks } from "./paste";

describe("parseTracks: Artist - Title", () => {
  it("reads hyphen, en dash and em dash separators and strips numbering", () => {
    const r = parseTracks("Low Hum Collective - Undertow\n2. Second Artist – Another One\n- Third — Song Name\n3) A - B");
    expect(r.format).toBe("lines");
    expect(r.tracks).toEqual([
      { artist: "Low Hum Collective", title: "Undertow" },
      { artist: "Second Artist", title: "Another One" },
      { artist: "Third", title: "Song Name" },
      { artist: "A", title: "B" },
    ]);
    expect(r.skipped).toBe(0);
  });

  it("splits at the first spaced dash, so hyphens inside names survive", () => {
    expect(parseLine("Jay-Z - 99 Problems - Remastered")).toEqual({ artist: "Jay-Z", title: "99 Problems - Remastered" });
  });
});

describe("parseTracks: Title by Artist", () => {
  it("uses the last ' by ' and prefers a dash when both occur", () => {
    expect(parseLine("Undertow by Low Hum Collective")).toEqual({ artist: "Low Hum Collective", title: "Undertow" });
    expect(parseLine("Stand by Me by Ben E. King")).toEqual({ artist: "Ben E. King", title: "Stand by Me" });
    expect(parseLine("Ben E. King - Stand by Me")).toEqual({ artist: "Ben E. King", title: "Stand by Me" });
  });
});

describe("parseTracks: Exportify CSV", () => {
  const csv =
    '"Track URI","Track Name","Album Name","Artist Name(s)","Release Date"\r\n' +
    '"spotify:track:1","Undertow","Night Shapes","Low Hum Collective","2020-01-01"\r\n' +
    '"spotify:track:2","Hello, World ""Live""","Album","Artist One;Artist Two","2021"\r\n' +
    '"spotify:track:3","","Album","Nobody","2021"\r\n';

  it("finds Track Name and Artist Name(s), handles quotes and commas, skips empty rows", () => {
    const r = parseTracks(csv);
    expect(r.format).toBe("csv");
    expect(r.tracks).toEqual([
      { artist: "Low Hum Collective", title: "Undertow" },
      { artist: "Artist One;Artist Two", title: 'Hello, World "Live"' },
    ]);
    expect(r.skipped).toBe(1);
  });

  it("accepts columns in any order and a BOM", () => {
    const r = parseTracks('﻿Artist Name(s),Track Name\nA,B\nC,D');
    expect(r.tracks).toEqual([{ artist: "A", title: "B" }, { artist: "C", title: "D" }]);
  });
});

describe("parseTracks: edge cases", () => {
  it("counts unreadable lines, de-duplicates, and caps at 200", () => {
    const r = parseTracks("just words\nA - B\na - b\n\n");
    expect(r.tracks).toHaveLength(1);
    expect(r.skipped).toBe(1);
    const many = Array.from({ length: 205 }, (_, i) => `Artist${i} - Song${i}`).join("\n");
    const big = parseTracks(many);
    expect(big.tracks).toHaveLength(200);
    expect(big.overflow).toBe(5);
  });

  it("returns empty for blank input and skips over-long fields", () => {
    expect(parseTracks("  \n ").format).toBe("empty");
    expect(parseTracks(`${"a".repeat(201)} - x`).skipped).toBe(1);
  });
});
