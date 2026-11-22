import { describe, expect, it } from "vitest";
import { moreFromStart } from "./results";
import type { Result } from "./types";

const r = (artist: string) => ({ artist }) as Result;

describe("moreFromStart", () => {
  it("finds the trailing run of repeated artists", () => {
    expect(moreFromStart([r("A"), r("B"), r("C"), r("a"), r("B")])).toBe(3);
  });

  it("returns the length when every artist is distinct", () => {
    expect(moreFromStart([r("A"), r("B")])).toBe(2);
  });

  it("ignores a repeat that is followed by a new artist", () => {
    expect(moreFromStart([r("A"), r("A"), r("B")])).toBe(3);
  });
});
