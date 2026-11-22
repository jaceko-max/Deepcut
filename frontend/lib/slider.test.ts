import { describe, expect, it } from "vitest";
import { SLIDER_DEFAULT, listenersToSlider, sliderToListeners } from "./slider";

describe("slider log mapping", () => {
  it("maps endpoints", () => {
    expect(sliderToListeners(0)).toBe(10_000);
    expect(sliderToListeners(100)).toBe(1_000_000);
  });
  it("default is about 200K", () => {
    const v = sliderToListeners(SLIDER_DEFAULT);
    expect(v).toBeGreaterThan(190_000);
    expect(v).toBeLessThan(210_000);
  });
  it("round-trips 10k, 200k, 1M", () => {
    for (const v of [10_000, 200_000, 1_000_000]) {
      const t = listenersToSlider(v);
      const back = sliderToListeners(t);
      expect(Math.abs(back - v) / v).toBeLessThan(0.02);
    }
    expect(listenersToSlider(10_000)).toBe(0);
    expect(listenersToSlider(1_000_000)).toBe(100);
    expect(listenersToSlider(200_000)).toBe(65);
  });
  it("clamps", () => {
    expect(sliderToListeners(-5)).toBe(10_000);
    expect(sliderToListeners(500)).toBe(1_000_000);
  });
});
