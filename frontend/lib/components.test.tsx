import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AnalyzeForm } from "@/components/AnalyzeForm";
import { PreviewProvider } from "@/components/PreviewPlayer";
import { ResultCard } from "@/components/ResultCard";
import { mockDone } from "./mock";

afterEach(cleanup);

describe("AnalyzeForm", () => {
  it("defaults to a 500K listener cap", () => {
    const onSubmit = vi.fn();
    render(<AnalyzeForm onSubmit={onSubmit} busy={false} initialUrl="https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M" />);
    expect(screen.getByText(/≤ 500K listeners/)).toBeTruthy();
    fireEvent.submit(screen.getByRole("button", { name: /start digging/i }).closest("form")!);
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ max_listeners: 500_000, limit: 20 }));
  });
});

describe("ResultCard", () => {
  const deep = mockDone.results.find((r) => r.source === "deep_cut")!;
  const other = mockDone.results.find((r) => r.source !== "deep_cut")!;

  it("badges deep cuts with accessible text and labels artist listeners", () => {
    render(
      <PreviewProvider>
        <ResultCard result={deep} showVibe={false} />
      </PreviewProvider>,
    );
    const badge = screen.getByTitle("A lesser-known track by an artist already in your playlist");
    expect(badge.textContent).toContain("Deep cut");
    expect(badge.textContent).toContain("a lesser-known track");
    expect(badge.textContent).toContain("from your playlist's artist");
    expect(screen.getByText(/artist listeners/)).toBeTruthy();
  });

  it("does not badge regular results", () => {
    render(
      <PreviewProvider>
        <ResultCard result={other} showVibe={false} />
      </PreviewProvider>,
    );
    expect(screen.queryByTitle("A lesser-known track by an artist already in your playlist")).toBeNull();
    expect(screen.queryByText(/artist listeners/)).toBeNull();
  });

  it("shows a subtle popularity-tier badge, and none for older payloads without a tier", () => {
    const under = mockDone.results.find((r) => r.tier === "underground")!;
    const { rerender } = render(
      <PreviewProvider>
        <ResultCard result={under} showVibe={false} />
      </PreviewProvider>,
    );
    expect(screen.getByTitle(/Underground artist/).textContent).toBe("Underground");
    rerender(
      <PreviewProvider>
        <ResultCard result={{ ...under, tier: undefined }} showVibe={false} />
      </PreviewProvider>,
    );
    expect(screen.queryByTitle(/Underground artist/)).toBeNull();
    expect(screen.queryByTitle(/Rising artist/)).toBeNull();
  });
});
