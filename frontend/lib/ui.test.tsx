import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AnalyzeForm } from "@/components/AnalyzeForm";
import { PreviewProvider } from "@/components/PreviewPlayer";
import { ResultList } from "@/components/ResultList";
import { VibeFilter } from "@/components/VibeFilter";
import { mockDone } from "./mock";

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  window.history.replaceState(null, "", "/");
});

describe("VibeFilter", () => {
  const vibes = mockDone.playlist_profile!.vibes;
  it("is a radiogroup with All first, shares, and arrow-key selection", () => {
    const onChange = vi.fn();
    render(<VibeFilter vibes={vibes} value={null} counts={new Map()} onChange={onChange} />);
    const radios = screen.getAllByRole("radio");
    expect(radios).toHaveLength(vibes.length + 1);
    expect(radios[0].textContent).toBe("All");
    expect(radios[0].getAttribute("aria-checked")).toBe("true");
    expect(radios[1].textContent).toContain("65%");
    fireEvent.keyDown(radios[0], { key: "ArrowRight" });
    expect(onChange).toHaveBeenCalledWith(vibes[0].id);
  });
});

describe("ResultList", () => {
  it("filters by vibe, writes ?vibe= and keeps the job param", () => {
    window.history.replaceState(null, "", "/?job=abc");
    render(<ResultList job={mockDone} />);
    const total = screen.getAllByRole("listitem").length;
    fireEvent.click(screen.getAllByRole("radio")[2]);
    expect(window.location.search).toBe("?job=abc&vibe=1");
    expect(screen.getAllByRole("listitem").length).toBeLessThan(total);
  });

  it("thumbs-down removes a track, shows the refine note, and Reset restores it", () => {
    render(
      <PreviewProvider>
        <ResultList job={mockDone} />
      </PreviewProvider>,
    );
    const first = mockDone.results[0];
    fireEvent.click(screen.getByRole("button", { name: `Less like ${first.title}` }));
    expect(screen.queryByRole("button", { name: `Less like ${first.title}` })).toBeNull();
    expect(screen.getByText(/Refined from your feedback/)).toBeTruthy();
    expect(window.localStorage.getItem(`deepcut:feedback:${mockDone.job_id}`)).toContain(String(first.deezer_id));
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.getByRole("button", { name: `Less like ${first.title}` })).toBeTruthy();
  });

  it("read-only view has no thumbs", () => {
    render(<ResultList job={mockDone} readOnly />);
    expect(screen.queryByRole("button", { name: /^Less like/ })).toBeNull();
  });
});

describe("AnalyzeForm paste tab", () => {
  it("counts lines, validates, and submits tracks without playlist_url", () => {
    const onSubmit = vi.fn();
    render(<AnalyzeForm onSubmit={onSubmit} busy={false} />);
    fireEvent.click(screen.getByRole("radio", { name: "Paste tracks" }));
    const box = screen.getByLabelText("Your tracks");
    fireEvent.change(box, { target: { value: "A - One\nB - Two" } });
    expect(screen.getByText(/2 of 200 tracks/)).toBeTruthy();
    fireEvent.submit(box.closest("form")!);
    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByRole("alert").textContent).toMatch(/at least 5/);
    fireEvent.change(box, { target: { value: "A - 1\nB - 2\nC - 3\nD - 4\nE - 5" } });
    fireEvent.submit(box.closest("form")!);
    const req = onSubmit.mock.calls[0][0];
    expect(req.playlist_url).toBeUndefined();
    expect(req.tracks).toHaveLength(5);
  });
});
