import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mockDone, mockRunning } from "./mock";
import { useJob } from "./useJob";
import type { JobResponse } from "./types";

function res(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

describe("useJob", () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://api.test");
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("polls queued -> running -> done and then stops", async () => {
    const queued: JobResponse = { ...mockRunning, status: "queued", stage: "queued", progress: 0 };
    fetchMock
      .mockResolvedValueOnce(res(queued))
      .mockResolvedValueOnce(res(mockRunning))
      .mockResolvedValue(res(mockDone));

    const { result } = renderHook(() => useJob("abc"));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(result.current.job?.status).toBe("queued");
    expect(result.current.phase).toBe("polling");

    await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
    expect(result.current.job?.status).toBe("running");

    await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
    expect(result.current.phase).toBe("done");
    expect(result.current.job?.results.length).toBeGreaterThan(0);

    const calls = fetchMock.mock.calls.length;
    expect(calls).toBe(3);
    await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
    expect(fetchMock.mock.calls.length).toBe(calls);
    expect(fetchMock.mock.calls[0][0]).toBe("http://api.test/api/jobs/abc");
  });

  it("slows to 3 s after 30 s", async () => {
    fetchMock.mockImplementation(async () => res(mockRunning));
    renderHook(() => useJob("abc"));
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
    const before = fetchMock.mock.calls.length; // ~21 calls at 1.5 s
    expect(before).toBe(21);
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(fetchMock.mock.calls.length).toBe(before + 1);
  });

  it("404 -> expired and stops", async () => {
    fetchMock.mockResolvedValue(res({ error: { code: "job_not_found", message: "nope" } }, 404));
    const { result } = renderHook(() => useJob("gone"));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(result.current.phase).toBe("expired");
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("error status stops with failed phase", async () => {
    const failed: JobResponse = {
      ...mockRunning,
      status: "error",
      stage: "error",
      error: { code: "no_previews", message: "x" },
    };
    fetchMock.mockResolvedValue(res(failed));
    const { result } = renderHook(() => useJob("e"));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(result.current.phase).toBe("failed");
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
