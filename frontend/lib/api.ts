import { mockDone, mockRunning } from "./mock";
import type {
  AnalyzeAccepted,
  AnalyzeRequest,
  ErrorEnvelope,
  HealthResponse,
  JobResponse,
  JobStage,
} from "./types";

export function getApiUrl(): string {
  return (process.env.NEXT_PUBLIC_API_URL ?? "").trim().replace(/\/+$/, "");
}

export function isMockMode(): boolean {
  return getApiUrl() === "";
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryAfter: number | null;
  constructor(status: number, code: string, message: string, retryAfter: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.retryAfter = retryAfter;
  }
}

async function toApiError(res: Response): Promise<ApiError> {
  let code = "http_error";
  let message = `Request failed (${res.status})`;
  try {
    const body = (await res.json()) as Partial<ErrorEnvelope>;
    if (body?.error) {
      code = body.error.code ?? code;
      message = body.error.message ?? message;
    }
  } catch {
    /* non-JSON body */
  }
  const ra = res.headers.get("Retry-After");
  const retryAfter = ra && Number.isFinite(Number(ra)) ? Number(ra) : null;
  return new ApiError(res.status, code, message, retryAfter);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${getApiUrl()}${path}`, init);
  } catch {
    throw new ApiError(0, "network_error", "Could not reach the Deepcut server.");
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

/** Absolute audio URL for a result. Mock mode uses the raw preview_url. */
export function previewSrc(r: { preview_url: string; preview_proxy_url: string }): string {
  return isMockMode() && r.preview_url ? r.preview_url : `${getApiUrl()}${r.preview_proxy_url}`;
}

// ---------------------------------------------------------------- mock engine
// Stateless: job id encodes the start time, so a page refresh resumes the simulation.
const MOCK_PREFIX = "mock";
const MOCK_DURATION_MS = 14_000;
const MOCK_STAGES: { stage: JobStage; from: number; to: number; message: string }[] = [
  { stage: "queued", from: 0, to: 0.01, message: "Waiting for a free worker" },
  { stage: "fetching_playlist", from: 0.01, to: 0.05, message: "Reading playlist" },
  { stage: "matching_previews", from: 0.05, to: 0.15, message: "Matching previews" },
  { stage: "analyzing_playlist", from: 0.15, to: 0.45, message: "Analyzing track 24/40" },
  { stage: "finding_candidates", from: 0.45, to: 0.55, message: "Digging through similar artists" },
  { stage: "analyzing_candidates", from: 0.55, to: 0.95, message: "Analyzing candidate 54/118" },
  { stage: "ranking", from: 0.95, to: 1, message: "Ranking matches" },
];

function mockJob(id: string, req?: Partial<AnalyzeRequest>): JobResponse {
  const start = Number(id.slice(MOCK_PREFIX.length + 1));
  const elapsed = Date.now() - start;
  const t = Math.min(1, Math.max(0, elapsed / MOCK_DURATION_MS));
  const params = { ...mockDone.params, ...(req ? { max_listeners: req.max_listeners ?? mockDone.params.max_listeners, limit: req.limit ?? mockDone.params.limit } : {}) };
  if (t >= 1) return { ...mockDone, job_id: id, params };
  const cur = MOCK_STAGES.find((s) => t >= s.from && t < s.to) ?? MOCK_STAGES[MOCK_STAGES.length - 1];
  const within = (t - cur.from) / (cur.to - cur.from || 1);
  const showProfile = cur.stage === "finding_candidates" || cur.stage === "analyzing_candidates" || cur.stage === "ranking";
  const showPlaylist = t >= 0.05;
  return {
    ...mockRunning,
    job_id: id,
    params,
    status: cur.stage === "queued" ? "queued" : "running",
    stage: cur.stage,
    progress: Math.round((cur.from + (cur.to - cur.from) * within) * 100) / 100,
    message: cur.message,
    playlist: showPlaylist ? mockRunning.playlist : null,
    playlist_profile: showProfile ? mockRunning.playlist_profile : null,
  };
}

// ------------------------------------------------------------------- public API
export async function ping(): Promise<HealthResponse | null> {
  if (isMockMode()) return { status: "ok", version: "mock" };
  try {
    return await request<HealthResponse>("/api/health");
  } catch {
    return null;
  }
}

export async function createJob(req: AnalyzeRequest): Promise<AnalyzeAccepted> {
  if (isMockMode()) {
    await new Promise((r) => setTimeout(r, 400));
    const id = `${MOCK_PREFIX}-${Date.now()}`;
    return { job_id: id, status: "queued", poll_url: `/api/jobs/${id}` };
  }
  return request<AnalyzeAccepted>("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function getJob(jobId: string): Promise<JobResponse> {
  if (isMockMode()) {
    if (!new RegExp(`^${MOCK_PREFIX}-\\d+$`).test(jobId)) {
      throw new ApiError(404, "job_not_found", "Job not found or expired.");
    }
    return mockJob(jobId);
  }
  return request<JobResponse>(`/api/jobs/${encodeURIComponent(jobId)}`);
}
