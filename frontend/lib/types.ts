// Mirrors PLAN.md section 2 / contract/*.example.json exactly. Do not diverge.

export type JobStatus = "queued" | "running" | "done" | "error";

export type JobStage =
  | "queued"
  | "fetching_playlist"
  | "matching_previews"
  | "analyzing_playlist"
  | "finding_candidates"
  | "analyzing_candidates"
  | "ranking"
  | "done"
  | "error";

export type JobErrorCode =
  | "playlist_not_found"
  | "playlist_unavailable"
  | "playlist_empty"
  | "no_previews"
  | "no_candidates"
  | "upstream_error"
  | "internal_error";

/** Codes returned in the `{"error":{...}}` envelope of non-2xx HTTP responses. */
export type HttpErrorCode =
  | "invalid_playlist_url"
  | "invalid_request"
  | "too_many_jobs"
  | "server_busy"
  | "job_not_found"
  | "preview_not_found";

export interface FeatureSummary {
  bpm: number;
  pulse_clarity: number;
  onset_density: number;
  regularity: number;
  syncopation: number;
  brightness: number;
  low_end: number;
  energy: number;
  dynamics: number;
}

export interface ExampleTrack {
  title: string;
  artist: string;
}

export interface Vibe {
  id: number;
  label: string;
  share: number;
  bpm: number;
  track_count: number;
  features: FeatureSummary;
  example_tracks: ExampleTrack[];
}

export interface PlaylistInfo {
  id: string;
  name: string;
  owner: string | null;
  image_url: string | null;
  total_tracks: number;
  analyzed_tracks: number;
  source: "spotify_api" | "spotify_embed" | "pasted";
}

export interface PlaylistProfile {
  bpm_median: number;
  bpm_range: [number, number];
  features: FeatureSummary;
  vibes: Vibe[];
}

export interface Result {
  rank: number;
  deezer_id: number;
  title: string;
  artist: string;
  album: string | null;
  artwork_url: string | null;
  preview_url: string;
  preview_proxy_url: string;
  deezer_url: string;
  spotify_search_url: string;
  listeners: number;
  match_score: number;
  bpm: number;
  vibe_id: number;
  why: string;
  /** "deep_cut": a lesser-known track by an artist already in the playlist (listeners = that artist's). */
  source: "similar_artist" | "similar_track" | "tag" | "deep_cut";
  features: FeatureSummary;
}

export interface JobStats {
  candidates_considered: number;
  candidates_analyzed: number;
  filtered_mainstream: number;
  filtered_in_playlist: number;
  no_preview: number;
}

export interface JobError {
  code: JobErrorCode;
  message: string;
}

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  stage: JobStage;
  progress: number;
  message: string | null;
  created_at: string;
  updated_at: string;
  params: { max_listeners: number; limit: number };
  playlist: PlaylistInfo | null;
  playlist_profile: PlaylistProfile | null;
  results: Result[];
  /** Next-best analyzed candidates that did not make the list; used for client-side refine. Optional. */
  reserve?: Result[];
  stats: JobStats | null;
  warnings: string[];
  error: JobError | null;
}

export interface PastedTrack {
  artist: string;
  title: string;
}

/** Exactly one of playlist_url or tracks. */
export interface AnalyzeRequest {
  playlist_url?: string;
  tracks?: PastedTrack[];
  max_listeners: number;
  limit: number;
}

export interface AnalyzeAccepted {
  job_id: string;
  status: "queued";
  poll_url: string;
}

export interface HealthResponse {
  status: "ok";
  version: string;
}

export interface ErrorEnvelope {
  error: { code: string; message: string };
}
