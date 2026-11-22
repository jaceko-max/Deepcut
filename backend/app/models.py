"""Pydantic models. Field names/types are a fixed contract (see PLAN.md section 2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Status = Literal["queued", "running", "done", "error"]
Stage = Literal[
    "queued", "fetching_playlist", "matching_previews", "analyzing_playlist",
    "finding_candidates", "analyzing_candidates", "ranking", "done", "error",
]
ErrorCode = Literal[
    "playlist_not_found", "playlist_unavailable", "playlist_empty", "no_previews",
    "no_candidates", "upstream_error", "internal_error",
]


class PastedTrack(BaseModel):
    artist: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)


class AnalyzeRequest(BaseModel):
    """Exactly one of playlist_url or tracks (a pasted track list) is required."""
    playlist_url: str | None = Field(None, max_length=512)
    tracks: list[PastedTrack] | None = Field(None, min_length=1, max_length=200)
    max_listeners: int = Field(500000, ge=10000, le=1000000)
    limit: int = Field(20, ge=5, le=50)

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if (self.playlist_url is None) == (self.tracks is None):
            raise ValueError("provide exactly one of playlist_url or tracks")
        return self


class AnalyzeResponse(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
    poll_url: str


class JobParams(BaseModel):
    max_listeners: int
    limit: int


class FeatureSummary(BaseModel):
    bpm: float
    pulse_clarity: float
    onset_density: float
    regularity: float
    syncopation: float
    brightness: float
    low_end: float
    energy: float
    dynamics: float


class TrackRef(BaseModel):
    title: str
    artist: str


class Vibe(BaseModel):
    id: int
    label: str
    share: float
    bpm: float
    track_count: int
    features: FeatureSummary
    example_tracks: list[TrackRef]


class PlaylistProfile(BaseModel):
    bpm_median: float
    bpm_range: tuple[float, float]
    features: FeatureSummary
    vibes: list[Vibe]


class PlaylistInfoOut(BaseModel):
    id: str
    name: str
    owner: str | None = None
    image_url: str | None = None
    total_tracks: int
    analyzed_tracks: int
    source: Literal["spotify_api", "spotify_embed", "pasted"]


class Result(BaseModel):
    rank: int
    deezer_id: int
    title: str
    artist: str
    album: str | None = None
    artwork_url: str | None = None
    preview_url: str
    preview_proxy_url: str
    deezer_url: str
    spotify_search_url: str
    listeners: int
    match_score: int = Field(ge=0, le=100)
    bpm: float
    vibe_id: int
    why: str
    source: Literal["similar_artist", "similar_track", "tag", "deep_cut"]
    features: FeatureSummary


class Stats(BaseModel):
    candidates_considered: int = 0
    candidates_analyzed: int = 0
    filtered_mainstream: int = 0
    filtered_in_playlist: int = 0
    no_preview: int = 0


class JobError(BaseModel):
    code: ErrorCode
    message: str


class JobResponse(BaseModel):
    job_id: str
    status: Status
    stage: Stage
    progress: float
    message: str | None = None
    created_at: str
    updated_at: str
    params: JobParams
    playlist: PlaylistInfoOut | None = None
    playlist_profile: PlaylistProfile | None = None
    results: list[Result] = Field(default_factory=list)
    reserve: list[Result] = Field(default_factory=list)  # next-best analyzed candidates, for client-side refine
    stats: Stats | None = None
    warnings: list[str] = Field(default_factory=list)
    error: JobError | None = None
