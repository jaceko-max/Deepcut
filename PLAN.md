# Deepcut — implementation plan

Deepcut takes a public Spotify playlist link and finds similar but non-mainstream music by matching "beat flow" (tempo, groove, drum density, tone, timbre) computed from 30-second audio previews. A Last.fm listener-count cap filters out mainstream/viral artists.

Monorepo: `backend/` (Python FastAPI, Docker, Render) and `frontend/` (Next.js App Router + TypeScript + Tailwind v4, Vercel). `contract/*.example.json` is the shared source of truth for the API shape.

## 0. Verified facts (2026-11) that override earlier assumptions

1. **Spotify Web API cannot read tracks of public playlists you don't own** (Feb 2026 dev-mode change: `GET /playlists/{id}/items` returns metadata only / no `items` for non-owned playlists; `popularity`/`followers` removed). Therefore:
   - **Primary playlist source = the public embed page** `https://open.spotify.com/embed/playlist/{id}` (verified live, no auth). Parse `<script id="__NEXT_DATA__">` JSON → `props.pageProps.state.data.entity`. Fields: `name`, `subtitle` (owner), `coverArt`, `trackList[]` each with `uri` (`spotify:track:ID`), `title`, `subtitle` (artists joined by `, `), `duration` (ms), `audioPreview.url` (Spotify 30s MP3, downloadable, no auth) — may be absent. A real copy is saved at `backend/tests/fixtures/spotify_embed.html` (50 tracks, all with previews). The embed may cap at ~50–100 tracks; that's fine.
   - `SPOTIFY_PLAYLIST_SOURCE=auto|api|embed`, default `auto`: if Spotify credentials are set, try the API (Client Credentials) first and fall back to embed when `items` is missing or on 401/403/404; if no credentials, use embed directly. **Spotify credentials are optional** — the app must work with only `LASTFM_API_KEY`.
   - Playlist-track audio: use the embed's `audioPreview.url` directly when present (no Deezer lookup needed); fall back to Deezer matching only when absent. Cache key for those features: `sp:{spotify_track_id}:v{FEATURE_VERSION}`.
   - README must note embed scraping is unofficial and may break.
2. **Deezer field-filter search (`q=artist:"x" track:"y"`) currently returns empty results.** Use plain keyword search `GET https://api.deezer.com/search?q={artist} {title}` and verify with difflib ratio ≥ 0.8 on normalized artist+title. Search results lack `bpm`; that's fine (we compute our own). Bad ISRC lookups return HTTP 200 with `"readable":false` and `"preview":""` (or `{"error":{"code":800}}`) — check content, not status. Quota errors arrive as HTTP 200 `{"error":{"code":4,...}}` → treat as rate limit and retry. Preview URLs are signed (`hdnea=exp=...`, ~15 min) → `/api/preview/{deezer_id}` endpoint redirects to a fresh URL.
3. **Toolchain:** local Python is 3.14 (no ffmpeg, no Docker). Do NOT use librosa. Use numpy + scipy for DSP and `miniaudio` for MP3 decode/resample (wheels exist for 3.14); ffmpeg subprocess only as fallback (Docker image installs ffmpeg). Use plain `uvicorn` (no `[standard]`).

## 1. Directory tree

```
deepcut/
  README.md  render.yaml  .gitignore  PLAN.md
  contract/ job_done.example.json  job_running.example.json
  backend/
    Dockerfile .dockerignore .env.example requirements.txt requirements-dev.txt pyproject.toml
    app/
      main.py config.py models.py jobs.py pipeline.py cache.py ratelimit.py
      clients/ http.py spotify.py lastfm.py deezer.py
      audio/ decode.py dsp.py features.py worker.py
      scoring/ normalize.py vibes.py score.py explain.py
    scripts/ smoke_live.py
    tests/ conftest.py synth.py fixtures/ test_*.py
  frontend/
    package.json next.config.ts tsconfig.json postcss.config.mjs vitest.config.ts .env.example
    app/ layout.tsx page.tsx globals.css
    components/ AnalyzeForm ListenerSlider JobProgress PlaylistProfile VibeCard FeatureBars ResultList ResultCard PreviewPlayer ErrorState Footer
    lib/ types.ts api.ts useJob.ts slider.ts format.ts mock.ts *.test.ts
```

## 2. API contract

Base URL `NEXT_PUBLIC_API_URL`. Errors always `{"error":{"code":string,"message":string}}`; FastAPI 422 remapped to 400 `invalid_request`.

- `GET /api/health` → `{"status":"ok","version":"0.1.0"}` (frontend pings on load to wake cold server).
- `POST /api/analyze` body `{"playlist_url": string, "max_listeners": int 10000..1000000 = 200000, "limit": int 5..50 = 20}`.
  `playlist_url` accepts `https://open.spotify.com[/intl-xx]/playlist/{id}?si=..`, `spotify:playlist:{id}`, bare 22-char id; regex `(?:playlist[/:])?([A-Za-z0-9]{22})`.
  → 202 `{"job_id","status":"queued","poll_url":"/api/jobs/{job_id}"}`; 400 `invalid_playlist_url`|`invalid_request`; 429 `too_many_jobs` (3 per IP per 10 min, `Retry-After`); 503 `server_busy` (>10 queued).
- `GET /api/jobs/{job_id}` → 200 JobResponse; 404 `job_not_found` (TTL 1h).
- `GET /api/preview/{deezer_id}` → 307 to fresh Deezer preview; 404 `preview_not_found`.

```
JobResponse {
  job_id: string,
  status: "queued"|"running"|"done"|"error",
  stage: "queued"|"fetching_playlist"|"matching_previews"|"analyzing_playlist"|"finding_candidates"|"analyzing_candidates"|"ranking"|"done"|"error",
  progress: number (0..1, monotonic), message: string|null,
  created_at: string, updated_at: string (ISO-8601 UTC),
  params: {max_listeners: int, limit: int},
  playlist: null | {id, name, owner: string|null, image_url: string|null, total_tracks: int, analyzed_tracks: int, source: "spotify_api"|"spotify_embed"},
  playlist_profile: null | {bpm_median: number, bpm_range: [number, number], features: FeatureSummary,
     vibes: [{id: int, label: string, share: number, bpm: number, track_count: int, features: FeatureSummary, example_tracks: [{title, artist}]}] (1..3)},
  results: Result[] ([] until done),
  stats: null | {candidates_considered, candidates_analyzed, filtered_mainstream, filtered_in_playlist, no_preview: int},
  warnings: string[],
  error: null | {code: "playlist_not_found"|"playlist_unavailable"|"playlist_empty"|"no_previews"|"no_candidates"|"upstream_error"|"internal_error", message: string}
}
FeatureSummary {bpm: number, pulse_clarity, onset_density, regularity, syncopation, brightness, low_end, energy, dynamics: number 0..1}
Result {rank: int, deezer_id: int, title, artist, album: string|null, artwork_url: string|null,
  preview_url: string, preview_proxy_url: string ("/api/preview/{deezer_id}"), deezer_url: string,
  spotify_search_url: string ("https://open.spotify.com/search/" + encodeURIComponent(artist+" "+title)),
  listeners: int, match_score: int 0..100, bpm: number (1 decimal), vibe_id: int, why: string (<=90 chars),
  source: "similar_artist"|"similar_track"|"tag", features: FeatureSummary}
```

## 3. Feature extraction (numpy/scipy only, `audio/features.py`, FEATURE_VERSION=1)

1. Decode with `miniaudio.decode(bytes, output_format=SampleFormat.FLOAT32, nchannels=1, sample_rate=22050)`; fallback `ffmpeg -v error -i pipe:0 -f f32le -ac 1 -ar 22050 pipe:1`. Reject < 10 s. Use ≤30 s, skip first 0.25 s, remove DC.
2. Loudness on raw signal: RMS frames 2048/hop 512 (43.07 fps), `rms_db=20log10(rms+1e-9)`; `rms_mean_db` = mean of frames above p10; `dynamic_range_db = p95 − p10`. Then peak-normalize to 0.99.
3. STFT Hann n_fft 2048 hop 512, power P.
4. 64 HTK mel bands 30–11025 Hz, triangular area-normalized; `logmel = 10log10(mel+1e-10)`.
5. Onset envelope: spectral flux on logmel (positive time diff, mean over bands), subtract 0.5 s moving average, clip ≥0, divide by max.
6. Tempo: FFT autocorrelation of envelope normalized by ac[0]; lags for 40–220 BPM; score `ac[l] + 0.5ac[2l] + 0.5ac[l/2]` × prior `exp(−0.5((log2 bpm − log2 120)/1.0)^2)`; argmax + parabolic refine → `bpm_raw`; fold ×/÷2 into [70,140) → `bpm_folded`. Tempo distance is octave-circular: `c=|frac(log2(a/b))|; c=min(c,1−c)` (60 vs 120 → 0).
7. `pulse_clarity = ac[best_lag]`.
8. Beat tracking: Ellis DP (tightness 100). `regularity = clip(1 − CV(IBI)/0.25, 0, 1)`. `pulse_stability` = mean over 4 s windows (2 s hop) of `ac_w[best_lag]/ac_w[0]`.
9. Onsets: local max in ±3 frames, > mean + 0.5·std, ≥3 frames apart. `onset_density` = onsets/sec.
10. `syncopation`: onset phase within beat quantized to 16th slot 0..3, weights {0:0, 2:0.5, 1:1, 3:1}; Σ(strength·w)/Σ strength.
11. Spectral (mean over non-silent frames): log2 centroid, log2 rolloff85, `bass_ratio` ΣP(<150Hz)/ΣP, `sub_ratio` ΣP(<60Hz)/ΣP, `flatness` (geo/arith mean).
12. Timbre: MFCC = orthonormal DCT-II of logmel, coeffs 1..12; mean+std = 24 values.
13. Output RawFeatures (scalars + 24 mfcc + bpm_raw, bpm_folded) cached raw as JSON in SQLite. Target < 0.5 s/clip. Run in `ProcessPoolExecutor(ANALYSIS_WORKERS=2)`.

Normalization (per job; cache stores raw): linear bounds clipped 0..1 — pulse_clarity 0.1–0.8; onset_density 0.5–8; regularity, pulse_stability identity; syncopation 0–0.6; log2 centroid log2(400)–log2(4500); log2 rolloff log2(1000)–log2(10000); bass 0–0.6; sub 0–0.3; rms_mean_db −35..−6; dynamic_range_db 2..30; flatness 0–0.4. MFCC dims: robust z over all tracks in the job `(x−median)/(1.4826·MAD+1e-6)` → `clip(0.5+z/6,0,1)`. FeatureSummary: brightness=centroid, low_end=bass, energy=rms, dynamics=dynamic_range.

## 4. Candidates and scoring

Stages/progress: fetching_playlist 0–.05, matching_previews .05–.15, analyzing_playlist .15–.45, finding_candidates .45–.55, analyzing_candidates .55–.95, ranking .95–1. Hard timeout 300 s.

1. Playlist: up to 200 tracks; if >40, sample 40 stratified round-robin by artist (seed 42).
2. Audio for playlist tracks: Spotify embed preview if present, else Deezer (ISRC if known → keyword search with difflib ≥0.8, duration ±5 s if known). <5 analyzed → error `no_previews`; <15 → warning.
3. Vibes: group-weighted embedding (each group dims × sqrt(w_g/n_g); tempo as (cosθ,sinθ)·sqrt(w_t)/0.908 with θ=2π·frac(log2(bpm_folded/70))). numpy k-means++ 10 inits, 100 iters, seed 42, k∈{1,2,3}; pick best silhouette if ≥0.25 and every cluster ≥ max(3, 15%·n); n<8 → k=1.
4. Seeds: top 15 playlist artists by frequency; 20 seed tracks nearest each vibe centroid, proportional to share.
5. Last.fm (`https://ws.audioscrobbler.com/2.0/?format=json`, User-Agent set): `artist.getSimilar` limit 50 per seed artist (source similar_artist, affinity=match); `track.getSimilar` limit 30 per seed track (similar_track, affinity=match); `artist.getTopTags` for top 10 artists → sum counts, drop junk (seen live, favorites, female vocalists, male vocalists, 4-digit years, spotify, under 2000 listeners) → top 5 tags → `tag.getTopArtists` limit 50 pages 2–3 (source tag, affinity 0.3). Merge by normalized name: affinity = max + 0.1·(sources−1), cap 1.
6. Filters: drop playlist artists (incl. any credited artist) → `filtered_in_playlist`. `artist.getInfo` for top 80 by affinity (`stats.listeners`, string→int, cache 7 d); keep 500 ≤ listeners ≤ max_listeners, else `filtered_mainstream`.
7. Candidate tracks: similar_track hits → Deezer keyword search; others → Deezer `search/artist?q=` exact normalized match → `/artist/{id}/top?limit=3` take 2. Drop no-preview (`no_preview`) and tracks whose title already in playlist. ≤2 per artist. Budget `min(150, max(80, 5·limit))` by affinity. Downloads `Semaphore(8)`, feature cache `dz:{id}:v1`.
8. Distance groups D=Σw_g·d_g ∈[0,1]: tempo 0.25 (`min(1, c/0.15)`); groove 0.30 RMS diff of pulse_clarity, onset_density, regularity, syncopation, pulse_stability; tone/energy 0.25 RMS diff of centroid, rolloff, bass, sub, rms, dynamics, flatness; timbre 0.20 RMS diff of 24 mfcc. Candidate: nearest vibe v; `D_c = 0.7·D(c,centroid_v) + 0.3·mean(D to 3 nearest playlist tracks in v)`.
9. `D_ref` = median leave-one-out D of playlist tracks to own vibe, clamp [0.06, 0.25]. `match_score = round(100·exp(−ln(100/85)·(D_c/D_ref)^2))`.
10. Rank by `match_score·(0.9+0.1·affinity)`; ≤2 per artist (≤1 in top 10); fill across vibes proportional to share; take `limit`.
11. `why` (≤90 chars, ≤3 clauses): tempo clause ("Same {bpm} BPM pocket" within 3%; "Half-time take on your {x} BPM groove" if raw ratio≈0.5; "Double-time …" ≈2; else "Near your {x} BPM"), then closest groove trait ("matching syncopation", "same busy hats"…), then biggest difference |Δ|≥0.12 (darker/brighter, heavier/lighter low end, punchier/looser beat, busier/sparser drums, more/less syncopated, louder/softer). Vibe label e.g. "94 BPM · dark · bass-heavy · syncopated" (top 2 traits deviating most from 0.5).

Rate limits: token buckets Deezer 40/5 s, Last.fm 5/s; httpx timeout 10 s; 3 retries backoff 0.5/1/2 s + jitter on 429, 5xx, Deezer code 4. SQLite WAL at `CACHE_DIR/deepcut.sqlite` (`features`, `http_cache` tables; TTL Last.fm 7 d, Deezer search 1 d, track lookup 10 min) via `asyncio.to_thread`. Jobs in-memory dict, 1 h TTL, sweep 5 min; `MAX_CONCURRENT_JOBS=2`; single process.

## 5. Env vars

Backend: `LASTFM_API_KEY` (required), `SPOTIFY_CLIENT_ID`/`SPOTIFY_CLIENT_SECRET` (optional), `SPOTIFY_PLAYLIST_SOURCE=auto`, `CORS_ORIGINS` (comma list, default `http://localhost:3000`), `CORS_ORIGIN_REGEX` (optional), `CACHE_DIR` (`./.cache`, `/data` in Docker), `MAX_CONCURRENT_JOBS=2`, `ANALYSIS_WORKERS=2`, `JOB_TTL_SECONDS=3600`, `HTTP_USER_AGENT=Deepcut/0.1`, `LOG_LEVEL=info`, `PORT`.
Frontend: `NEXT_PUBLIC_API_URL` (unset → mock mode using `contract/*.example.json`).

## 6. Deployment

Dockerfile: python:3.12-slim, apt ffmpeg, non-root user, `ENV CACHE_DIR=/data`, `CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1`. `render.yaml` at repo root: web service `deepcut-api`, runtime docker, `dockerContext: ./backend`, `dockerfilePath: ./backend/Dockerfile`, `healthCheckPath: /api/health`, `plan: free`, secrets `sync: false`. Vercel: root dir `frontend`, set `NEXT_PUBLIC_API_URL`; `images.remotePatterns` for `*.dzcdn.net`, `i.scdn.co`, `image-cdn-*.spotifycdn.com`. README: what/how, keys (Last.fm https://www.last.fm/api/account/create; Spotify optional), embed caveat, local dev, deploy, attribution ("Previews provided by Deezer / Spotify", "Data from Last.fm").

Frontend behavior: client page; job id in URL (`/?job=<id>`) so refresh resumes; poll 1.5 s → 3 s after 30 s; stop on done/error; 404 → "job expired, run again"; POST >5 s → "Waking up server…". Log slider `value = round(10^(4 + 2·t/100))`, t 0..100, default ≈200k, labels "≤ 200K listeners". Limit choice 10/20/30. Single shared `<audio>` with `src = API + preview_proxy_url`.

## 7. Tests

Backend pytest, no network by default (`-m live` opt-in): synthetic audio (`tests/synth.py`: kick 55 Hz sine 120 ms decay per beat, hats = HP noise >7 kHz on 8ths/16ths, optional off-16th snare, −50 dB noise floor). Tests: tempo (90/120/140 within ±2% or octave-equivalent; 174 folds to 87±2), features orderings (bass vs hats, syncopated vs straight, 16th vs 8th density, click regularity > 0.8, finite & deterministic, WAV decode via miniaudio), normalize, vibes (2 blobs→k=2, 1 blob→k=1, n<8→1), score (identical→100, monotonic, D=D_ref→85, diversity), explain (≤90 chars, half-time wording), url parse, clients (respx: Deezer code 4 retry, ISRC miss→search, Spotify no-items→embed fixture, listeners string→int), **offline end-to-end pipeline** with respx fixtures + synthetic WAV previews (2 vibes; 5M-listener artist filtered; playlist artist excluded; 120 BPM candidate outranks 170 BPM noise; second run hits feature cache), contract examples validate, API (202, poll to done, 404 shape, 4th POST → 429, bad body → 400).
Frontend: vitest (slider round-trip, format, useJob with fake timers), `tsc --noEmit`, lint, `next build`; mock mode renders.
