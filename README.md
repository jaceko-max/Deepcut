# Deepcut

Find underground music by **beat flow** instead of by what's viral.

Paste a public Spotify playlist and Deepcut works out what its tracks feel like underneath the genre label: tempo, groove, drum density, low end, brightness and timbre. It then searches for songs that move the same way but have few listeners, so you get the 40,000-listener track that sits in the same pocket as your favourites instead of the same TikTok and Instagram hits everyone has already heard.

## How it works

1. **Read the playlist.** Spotify's embed page gives the track list and a 30-second preview MP3 per track. Up to 40 tracks are sampled, spread across artists.
2. **Listen.** Each preview is decoded and analysed with plain numpy and scipy (no librosa): tempo with octave-aware folding, pulse clarity, beat regularity, syncopation, onset density, spectral balance and 12 MFCC coefficients, with mean and standard deviation. A clip takes well under a second.
3. **Find your vibes.** Tracks are clustered (k-means, 1 to 3 clusters, chosen by silhouette) so a playlist that mixes 90 BPM slow burners and 124 BPM dance tracks becomes two vibes, not one muddy average.
4. **Gather candidates.** Last.fm's similar-artist, similar-track and tag graphs, plus ListenBrainz co-listening "similar artists", are walked from your main (primary, non-soundtrack) artists and the tracks closest to each vibe. Artists found by both sources, or by several of your artists, rank higher.
5. **Cut the mainstream.** Artists in your playlist and tribute/karaoke acts are removed. Last.fm listener counts enforce your cap (default 200K, adjustable from 10K to 1M); ListenBrainz user counts and Deezer fan counts act as backup caps for artists Last.fm undercounts.
6. **Stay in genre.** Your playlist's Last.fm tags become a genre profile (umbrella tags like "pop" or "rock" count less than specific ones like "shoegaze"), per vibe and overall. Candidates whose tags don't overlap enough are dropped before any audio is fetched. Christmas songs, live recordings, karaoke/cover/tribute versions, sped-up/slowed edits, instrumental versions and film/TV soundtrack cuts are filtered out unless your playlist has them too.
7. **Compare beat to beat.** Up to two Deezer previews per candidate artist get the same analysis, and the best-matching one wins (not just the most popular one). Audio distance is a weighted blend of timbre, tone, groove and tempo (octave-circular, with a small penalty for half/double time), calibrated against how tightly your own playlist clusters. The 0-100 match score is 65% audio match and 35% genre-tag fit, so off-genre tracks score visibly lower.
8. **Explain.** Every result says why it matched, for example "Same 92 BPM pocket, matching syncopation; shared tags: shoegaze, dream pop".

## Architecture

```
 Browser (Next.js, Vercel)
   |  POST /api/analyze  -> job id
   |  GET  /api/jobs/{id} (poll)      GET /api/preview/{deezer_id} -> 307 fresh preview
   v
 FastAPI (Render, Docker)  --------------------------------------------+
   |                                                                    |
   |  JobStore (in memory, 1 h TTL, admission control, per-IP limit)    |
   |        |                                                           |
   |        v                                                           |
   |  Pipeline                                                          |
   |    1 Spotify embed page --> tracks + preview URLs                  |
   |    2 Deezer (fallback)  --> previews for tracks without one        |
   |    3 audio/  decode (miniaudio) -> DSP features   [process pool]   |
   |    4 scoring/ vibes (k-means) -> candidate scoring -> explain      |
   |    5 Last.fm --> similar artists/tracks/tags, listener counts      |
   |    6 Deezer  --> candidate tracks + 30 s previews                  |
   |                                                                    |
   |  SQLite (WAL): audio feature cache + HTTP response cache           |
   +--------------------------------------------------------------------+
```

The API contract is fixed and documented by `contract/job_done.example.json` and `contract/job_running.example.json`; both the backend tests and the frontend mock mode use those files.

```
backend/   FastAPI app, audio analysis, scoring, tests, Dockerfile
frontend/  Next.js (App Router) + TypeScript + Tailwind
contract/  shared example API responses
render.yaml  Render blueprint for the backend
```

## API keys

**Last.fm (required).** Create a key at <https://www.last.fm/api/account/create> (any application name; no callback URL needed). Put it in `LASTFM_API_KEY`.

**Spotify (optional).** Deepcut works with only a Last.fm key. If you set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` (from <https://developer.spotify.com/dashboard>), the backend tries the Spotify Web API first and falls back to the embed page when the API does not return tracks. `SPOTIFY_PLAYLIST_SOURCE` can force `api` or `embed`.

### Caveat: playlist reading is unofficial

Spotify's Web API no longer returns the tracks of other users' public playlists to new apps, so Deepcut reads playlists from Spotify's **public embed page** (`open.spotify.com/embed/playlist/{id}`), which needs no login. This is unofficial and undocumented. It can break or be rate limited at any time, it only exposes roughly the first 50 to 100 tracks, and playlists must be public. If it stops working, the playlist step is the only part affected; the code lives in `backend/app/clients/spotify.py`.

## Local development

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # then set LASTFM_API_KEY
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --reload --port 8000
```

Audio decoding uses `miniaudio`, so ffmpeg is not required locally (the Docker image installs it as a fallback decoder).

Tests run fully offline (synthetic audio and mocked HTTP):

```bash
cd backend && python -m pytest          # offline suite
python -m pytest -m live                # opt in to tests that hit real services
python scripts/smoke_live.py --playlist-only            # no Last.fm key needed
LASTFM_API_KEY=... python scripts/smoke_live.py <playlist-url>   # full pipeline
```

### Frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open <http://localhost:3000>. With `NEXT_PUBLIC_API_URL` unset, the frontend runs in mock mode using the files in `contract/`.

### Backend environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `LASTFM_API_KEY` | required | |
| `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` | unset | optional |
| `SPOTIFY_PLAYLIST_SOURCE` | `auto` | `auto`, `api` or `embed` |
| `CORS_ORIGINS` | `http://localhost:3000` | comma-separated |
| `CORS_ORIGIN_REGEX` | unset | e.g. Vercel preview URLs |
| `CACHE_DIR` | `./.cache` (`/data` in Docker) | SQLite cache |
| `MAX_CONCURRENT_JOBS` | `2` | |
| `ANALYSIS_WORKERS` | `2` | audio process pool size |
| `JOB_TTL_SECONDS` | `3600` | |
| `JOB_TIMEOUT_SECONDS` | `420` | hard limit for one search; cold-cache searches on Render's free tier take about 2–4 minutes |
| `HTTP_USER_AGENT` | `Deepcut/0.1` | also sent to ListenBrainz (no key needed; if it is down, results fall back to Last.fm only with a warning) |
| `LOG_LEVEL` | `info` | |
| `GLOBAL_JOB_LIMIT` | `30` | max analyses started across all clients per 10 minutes (429 `too_many_jobs`) |
| `PREVIEW_SECRET` | random per start | signs preview links; set a fixed value (Render generates one) so play buttons survive restarts |
| `ENABLE_DOCS` | unset | set to `1` to serve `/docs`, `/redoc` and `/openapi.json` (off by default) |

## Deploy

### Backend on Render

1. Push this repo to GitHub.
2. In Render choose **New > Blueprint** and select the repo. `render.yaml` defines the `deepcut-api` Docker web service (free plan, health check `/api/health`).
3. When prompted, set `LASTFM_API_KEY` and `CORS_ORIGINS` (your Vercel URL, e.g. `https://deepcut.vercel.app`). `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` are optional.
4. Note the service URL, e.g. `https://deepcut-api.onrender.com`.

The free plan sleeps when idle and has an ephemeral disk, so the first request after a pause is slow and the feature cache resets on redeploy. The frontend pings `/api/health` on load to wake the server.

### Frontend on Vercel

1. Import the repo in Vercel and set **Root Directory** to `frontend`.
2. Add the environment variable `NEXT_PUBLIC_API_URL` set to your Render URL (no trailing slash). Optionally add `NEXT_PUBLIC_REPO_URL` (your GitHub repo URL) to show a "Source on GitHub" link in the footer.
3. Deploy, then copy the production URL into the backend's `CORS_ORIGINS` on Render and redeploy the backend if needed.

## Attribution

- Playlist data and 30-second previews: Spotify.
- Candidate previews and artwork: Deezer.
- Similarity, tags and listener counts: Last.fm.
- Co-listening similar artists and popularity: ListenBrainz (MetaBrainz Foundation).

Deepcut is an independent project and is not affiliated with or endorsed by Spotify, Deezer, Last.fm or ListenBrainz. Previews are streamed from the providers and are never stored.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Deepcut contributors.
