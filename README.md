# Deepcut

Find underground music by **mood and beat flow** instead of by what's viral.

Paste a public Spotify playlist and Deepcut works out what its tracks feel like underneath the genre label: their mood (relaxed, sad, happy, aggressive, party, danceable, acoustic or electronic...), style, tempo, groove, drum density, low end, brightness and timbre. It then searches for songs with the same mood that move the same way but have few listeners, so you get the 40,000-listener track that sits in the same late-night pocket as your favourites instead of the same TikTok and Instagram hits everyone has already heard.

## How it works

1. **Read the playlist.** Spotify's embed page gives the track list and a 30-second preview MP3 per track. Up to 40 tracks are sampled, spread across artists.
2. **Listen.** Each preview is decoded and analysed with plain numpy and scipy (no librosa): tempo with octave-aware folding, pulse clarity, beat regularity, syncopation, onset density, spectral balance and 12 MFCC coefficients, with mean and standard deviation. A pretrained network (Essentia's Discogs-EffNet, run with onnxruntime on CPU) also listens to six 2-second windows and gives a profile over 400 Discogs music styles, which captures sound and production far better than MFCCs. The same pass feeds Essentia's small **mood heads** (relaxed, sad, happy, aggressive, party, acoustic, electronic, danceable, voice/instrumental, bright/dark, approachability, engagement, plus 56 mood/theme tags) at almost no extra cost (~2 ms and ~4 MB). A clip takes about 0.2 s.
3. **Find your vibes.** Tracks are clustered (k-means, 1 to 3 clusters, chosen by silhouette) so a playlist that mixes 90 BPM slow burners and 124 BPM dance tracks becomes two vibes, not one muddy average. Each vibe is named by its mood, e.g. "late-night · mellow · sensual" (mood words come from the mood heads, relative to a broad reference set of songs and, with several vibes, to the rest of the playlist).
4. **Gather candidates.** Last.fm's similar-artist, similar-track and tag graphs, plus ListenBrainz co-listening "similar artists", are walked from your main (primary, non-soundtrack) artists and the tracks closest to each vibe. Artists found by both sources, or by several of your artists, rank higher.
5. **Cut the mainstream, but stay findable.** Artists in your playlist and tribute/karaoke acts are removed. Last.fm listener counts enforce your cap (default 500K, adjustable from 10K to 1M); ListenBrainz user counts and Deezer fan counts act as backup caps for artists Last.fm undercounts. There is also a floor, so results are not so obscure that you can't find them anywhere: new artists need at least 15K Last.fm listeners (never more than a quarter of the cap).
6. **Stay in genre.** Your playlist's Last.fm tags become a genre profile (umbrella tags like "pop" or "rock" count less than specific ones like "shoegaze"), per vibe and overall. Candidates whose tags don't overlap enough are dropped before any audio is fetched. Christmas songs, live recordings, karaoke/cover/tribute versions, sped-up/slowed edits, instrumental versions and film/TV soundtrack cuts are filtered out unless your playlist has them too.
7. **Compare mood first, then sound.** Up to two Deezer previews per candidate artist get the same analysis, and the best-matching one wins (not just the most popular one). Audio distance is 53% mood (the mood heads as z-scores plus derived arousal and valence), 29% style profile and 18% the hand-built groups (timbre, tone, groove and tempo, octave-circular with a small penalty for half/double time), calibrated against how tightly your own playlist clusters. The 0-100 match score is 80% audio and 20% genre-tag fit, so mood leads and genre is only a light check.
8. **Drop what doesn't belong instead of padding the list.** A candidate whose mood falls outside your playlist's own mood envelope (further from the nearest vibe's mood centre than 1.3x the 80th percentile of your own tracks) is dropped, whatever its genre; so is anything below the quality floor (70, or lower for loosely knit playlists: what your own tracks score against the rest of the playlist). If fewer strong matches remain than you asked for, you get a shorter list and a note. Two picks that sound nearly identical (closer than almost any two tracks in your playlist) count once.
9. **A mix of popularity.** About a third of the list are deep cuts (below), a third **rising** artists (100K listeners up to your cap) and a third **underground** artists (15K to 100K). Each tier fills by score and a short tier is topped up from the others. Because similar-artist graphs of popular seeds mostly lead to popular artists, Deepcut also takes one ListenBrainz hop from the artists it kept and pre-selects neighbours whose ListenBrainz user counts predict the underground band. Results carry a small "Rising" / "Underground" badge.
10. **Deep cuts from artists you like.** For up to 8 of your playlist's main artists (at least 2 tracks each), Deepcut also looks for their lesser-known songs: tracks outside the artist's 5 biggest on Deezer with at most 30% of the Last.fm listeners of the artist's most-listened track (Deezer's popularity rank is the fallback when Last.fm has no track list), not in your playlist, not a live/remix/sped-up/holiday version and not another version of a hit. Up to 3 per artist are analysed and scored the same way. About a third of the list (7 of 20) are these deep cuts, at most 2 per artist, scored, mood-checked and floored like everything else. The listener cap doesn't apply to them, because the artist is known and only the track is obscure. They're marked "Deep cut" in the results.
11. **Explain.** Every result says why it matched, mood first, for example "Same late-night/sensual mood, same 92 BPM pocket; shared tags: rnb" or "Deep cut from Don Toliver · Same groovy mood, near your 112 BPM".

## Features

- **Paste tracks instead of a link.** The form has a "Spotify link | Paste tracks" switch. Paste one track per line as `Artist - Title` or `Title by Artist`, or a CSV exported by Exportify (`"Track Name"`, `"Artist Name(s)"`). Up to 200 tracks, with a live line counter. Pasted tracks have no previews, so Deezer matching finds the audio, and the 100-track embed limit no longer applies.
- **Vibe filter.** When your playlist has more than one vibe, chips above the results ("All", then each vibe with its share) filter the list. The choice lives in the URL as `?vibe=` next to `?job=`.
- **Thumbs up / down and refine.** Every result has More like this / Less like this buttons. The backend returns a reserve pool of the next ~40 best analyzed candidates; the browser re-scores list plus reserve (`score + 60·similarity to liked − 60·similarity to disliked`, using tempo, groove, tone features, artist, source and vibe), drops disliked tracks and pulls in the best reserve tracks. A note offers Undo last and Reset. Feedback is kept in `localStorage` per job.
- **Permanent share links.** "Share results" packs the finished list (no reserve) into the URL fragment (`#r=…`, deflate-raw + base64url; plain base64url if `CompressionStream` is missing). The link renders read-only without the server and never reaches one, because fragments are not sent in requests. Play buttons still work because `preview_proxy_url` links are HMAC-signed (keep `PREVIEW_SECRET` fixed).
- **Copy / export.** Copy the list as `1. Artist – Title` lines, or download `.txt` or `.csv`.

## API

- `POST /api/analyze` takes `{playlist_url | tracks: [{artist, title}], max_listeners?, limit?}`. Exactly one of `playlist_url` or `tracks` (1 to 200 items, artist up to 200 and title up to 300 characters) is required, otherwise `400 invalid_request`. Returns `202 {job_id, status, poll_url}`. The request body limit is 64 KB (it was 16 KB) so 200 pasted lines fit.
- `GET /api/jobs/{id}` returns the job. `playlist.source` is `spotify_api`, `spotify_embed` or `pasted`. When done, `results` holds the list (it can be shorter than `limit` when fewer candidates pass the quality floor; a warning then says so) and the optional `reserve` (default `[]`) holds up to ~40 more analyzed candidates that passed the same filters, same shape as `results`, with `rank` continuing after the list and one per artist where possible. Each result has an optional `tier` (`deep_cut`, `rising` or `underground`; absent in older payloads).
- `GET /api/preview/{deezer_id}?sig=…` redirects to a fresh preview.

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

The style model (18 MB) and the 13 mood heads (~9 MB together) are not in the repo. The Docker image downloads them at build time (each pinned by sha256 in `app/audio/embed.py`); locally, fetch them once with `python -m app.audio.embed download` (they land in the git-ignored `backend/.cache/models/`). Without the heads the backend scores with style + DSP (the pre-mood weights); without the style model, with the DSP features only. Cached clips analysed without mood outputs are re-analysed once (`FEATURE_VERSION` 3).

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
| `IP_JOB_LIMIT` | `6` | searches per visitor per 10 minutes; visitors are told apart by Cloudflare's `CF-Connecting-IP` header on Render |
| `HTTP_USER_AGENT` | `Deepcut/0.1` | also sent to ListenBrainz (no key needed; if it is down, results fall back to Last.fm only with a warning) |
| `LOG_LEVEL` | `info` | |
| `GLOBAL_JOB_LIMIT` | `30` | max analyses started across all clients per 10 minutes (429 `too_many_jobs`) |
| `PREVIEW_SECRET` | random per start | signs preview links; set a fixed value (Render generates one) so play buttons survive restarts |
| `ENABLE_DOCS` | unset | set to `1` to serve `/docs`, `/redoc` and `/openapi.json` (off by default) |
| `EMBEDDINGS` | `onnx` | `off` scores with the hand-built DSP features only; `onnx` uses the style model when it is installed |
| `EMBEDDING_MODEL_PATH` | `backend/.cache/models/discogs-effnet-bsdynamic-1.onnx` (`/opt/models/...` in Docker) | where the style model lives |
| `MOOD_MODELS_DIR` | the style model's directory | where the mood heads live |

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

## Accuracy benchmark

Changes to scoring and candidate search are checked against 8 public playlists of different genres (a user R&B/rap mix, lofi beats, RapCaviar, All New Indie, Jazz Classics, Kickass Metal, mint, Hot Country) with in-process runs of the pipeline:

- **Held-out artists:** 20% of a playlist's artists (all their tracks) are removed, the listener cap is lifted, and we measure how many come back: in the candidate pool, among the scored candidates, and in the top 20 by match score (R@20), plus how well the score ranks them above the other candidates (AUC).
- **Own vs other playlists:** AUC of the audio distance for a playlist's own tracks (leave-one-out) against the other playlists' tracks.
- **Genre coherence:** mean Last.fm tag cosine of the top-20 discovery results to the playlist, from raw tags independent of the pipeline's tag weighting.

| mean of 8 playlists | before | after |
| --- | --- | --- |
| own vs other playlist AUC | 0.855 | 0.949 |
| held-out artists among scored candidates | 0.40 | 0.42 |
| held-out artists in top 20 (R@20) | 0.20 | 0.28 |
| held-out AUC, final score / audio only | 0.65 / 0.53 | 0.68 / 0.59 |
| genre coherence of top 20 | 0.70 | 0.69 |

Re-scoring the same candidates without the style model gives R@20 0.21 and audio AUC 0.50, so most of the ranking gain comes from the learned style profile, the rest from ordering Last.fm lookups by how many playlist artists point at a candidate.

**Mood rework (Nov 2026).** Two more measures: **mood fit**, the mean cosine of the top-20 results' mood vectors to the playlist's mood centroid (the playlists' own tracks score 0.71 leave-one-out), and **in envelope**, the share of top-20 results inside the playlist's own mood envelope. Both are computed from the same mood vectors for every run.

| mean of 8 playlists | before | after |
| --- | --- | --- |
| mood fit of top 20 | 0.767 | 0.810 |
| top 20 inside the playlist's mood envelope | 0.969 | 1.000 |
| popularity mix (deep cut / rising / underground, of 20) | 4.9 / 12.3 / 2.9 | 5.0 / 9.0 / 6.0 |
| own vs other playlist AUC | 0.949 | 0.944 |
| genre coherence of top 20 | 0.692 | 0.676 |
| held-out AUC, same candidates re-scored (final / audio) | 0.69 / 0.59 | 0.74 / 0.67 |
| held-out R@20, same candidates re-scored | 0.28 | 0.26 |
| held-out R@20, live run | 0.28 | 0.15 |

The live held-out recall drops because the popularity mix spends part of the lookup and analysis budget on small (under 100K) artists, while held-out playlist artists are mostly big: old scoring on the new candidate pool gets R@20 0.21. Mood-based scoring ranks held-out artists' audio better (AUC 0.59 -> 0.67 on identical candidates). Clustering vibes on mood instead of DSP was tried and lost (audio AUC 0.61 vs 0.67), so vibes are clustered as before and only labelled by mood.

## Attribution

- Playlist data and 30-second previews: Spotify.
- Candidate previews and artwork: Deezer.
- Similarity, tags and listener counts: Last.fm.
- Co-listening similar artists and popularity: ListenBrainz (MetaBrainz Foundation).
- Style model: Discogs-EffNet (`discogs-effnet-bsdynamic-1`) from the [Essentia models](https://essentia.upf.edu/models/) by the Music Technology Group, Universitat Pompeu Fabra, licensed under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use only; commercial licences are available from UPF). Alonso-Jiménez, Serra and Bogdanov, "Music Representation Learning Based on Editorial Metadata from Discogs", ISMIR 2022. Deepcut uses the model file unmodified.
- Mood heads: Essentia's classification/regression heads for Discogs-EffNet (`mood_relaxed`, `mood_sad`, `mood_happy`, `mood_aggressive`, `mood_party`, `mood_acoustic`, `mood_electronic`, `danceability`, `voice_instrumental`, `timbre`, `approachability_regression`, `engagement_regression`, `mtg_jamendo_moodtheme`, all `-discogs-effnet-1`), same authors and same license (the models directory's LICENSE file), used unmodified.

Deepcut is an independent project and is not affiliated with or endorsed by Spotify, Deezer, Last.fm or ListenBrainz. Previews are streamed from the providers and are never stored.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Deepcut contributors. The MIT license covers Deepcut's code only; the style model and mood heads keep their own non-commercial license (see Attribution).
