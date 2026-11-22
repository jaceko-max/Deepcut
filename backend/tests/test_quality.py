"""Recommendation-quality units: ListenBrainz client, genre-tag profile/cosine, contamination filters."""
import json

import httpx
import pytest
import respx

from app.cache import Cache
from app.clients.http import HttpClient, UpstreamError
from app.clients.listenbrainz import (LABS, PREFERRED_ALGOS, ListenBrainzClient, closest_algorithm,
                                      parse_permitted_algorithms)
from app.filters import allowed_categories, artist_is_tribute, is_contaminated, track_flags
from app.scoring.explain import MAX_LEN, with_tags
from app.scoring.tags import build_profile, clean_tag, shared_tags, similarity, tag_vector

SIM = LABS + "/similar-artists/json"
POP = "https://api.listenbrainz.org/1/popularity/artist"
LOOKUP = "https://api.listenbrainz.org/1/metadata/lookup/"
A = "d5cc67b8-1cc4-453b-96e8-44487acdebea"
B = "a74b1b7f-71a5-4011-9441-d0b5e4122711"
C = "63aa26c3-d59b-4da4-84ac-716b54f1ef4d"
NEW_ALGO = "session_based_days_9000_session_300_contribution_5_threshold_15_limit_50_skip_30"
ERR_400 = ("<!doctype html><title>400 Bad Request</title><p>1 validation error for SimilarArtistsViewerInput<br>"
           "algorithm<br>  value is not a valid enumeration member; permitted: "
           "&#39;session_based_days_75_session_300_contribution_5_threshold_10_limit_100_filter_True_skip_30&#39;, "
           f"&#39;{NEW_ALGO}&#39;, "
           "&#39;session_based_days_1825_session_300_contribution_3_threshold_10_limit_100_filter_True_skip_30&#39; "
           "(type=type_error.enum; enum_values=[&lt;AlgorithmEnum.x: &#39;bogus_should_not_be_parsed&#39;&gt;])</p>")


@pytest.fixture
async def lb(tmp_path):
    async with httpx.AsyncClient() as c:
        yield ListenBrainzClient(HttpClient(c), Cache(str(tmp_path / "c")))


def rows(ref, items):
    return [{"artist_mbid": m, "name": n, "score": s, "reference_mbid": ref, "type": "Group"} for m, n, s in items]


# -- ListenBrainz -------------------------------------------------------------
@respx.mock
async def test_lb_similar_artists_parses_sorts_and_drops_seed(lb):
    route = respx.get(SIM).mock(return_value=httpx.Response(200, json=rows(A, [(B, "Radiohead", 10), (A, "Beach House", 99), (C, "Tame Impala", 40)])))
    out = await lb.similar_artists(A)
    assert [n for _, n, _ in out] == ["Tame Impala", "Radiohead"]
    assert route.calls[0].request.url.params["algorithm"] == PREFERRED_ALGOS[0]
    assert await lb.similar_artists(A) == out and route.call_count == 1, "second call must hit the http cache"


@respx.mock
async def test_lb_400_algorithm_fallback_is_parsed_chosen_and_cached(lb, tmp_path):
    def handler(request):
        if request.url.params["algorithm"] != NEW_ALGO:
            return httpx.Response(400, text=ERR_400)
        return httpx.Response(200, json=rows(A, [(B, "Radiohead", 10)]))
    route = respx.get(SIM).mock(side_effect=handler)
    out = await lb.similar_artists(A)
    assert out == [(B, "Radiohead", 10.0)]
    assert lb.algorithm == NEW_ALGO and route.call_count == 2
    # a fresh client (new process) reuses the cached algorithm choice: no 400 round trip
    async with httpx.AsyncClient() as c:
        lb2 = ListenBrainzClient(HttpClient(c), lb.cache)
        await lb2.similar_artists(B)
    assert route.calls[-1].request.url.params["algorithm"] == NEW_ALGO and route.call_count == 3


def test_parse_permitted_and_closest_algorithm():
    got = parse_permitted_algorithms(ERR_400)
    assert NEW_ALGO in got and "bogus_should_not_be_parsed" not in got and len(got) == 3
    # 7500-day preference: 9000 days is closer than 75 or 1825
    assert closest_algorithm(got, PREFERRED_ALGOS[0]) == NEW_ALGO
    assert closest_algorithm([], PREFERRED_ALGOS[0]) is None
    assert closest_algorithm([PREFERRED_ALGOS[0], NEW_ALGO]) == PREFERRED_ALGOS[0]


@respx.mock
async def test_lb_400_without_usable_algorithm_raises(lb):
    respx.get(SIM).mock(return_value=httpx.Response(400, text="nope"))
    with pytest.raises(UpstreamError):
        await lb.similar_artists(A)


@respx.mock
async def test_lb_server_errors_raise_upstream_error_after_retries(lb):
    route = respx.get(SIM).mock(return_value=httpx.Response(503))
    with pytest.raises(UpstreamError):
        await lb.similar_artists(A)
    assert route.call_count == 4  # HttpClient retries
    assert await lb.similar_artists("not-a-uuid") == []


@respx.mock
async def test_lb_popularity_batches_and_caches(lb):
    def handler(request):
        ids = json.loads(request.content)["artist_mbids"]
        return httpx.Response(200, json=[{"artist_mbid": i, "total_user_count": 7, "total_listen_count": 70} for i in ids if i != C])
    route = respx.post(POP).mock(side_effect=handler)
    assert await lb.popularity([A, B, C, "junk"]) == {A: (7, 70), B: (7, 70)}
    assert await lb.popularity([A, C]) == {A: (7, 70)}
    assert route.call_count == 1


@respx.mock
async def test_lb_lookup_disables_itself_on_401(lb):
    route = respx.get(LOOKUP).mock(return_value=httpx.Response(401, json={"code": 401}))
    assert await lb.lookup_artist_mbid("Beach House", "Space Song") is None
    assert await lb.lookup_artist_mbid("Slowdive", "Alison") is None
    assert route.call_count == 1


@respx.mock
async def test_lb_lookup_ok(lb):
    respx.get(LOOKUP).mock(return_value=httpx.Response(200, json={"artist_mbids": [A], "recording_name": "Space Song"}))
    assert await lb.lookup_artist_mbid("Beach House", "Space Song") == A


# -- tags ---------------------------------------------------------------------
def test_clean_tag_drops_junk_nationalities_years():
    assert clean_tag("Seen Live") is None and clean_tag("90s") is None and clean_tag("british") == "british"
    assert clean_tag("2019") is None and clean_tag("under 2000 listeners") is None
    assert clean_tag("Hip-Hop") == "hip hop" and clean_tag("Dream Pop") == "dream pop"


SHOEGAZE = tag_vector([("shoegaze", 100), ("dream pop", 80), ("indie", 40), ("seen live", 30), ("british", 20)])
DREAM = tag_vector([("dream pop", 100), ("shoegaze", 60), ("indie pop", 30)])
REGGAETON = tag_vector([("reggaeton", 100), ("latin", 70), ("pop", 40), ("spanish", 30)])
XMAS = tag_vector([("christmas", 100), ("pop", 60), ("holiday", 40)])
POP_A = tag_vector([("pop", 100), ("dance pop", 50), ("synthpop", 30)])
POP_B = tag_vector([("pop", 100), ("electropop", 40), ("indie pop", 30), ("dance pop", 20)])


def test_tag_vector_cleans():
    assert set(SHOEGAZE) == {"shoegaze", "dream pop", "indie", "british"}
    assert SHOEGAZE["shoegaze"] == 1.0


def test_profile_cosine_separates_genres():
    prof = build_profile({"mbv": SHOEGAZE, "bh": DREAM}, {"mbv": 3, "bh": 1})
    assert similarity(DREAM, prof) > 0.8
    assert similarity(REGGAETON, prof) == 0.0
    assert similarity({}, prof) == 0.0 and similarity(DREAM, {}) == 0.0


def test_generic_tags_are_downweighted():
    pop_prof = build_profile({"a": POP_A, "b": POP_B}, {"a": 2, "b": 2})
    # sharing only the umbrella tag "pop" must not be enough to pass the 0.12 filter...
    assert similarity(REGGAETON, pop_prof) < 0.12
    assert similarity(XMAS, pop_prof) < 0.2
    # ...while a specific shared tag (dance pop) is
    assert similarity(tag_vector([("dance pop", 100), ("pop", 80)]), pop_prof) > 0.4


def test_scene_tags_separate_same_genre_other_language():
    us = build_profile({"a": tag_vector([("rap", 100), ("hip hop", 80), ("trap", 50), ("american", 40)]),
                        "b": tag_vector([("pop", 100), ("american", 50), ("dance pop", 40)])}, {"a": 1, "b": 1})
    finnish = tag_vector([("finnish", 100), ("hip-hop", 80), ("rap", 70), ("suomirap", 50)])
    us_rap = tag_vector([("rap", 100), ("trap", 70), ("hip-hop", 60), ("american", 30)])
    assert similarity(us_rap, us) > similarity(finnish, us) + 0.2


def test_profile_weights_artist_frequency_sublinearly():
    p = build_profile({"x": SHOEGAZE, "y": REGGAETON}, {"x": 9, "y": 1})
    assert p["shoegaze"] > p["reggaeton"] > 0  # sqrt weighting: 3:1, not 9:1


def test_shared_tags_skip_umbrella_tags():
    prof = build_profile({"a": SHOEGAZE}, {"a": 1})
    assert set(shared_tags(DREAM, prof)) == {"shoegaze", "dream pop"}
    assert "pop" not in shared_tags(POP_A, build_profile({"b": POP_B}, {"b": 1}))


def test_with_tags_fits_90_chars():
    s = with_tags("Same 92 BPM pocket, matching syncopation", ["shoegaze", "dream pop"])
    assert s == "Same 92 BPM pocket, matching syncopation; shared tags: shoegaze, dream pop"
    long = "Half-time take on your 140 BPM groove, matching straight feel, heavier low end"
    out = with_tags(long, ["experimental hip hop", "abstract hip hop"])
    assert len(out) <= MAX_LEN and out.startswith("Half-time take on your 140 BPM groove") and "shared tags:" in out
    assert with_tags("x", []) == "x"
    assert with_tags("Near your 90 BPM", ["a" * 100]) == "Near your 90 BPM"


# -- contamination filters ----------------------------------------------------
@pytest.mark.parametrize("title,album,artist,cat", [
    ("Last Christmas", "Pop Hits", "Someone", "holiday"),
    ("Mad - Wicked Wonderland", "Wicked Wonderland (Disney Original Soundtrack)", "Kylie Cantrall", "soundtrack"),
    ("How Far I'll Go (From \"Moana\")", "Moana", "Alessia Cara", "soundtrack"),
    ("Hypertension (Live at Red Rocks '23)", "Live at Red Rocks", "bootleg gizzard", "live"),
    ("Song - Live", "Album", "X", "live"),
    ("Hit Song (Karaoke Version)", "Karaoke Hits", "X", "karaoke"),
    ("Hit Song (Originally Performed by Y)", "", "X", "cover"),
    ("Hit Song (Acoustic Cover)", "", "X", "cover"),
    ("Espresso (Sped Up)", "", "X", "altered"),
    ("Espresso - slowed + reverb", "", "X", "altered"),
    ("Song (8D Audio)", "", "X", "altered"),
    ("Song (Nightcore)", "", "X", "altered"),
    ("Song (Instrumental Version)", "", "X", "instrumental"),
    ("Song", "Album", "The Gizzard Tribute Band", "tribute"),
    ("Song", "A Tribute to Radiohead", "X", "tribute"),
    ("Feliz Navidad", "", "X", "holiday"),
])
def test_contamination_rejects(title, album, artist, cat):
    assert is_contaminated(title, album, artist) == cat


@pytest.mark.parametrize("title,album,artist", [
    ("Holiday", "American Idiot", "Green Day"),
    ("Live Forever", "Definitely Maybe", "Oasis"),
    ("Cover Me Up", "Southeastern", "Jason Isbell"),
    ("Alive", "Alive", "Daft Punk"),
    ("Slow Down", "Slow", "Selena Faider"),
    ("Delivery", "Deliverance", "Live Wire"),
])
def test_contamination_keeps_normal_tracks(title, album, artist):
    assert is_contaminated(title, album, artist) is None


def test_tribute_artist_names():
    assert artist_is_tribute("Vitamin String Quartet") and artist_is_tribute("The Karaoke Channel")
    assert artist_is_tribute("Rockabye Baby!") and artist_is_tribute("Original Broadway Cast of Hamilton")
    assert not artist_is_tribute("King Gizzard & The Lizard Wizard") and not artist_is_tribute("Castlebeat")


def test_playlist_share_allows_category():
    titles = ["Last Christmas", "Jingle Bell Rock", "Santa Baby", "Normal Song"] + [f"S{i}" for i in range(6)]
    allowed = allowed_categories(titles)
    assert "holiday" in allowed and "live" not in allowed
    assert is_contaminated("Rudolph the Red-Nosed Reindeer", "", "X", allowed) is None
    assert is_contaminated("Song (Live)", "", "X", allowed) == "live"
    assert allowed_categories([]) == set()
    assert track_flags("Song (Live) [Sped Up]", None) == {"live", "altered"}


# -- seeds / names / Deezer fans ----------------------------------------------
def test_strict_name_catches_bogus_autocorrect():
    from app.textutil import strict_name
    assert strict_name("Florence + The Machine") != strict_name("Florence   The Machine")
    assert strict_name("Florence + The Machine") == strict_name("Florence and the Machine")
    assert strict_name("Beyoncé") == strict_name("beyonce")


def test_pick_seeds_primary_artists_and_non_artists():
    from app.clients.spotify import PlaylistTrack
    from app.pipeline import pick_seeds
    from app.scoring.normalize import Norm
    from app.scoring.vibes import Vibe
    import numpy as np
    tracks = [PlaylistTrack(f"id{i}", f"T{i}", arts) for i, arts in enumerate([
        ["Bruno Mars"], ["Bruno Mars"], ["Taylor Swift", "SZA"], ["Grand Theft Auto VI"],
        ["Various Artists"], ["Doja Cat", "SZA"], ["King Gizzard Tribute Band"], ["Far Away"], ["Near"]])]
    n = lambda x: Norm(120, 120, np.full(5, x), np.full(7, x), np.full(24, x))
    pl = [(tracks[7], {}), (tracks[8], {})]
    norms = [n(0.9), n(0.5)]
    vibes = [Vibe(0, [0, 1], 1.0, 120, n(0.5))]
    seeds, counts = pick_seeds(tracks, pl, norms, vibes)
    assert seeds[0] == "Bruno Mars"
    assert "SZA" not in seeds and "Grand Theft Auto VI" not in seeds and "Various Artists" not in seeds
    assert "King Gizzard Tribute Band" not in seeds
    assert seeds.index("Near") < seeds.index("Far Away"), "ties broken by distance to the vibe centroid"
    assert counts["Bruno Mars"] == 2 and "SZA" not in counts


@respx.mock
async def test_deezer_find_artist_returns_fans(tmp_path):
    from app.clients.deezer import DeezerClient
    respx.get("https://api.deezer.com/search/artist").mock(return_value=httpx.Response(200, json={"data": [
        {"id": 1, "name": "Greeicy", "nb_fan": 838000}]}))
    async with httpx.AsyncClient() as c:
        d = DeezerClient(HttpClient(c), Cache(str(tmp_path / "c")))
        assert await d.find_artist("Greeicy") == (1, 838000)
        assert await d.find_artist_id("Greeicy") == 1
        assert await d.find_artist("Nobody") == (None, None)


def test_sibling_vibe_labels_are_not_mirror_images():
    import numpy as np
    from app.scoring.explain import vibe_labels
    from app.scoring.normalize import Norm
    a = Norm(90, 90, np.array([.5, .2, .5, .3, .5]), np.array([.2, .5, .8, .5, .4, .5, .5]), np.zeros(24))
    b = Norm(120, 120, np.array([.5, .8, .5, .7, .5]), np.array([.8, .5, .2, .5, .6, .5, .5]), np.zeros(24))
    ref = Norm(105, 105, (a.groove + b.groove) / 2, (a.tone + b.tone) / 2, np.zeros(24))
    la, lb = vibe_labels([a, b], [90, 120], ref)
    ta, tb = la.split(" · ")[1:], lb.split(" · ")[1:]
    opposite = {"darker": "brighter", "heavier low end": "lighter low end", "busier drums": "sparser drums",
                "more syncopated": "straighter beat", "louder": "softer"}
    opposite.update({v: k for k, v in opposite.items()})
    assert not any(opposite.get(x) in tb for x in ta), (la, lb)
    assert len(ta) == len(tb) == 2


def test_interleave_families_gives_every_seed_family_turns():
    from app.pipeline import Cand, interleave_families

    def cand(name, seed, aff):
        c = Cand(name)
        c.seed_aff[seed] = aff
        return c

    dominant = [cand(f"d{i}", "daniel caesar", 0.9 - i * 0.01) for i in range(8)]
    other = [cand(f"t{i}", "don toliver", 0.5) for i in range(3)]
    out = interleave_families(dominant + other, {"daniel caesar": 9, "don toliver": 4})
    names = [c.name for c in out]
    assert names[:5] == ["d0", "d1", "d2", "t0", "t1"]  # 3 turns (sqrt 9) then 2 turns (sqrt 4)
    assert sorted(names) == sorted(c.name for c in dominant + other)


def test_select_diverse_caps_one_family():
    from app.scoring.score import Scored, select_diverse

    scored = [Scored(key=i, artist=f"a{i}", vibe_id=0, score=99 - i, affinity=0.5, family="daniel caesar") for i in range(15)]
    scored += [Scored(key=100 + i, artist=f"b{i}", vibe_id=0, score=70, affinity=0.5, family="don toliver") for i in range(10)]
    chosen = select_diverse(scored, 20, {0: 1.0})
    fams = [s.family for s in chosen]
    assert len(chosen) == 20
    assert fams.count("don toliver") == 10  # the weaker family is not crowded out
    assert fams.count("daniel caesar") == 10  # cap of 6 is lifted only to fill the remaining slots


def test_holiday_filter_catches_stocking_stuffer_but_not_lookalikes():
    from app.filters import TITLE_PATTERNS

    h = TITLE_PATTERNS["holiday"]
    assert h.search("Stocking Stuffer") and h.search("Santa Tell Me") and h.search("Under the Tree")
    assert not h.search("Stockholm Syndrome") and not h.search("Santana") and not h.search("Red Wine Holiday")


def test_family_cap_follows_playlist_share():
    from app.scoring.score import Scored, select_diverse

    outlier = [Scored(key=i, artist=f"w{i}", vibe_id=0, score=95, affinity=0.5, family="wu tang clan") for i in range(6)]
    core = [Scored(key=50 + i, artist=f"c{i}", vibe_id=0, score=80, affinity=0.5, family="daniel caesar") for i in range(10)]
    other = [Scored(key=90 + i, artist=f"t{i}", vibe_id=0, score=70, affinity=0.5, family="don toliver") for i in range(10)]
    fourth = [Scored(key=200 + i, artist=f"f{i}", vibe_id=0, score=60, affinity=0.5, family="frank ocean") for i in range(10)]
    chosen = select_diverse(outlier + core + other + fourth, 12, {0: 1.0},
                            {"wu tang clan": 0.02, "daniel caesar": 0.12, "don toliver": 0.08, "frank ocean": 0.2})
    fams = [s.family for s in chosen]
    assert fams.count("wu tang clan") == 2  # 2% of the playlist: the minimum, despite the top scores
    assert fams.count("daniel caesar") == 4 and fams.count("don toliver") == 3 and fams.count("frank ocean") == 3
    assert len(chosen) == 12
