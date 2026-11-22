import pytest

from app.clients.spotify import parse_playlist_id

ID = "37i9dQZF1DXcBWIGoYBM5M"


@pytest.mark.parametrize("s", [
    f"https://open.spotify.com/playlist/{ID}?si=abc123",
    f"https://open.spotify.com/intl-de/playlist/{ID}",
    f"http://open.spotify.com/playlist/{ID}/",
    f"open.spotify.com/playlist/{ID}",
    f"spotify:playlist:{ID}",
    f"  {ID}  ",
])
def test_valid(s):
    assert parse_playlist_id(s) == ID


@pytest.mark.parametrize("s", ["", "hello", "https://open.spotify.com/track/" + ID,
                               "https://example.com/playlist/" + ID, "short", ID + "extra"])
def test_invalid(s):
    assert parse_playlist_id(s) is None


def test_split_artists_keeps_commas_inside_names():
    from app.textutil import split_artists

    assert split_artists("Justice, Tame Impala") == ["Justice", "Tame Impala"]
    assert split_artists("Tyler, The Creator, Kali Uchis") == ["Tyler, The Creator", "Kali Uchis"]
    assert split_artists("") == []
