"""cookies.txt handling for the YouTube resolver."""

from __future__ import annotations

import pytest

from spotter.ingest import resolver
from spotter.ingest.resolver import ResolveError, find_cookies_file, resolve_stream

COOKIES = (
    "# Netscape HTTP Cookie File\n"
    ".youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\tabc\n"
)


def test_find_cookies_file_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_cookies_file(None) is None
    (tmp_path / "cookies.txt").write_text(COOKIES)
    assert find_cookies_file(None) == "cookies.txt"
    assert find_cookies_file("") == "cookies.txt"


def test_find_cookies_file_missing_configured_path_fails(tmp_path):
    with pytest.raises(ResolveError, match="does not exist"):
        find_cookies_file(str(tmp_path / "nope.txt"))


def test_load_cookie_jar_rejects_non_netscape(tmp_path):
    bad = tmp_path / "cookies.json"
    bad.write_text('[{"name": "SID"}]')
    with pytest.raises(ResolveError, match="Netscape"):
        resolver._load_cookie_jar(str(bad))


def test_cookies_reach_both_backends(tmp_path, monkeypatch):
    path = tmp_path / "cookies.txt"
    path.write_text(COOKIES)
    seen = []

    def fake(url, quality, timeout_s, cookies_file=None):
        seen.append(cookies_file)
        raise ResolveError("Sign in to confirm you're not a bot")

    monkeypatch.setattr(resolver, "_resolve_streamlink", fake)
    monkeypatch.setattr(resolver, "_resolve_ytdlp", fake)
    with pytest.raises(ResolveError):
        resolve_stream("https://youtu.be/x", cookies_file=str(path))
    assert seen == [str(path), str(path)]
